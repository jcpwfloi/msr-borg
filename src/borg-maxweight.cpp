#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include "generation.hpp"
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include <cassert>

using namespace std;

string line;
ifstream csv_file;
map <pair<double, double>, size_t> job_type_map;
vector<pair<double, double>> type_resources;

double total_mw_time = 0.0;
long long total_samples = 0;

template <typename T, typename U>
std::pair<T,U> operator+ (const std::pair<T,U> &l,const std::pair<T,U> &r) {   
    return {l.first + r.first, l.second + r.second};
}

template <typename T, typename U>
std::pair<T,U> operator* (const std::pair<T,U> &l,const size_t &r) {   
    return {l.first * r, l.second * r};
}

struct Job {
    double arrival_time, job_size, cpus, memory;
    double last_scheduled = -1;

    size_t type;

    void preempt(double t) {
        last_scheduled = -1;
        job_size -= t - last_scheduled;
    }

    double departure() const {
        return last_scheduled < 0 ? INFINITY : last_scheduled + job_size;
    }

    bool read() {
        last_scheduled = -1;
        csv_file >> arrival_time;
        if (csv_file.eof()) return false;
        csv_file.get();
        csv_file >> job_size; csv_file.get();
        csv_file >> cpus; csv_file.get();
        csv_file >> memory; csv_file.get();
        type = job_type_map[make_pair(cpus, memory)];
        return true;
    }

    void print() {
        cout << "arrival: " << arrival_time << endl;
        cout << "job size: " << job_size << endl;
        cout << "cpus: " << cpus << endl;
        cout << "memory: " << memory << endl;
        cout << "last scheduled: " << last_scheduled << endl;
    }
    
    bool operator < (const Job &b) const {
        return arrival_time < b.arrival_time;
    }
};

double current_time, next_arrival_time, next_departure_time, next_transition_time;
Job next_arrival;
bool more_arrivals;
size_t n_types;
size_t n_candidates;
vector<size_t> u;
vector<deque<Job>> q;
priority_queue<Job, vector<Job>, function<bool(Job, Job)>> in_service;
vector<double> arrival_rates;
vector<double> msrrates;
vector<double> alphas;
vector<Exponential*> msr_trans;
vector<vector<unsigned long>> candidates;
double sum_rt = 0.0;
long long rt_samples = 0;
pair<double, double> resource_in_use;
size_t current_state;
vector<size_t> idx;

bool valid(const vector<size_t> u) {
    pair<double, double> sum = {0, 0};
    for (int i = 0; i < n_types; ++ i) {
        sum = sum + type_resources[i] * u[i];
    }
    return sum.first <= 1.0 && sum.second <= 1.0;
}

void compute_resource_in_use() {
    resource_in_use = {0.0, 0.0};
    for (int i = 0; i < n_types; ++ i) {
        resource_in_use = resource_in_use + type_resources[i] * u[i];
    }
}

bool valid_type(size_t some_type) {
    return resource_in_use.first + type_resources[some_type].first <= 1.0 && 
           resource_in_use.second + type_resources[some_type].second <= 1.0;
}

size_t weight(size_t type) {
    return q[type].size() + u[type];
}

void schedule() {
    vector<Job> stash;
    while (!in_service.empty()) {
        Job j = in_service.top();
        j.job_size -= current_time - j.last_scheduled;
        assert(j.job_size > 0);
        j.last_scheduled = -1;
        // q[j.type].push_front(j);
        stash.push_back(j);
        u[j.type]--;
        in_service.pop();
    }
    sort(stash.begin(), stash.end());
    for (auto j = stash.rbegin(); j != stash.rend(); ++ j) {
        q[j -> type].push_front(*j);
    }
    auto now = std::chrono::system_clock::now(); 
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    auto vars = model.addVars(n_types, GRB_INTEGER);
    GRBLinExpr obj = vars[0] * weight(0);
    for (int i = 1; i < n_types; ++ i) obj += vars[i] * weight(i);
    for (int i = 0; i < n_types; ++ i) {
        model.addConstr(vars[i] >= 0, "nonnegative");
        model.addConstr(vars[i] <= q[i].size() + u[i], "at most queue length");
    }
    GRBLinExpr r1 = vars[0] * type_resources[0].first;
    for (int i = 1; i < n_types; ++ i) r1 += vars[i] * type_resources[i].first;
    GRBLinExpr r2 = vars[0] * type_resources[0].second;
    for (int i = 1; i < n_types; ++ i) r2 += vars[i] * type_resources[i].second;
    model.addConstr(r1 <= 1.0, "capacity 1");
    model.addConstr(r2 <= 1.0, "capacity 2");
    model.setObjective(obj, GRB_MAXIMIZE);
    model.optimize();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now).count();
    total_mw_time += timestamp;
    total_samples ++;
    if (total_samples % 1000 == 0) {
        cout << total_samples << " " << total_mw_time / total_samples << endl;
    }
    // cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    for (int t = 0; t < n_types; ++ t) {
        int new_u = vars[t].get(GRB_DoubleAttr_X);
        while (u[t] < new_u) {
            Job j = q[t].front();
            j.last_scheduled = current_time;
            in_service.push(j);
            q[t].pop_front();
            u[t]++;
        }
    }
    delete[] vars;
}

bool cmp(Job a, Job b) {
    return a.departure() > b.departure();
}

int main() {
    getline(cin, line);
    csv_file = ifstream(line);
    getline(csv_file, line); // get rid of header
    more_arrivals = next_arrival.read();
    next_departure_time = INFINITY;
    cin >> n_types;
    type_resources.resize(n_types, {0.0, 0.0});
    u.resize(n_types, 0);
    in_service = priority_queue<Job, vector<Job>, function<bool(Job, Job)>>(cmp);
    q.resize(n_types);
    arrival_rates.resize(n_types);
    idx.resize(n_types);
    for (int i = 0; i < n_types; ++ i) idx[i] = i;
    for (int i = 0; i < n_types; ++ i) {
        double cpu, memory, arrival_rate;
        cin >> cpu >> memory >> arrival_rate;
        job_type_map[make_pair(cpu, memory)] = i;
        type_resources[i] = make_pair(cpu, memory);
        arrival_rates[i] = arrival_rate;
    }
    while (true) {
        next_arrival_time = INFINITY;
        if (more_arrivals) next_arrival_time = next_arrival.arrival_time;
        if (next_arrival_time == INFINITY && next_departure_time == INFINITY) break;
        if (next_departure_time <= next_arrival_time) {
            // a departure
            current_time = next_departure_time;
            while (!in_service.empty() && in_service.top().departure() <= current_time) {
                Job departing = in_service.top();
                sum_rt += current_time - departing.arrival_time;
                rt_samples += 1;
                // departing.print();
                // cout << departing.arrival_time << endl;
                in_service.pop();
                u[departing.type]--;
            }
            schedule();
        } else {
            // guaranteed an arrival, this also indicates more_arrivals is true
            current_time = next_arrival_time;
            q[next_arrival.type].push_back(next_arrival);
            schedule();
            more_arrivals = next_arrival.read();
        }
        // we should update next_departure_time here
        next_departure_time = INFINITY;
        if (!in_service.empty()) {
            next_departure_time = in_service.top().departure();
        }
    }
    // cout << "result" << endl;
    cout << sum_rt / rt_samples << endl;
    return 0;
}
