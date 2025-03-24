#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <map>
#include "generation.hpp"

using namespace std;

string line;
ifstream csv_file;
map <pair<double, double>, size_t> job_type_map;
vector<pair<double, double>> type_resources;
size_t epoch = 0;
const size_t MAX_EPOCH = 200;
const double TIME_EPOCH = 2679600.0;
double sum_rt = 0.0;
long long rt_samples = 0;
double last_sum_rt = 0.0;
long long last_rt_samples = 0;

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
        arrival_time += epoch * TIME_EPOCH;
        if (csv_file.eof()) {
            epoch++;
            cout << (sum_rt - last_sum_rt) / (rt_samples - last_rt_samples) << endl;
            last_sum_rt = sum_rt;
            last_rt_samples = rt_samples;
            if (epoch > MAX_EPOCH) return false;
            else {
                csv_file.clear();
                csv_file.seekg(0);
                getline(csv_file, line);
                return read();
            }
        }
        csv_file.get();
        csv_file >> job_size; csv_file.get();
        csv_file >> cpus; csv_file.get();
        csv_file >> memory; csv_file.get();
        type = job_type_map[make_pair(cpus, memory)];
        if (job_size > 2e5) return read();
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
pair<double, double> resource_in_use;
size_t current_state;
vector<size_t> idx;
bool switching = false;

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
    compute_resource_in_use();
    return resource_in_use.first + type_resources[some_type].first <= 1.0 && 
           resource_in_use.second + type_resources[some_type].second <= 1.0;
}

bool cmp_idx(size_t a, size_t b) {
    double arr_a = q[a].empty() ? INFINITY : q[a].front().arrival_time;
    double arr_b = q[b].empty() ? INFINITY : q[b].front().arrival_time;
    return arr_a < arr_b;
}

void schedule() {
    for (int i = 0; i < n_types; ++ i) {
        while (!q[i].empty() && u[i] < candidates[current_state][i] && valid_type(i)) {
            Job j = q[i].front();
            j.last_scheduled = current_time;
            in_service.push(j);
            q[i].pop_front();
            u[i]++;
        }
    }
    if (switching) {
        bool flag = true;
        for (int i = 0; i < n_types; ++ i) {
            if (u[i] > candidates[current_state][i]) {
                flag = false;
                break;
            }
        }
        if (flag) switching = false;
    }
    // if (!switching) {
    //     bool flag;
    //     while (true) {
    //         flag = false;
    //         compute_resource_in_use();
    //         sort(idx.begin(), idx.end(), cmp_idx);
    //         for (int i = 0; i < n_types; ++ i) {
    //             if (!q[idx[i]].empty() && valid_type(idx[i])) {
    //                 Job j = q[idx[i]].front();
    //                 j.last_scheduled = current_time;
    //                 in_service.push(j);
    //                 q[idx[i]].pop_front();
    //                 u[idx[i]]++;
    //                 flag = true;
    //                 break;
    //             }
    //         }
    //         if (!flag) break;
    //     }
    // }
}

bool cmp(Job a, Job b) {
    return a.departure() > b.departure();
}

int main() {
    switching = false;
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
    cin >> n_candidates;
    msrrates.resize(n_candidates);
    alphas.resize(n_candidates);
    for (int i = 0; i < n_candidates; ++ i) {
        cin >> alphas[i];
    }
    DiscreteDistribution initial_state(alphas);
    current_state = initial_state.get();
    for (int i = 0; i < n_candidates; ++ i) {
        cin >> msrrates[i];
        msr_trans.push_back(new Exponential(msrrates[i]));
    }
    candidates.resize(n_candidates);
    for (int i = 0; i < n_candidates; ++ i) {
        candidates[i].resize(n_types);
        for (int j = 0; j < n_types; ++ j) {
            cin >> candidates[i][j];
        }
    }
    while (true) {
        next_arrival_time = INFINITY;
        if (more_arrivals) next_arrival_time = next_arrival.arrival_time;
        if (next_arrival_time == INFINITY && next_departure_time == INFINITY) break;
        next_transition_time = current_time + msr_trans[current_state] -> get();
        if (!switching && next_transition_time < next_arrival_time && next_transition_time < next_departure_time) {
            // a state transition
            switching = true;
            current_time = next_transition_time;
            current_state = (current_state + 1) % n_candidates;
            schedule();
        } else if (next_departure_time <= next_arrival_time) {
            // a departure
            current_time = next_departure_time;
            while (!in_service.empty() && in_service.top().departure() <= current_time) {
                Job departing = in_service.top();
                sum_rt += current_time - departing.arrival_time;
                rt_samples += 1;
                // departing.print();
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
    cout << sum_rt / rt_samples << endl;
    return 0;
}
