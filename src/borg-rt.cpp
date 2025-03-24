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
#include <cassert>

using namespace std;

string line, csv_file_name;
ifstream csv_file;
map <pair<double, double>, size_t> job_type_map;
vector<pair<double, double>> type_resources;
size_t epoch = 0;
const size_t MAX_EPOCH = 100;
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

double current_time, next_arrival_time, next_departure_time, next_transition_time, total_clock_rate;
Job next_arrival;
bool more_arrivals;
size_t n_types;
size_t n_candidates;
vector<size_t> u, dummy_u;
vector<deque<Job>> q;
priority_queue<Job, vector<Job>, function<bool(Job, Job)>> in_service;
vector<double> arrival_rates;
vector<double> msrrates;
vector<double> alphas, retention, clock_rate;
vector<Exponential*> msr_trans;
vector<vector<unsigned long>> candidates;
pair<double, double> resource_in_use;
size_t current_state;
vector<size_t> idx;
DiscreteDistribution next_retention {{}}, which_tick {{}};

double f(long long x) {
    return pow(log(1.0 + x), 0.7);
}

double w(long long x) {
    size_t qmax = 0;
    for (auto _q : q) qmax = std::max(_q.size(), qmax);
    return std::max(f(x), 1e-4 * f(qmax));
}

bool valid() {
    pair<double, double> sum = {0, 0};
    for (int i = 0; i < n_types; ++ i) {
        sum = sum + type_resources[i] * (u[i] + dummy_u[i]);
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

bool cmp_idx(size_t a, size_t b) {
    double arr_a = q[a].empty() ? INFINITY : q[a].front().arrival_time;
    double arr_b = q[b].empty() ? INFINITY : q[b].front().arrival_time;
    return arr_a < arr_b;
}

bool cmp(Job a, Job b) {
    return a.departure() > b.departure();
}

int main() {
    retention.resize(2, 0);
    getline(cin, line);
    csv_file_name = line;
    csv_file = ifstream(line);
    getline(csv_file, line); // get rid of header
    more_arrivals = next_arrival.read();
    next_departure_time = INFINITY;
    cin >> n_types;
    type_resources.resize(n_types, {0.0, 0.0});
    u.resize(n_types, 0);
    dummy_u.resize(n_types, 0);
    in_service = priority_queue<Job, vector<Job>, function<bool(Job, Job)>>(cmp);
    q.resize(n_types);
    arrival_rates.resize(n_types);
    idx.resize(n_types);
    clock_rate.resize(n_types);
    for (int i = 0; i < n_types; ++ i) idx[i] = i;
    for (int i = 0; i < n_types; ++ i) {
        double cpu, memory, arrival_rate;
        cin >> cpu >> memory >> arrival_rate;
        job_type_map[make_pair(cpu, memory)] = i;
        type_resources[i] = make_pair(cpu, memory);
        arrival_rates[i] = arrival_rate;
    }
    for (int i = 0; i < n_types; ++ i) {
        cin >> clock_rate[i];
    }
    total_clock_rate = std::reduce(clock_rate.begin(), clock_rate.end());
    Exponential next_tick {total_clock_rate};
    which_tick.set(clock_rate);
    while (true) {
        next_arrival_time = INFINITY;
        if (more_arrivals) next_arrival_time = next_arrival.arrival_time;
        if (next_arrival_time == INFINITY && next_departure_time == INFINITY) break;
        next_transition_time = current_time + next_tick.get();
        if (next_transition_time < next_arrival_time && next_transition_time < next_departure_time) {
            // a state transition
            current_time = next_transition_time;
            size_t t = which_tick.get();
            // clock tick for type t
            u[t]++;
            if (valid()) {
                if (!q[t].empty()) {
                    Job j = q[t].front();
                    j.last_scheduled = current_time;
                    in_service.push(j);
                    q[t].pop_front();
                } else {
                    u[t]--;
                    dummy_u[t]++;
                }
            } else u[t]--;
            assert(valid());
        } else if (next_departure_time <= next_arrival_time) {
            // a departure
            current_time = next_departure_time;
            // one departure at a time
            Job departing = in_service.top();
            sum_rt += current_time - departing.arrival_time;
            rt_samples += 1;
            // departing.print();
            in_service.pop();
            u[departing.type]--;

            retention[0] = exp(-w(q[departing.type].size()));
            retention[1] = 1.0 - retention[0];

            next_retention.set(retention);
            if (next_retention.get() == 1) { // retention
                if (!q[departing.type].empty()) {
                    Job j = q[departing.type].front();
                    j.last_scheduled = current_time;
                    in_service.push(j);
                    u[departing.type]++;
                    q[departing.type].pop_front();
                } else dummy_u[departing.type]++;
            }
            assert(valid());
        } else {
            // guaranteed an arrival, this also indicates more_arrivals is true
            current_time = next_arrival_time;
            Job j = next_arrival;
            more_arrivals = next_arrival.read();
            if (dummy_u[j.type]) {
                dummy_u[j.type]--;
                u[j.type]++;
                j.last_scheduled = current_time;
                in_service.push(j);
            } else {
                q[j.type].push_back(j);
            }
            assert(valid());
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
