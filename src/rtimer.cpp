#include "generation.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <ranges>
#include <cmath>
#include <map>

long long steps = 1e5;

std::vector<double> lambda, mu, clock_rate;
std::vector<long long> Q;
std::vector<double> sum_Q;
std::vector<long long> samples;
std::vector<std::vector<long long> > num;
std::map<std::vector<long long>, long long> stats_map;

// bool flag = false;
// std::uniform_real_distribution<double> dist(0,1);
// std::random_device rd;
// std::mt19937 e2(rd());
// double threshold = 0.0;

int n, K;
double total_arrival_rate, total_clock_rate;

template<typename T1, typename T2> void multiply(std::vector<T1> &c, std::vector<T1> a, std::vector<T2> b) {
    assert(a.size() == b.size());
    for (long long i = 0; i < a.size(); ++ i) {
        c[i] = a[i] * b[i];
    }
}

template<typename T1, typename T2> std::vector<T1> operator += (std::vector<T1> &a, std::vector<T2> b) {
    assert(a.size() == b.size());
    for (long long i = 0; i < a.size(); ++ i) {
        a[i] += b[i];
    }
    return a;
}

template <typename T1, typename T2> std::vector<T1> operator / (std::vector<T1> a, std::vector<T2> b) {
    std::vector<T1> results(a);
    for (long long i = 0; i < a.size(); ++ i) {
        results[i] = a[i] / b[i];
    }
    return results;
}

template<typename T> std::ostream& operator << (std::ostream &o, std::vector<T> vec) {
    o << *vec.begin();
    for (auto x : vec | std::views::drop(1)) {
        o << "," << x;
    }
    return o;
}

template<typename T> void element_wise_min(std::vector<T> &c, std::vector<T> a, std::vector<T> b) {
    for (long long i = 0; i < a.size(); ++ i) {
        c[i] = std::min(a[i], b[i]);
    }
}

template<typename T> bool element_wise_at_most(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    for (long long i = 0; i < a.size(); ++ i) {
        if (a[i] > b[i]) return false;
    }
    return true;
}

bool valid_schedule(std::vector<long long> u) {
    for (auto sc : num) {
        if (element_wise_at_most(u, sc)) return true;
    }
    return false;
}

template<typename T> void add(std::vector<T> &a, std::vector<T> b, std::vector<T> c) {
    assert(b.size() == c.size());
    for (int i = 0; i < b.size(); ++ i) {
        a[i] = b[i] + c[i];
    }
}

std::tuple <int, double> next_event(std::vector<double> p) {
    double total_rate = 0;
    for (auto i : p) total_rate += i;
    Exponential next_t = {total_rate};
    DiscreteDistribution which_e = {p};
    return {which_e.get(), next_t.get()};
}

void max_weight(std::vector<long long> &u, std::vector<long long> Q) {
    long long largest = -1, weight;
    std::vector<long long> *result;
    for (auto c : num) {
        weight = std::inner_product(c.begin(), c.end(), Q.begin(), 0ll);
        if (weight > largest) {
            largest = weight;
            result = &c;
        }
    }
    element_wise_min(u, *result, Q);
}

std::vector<double> p, rates;
std::vector<long long> num_running, dummy_running, total_running, in_system;

double f(long long x) {
    return pow(log(1.0 + x), 0.7);
}

double w(long long x) {
    long long qmax = 0;
    for (auto q : Q) qmax = std::max(q, qmax);
    return std::max(f(x), 1e-4 * f(qmax));
}

void simulate(long long steps) {
    DiscreteDistribution next_e {{}};
    DiscreteDistribution next_a {lambda};
    DiscreteDistribution next_d {{}};
    DiscreteDistribution next_p {{}};
    DiscreteDistribution next_retention {{}};

    long long which_arrival = 0;
    long long which_d = 0;
    long long which_e = 0;

    std::vector<double> retention = {0.0, 0.0};

    rates.resize(K, 0.0);
    num_running.resize(K, 0);
    dummy_running.resize(K, 0);
    total_running.resize(K, 0);
    in_system.resize(K, 0);

    for (long long step = 0; step < steps;) {
        // iterate next event: {arrival, transition, departure}
        add(total_running, num_running, dummy_running);
        add(in_system, Q, num_running);
        multiply(rates, mu, total_running);
        //rates = mu * num_running;
        double total_departure_rate = std::reduce(rates.begin(), rates.end());
        double probability = 0.0;
        p = {total_arrival_rate, total_departure_rate, total_clock_rate};
        next_p.set(p);
        which_e = next_p.get();
        switch (which_e) {
            case 0:
                which_arrival = next_a.get();
                if (step > 0.2 * steps) {
                    sum_Q[which_arrival] += in_system[which_arrival];
                    samples[which_arrival]++;
                    stats_map[num_running]++;
                    if (step % (steps / 100) == 0) {
                        std::cout << sum_Q / samples << std::endl;
                    }
                }
                if (dummy_running[which_arrival]) {
                    dummy_running[which_arrival] --;
                    num_running[which_arrival] ++;
                } else Q[which_arrival]++;
                // Potential optimization: replace dummy immediately
            break;
            case 1:
                /*
                 * Randomized Timer:
                 * On departure of type-j job, with probability exp(-w_j(t)), keep dummy/actual type-j job
                 * Otherwise, depart the job and do nothing
                */
                next_d.set(rates);
                which_d = next_d.get();

                // if (flag && !(num_running[0] == 0 && num_running[1] == 0 && num_running[2] == 2)) {
                //     std::cout << "exiting " << step << " " << Q[0] << " " << Q[1] << " " << Q[2] << std::endl;
                //     flag = false;
                // }
                // if (!flag && num_running[0] == 0 && num_running[1] == 0 && num_running[2] == 2) {
                //     std::cout << which_d << std::endl;
                //     std::cout << "entering " << step << " " << Q[0] << " " << Q[1] << " " << Q[2] << " " << exp(-w(Q[which_d])) << std::endl;
                //     flag = true;
                // }

                // a job departure
                retention[0] = dummy_running[which_d];
                // server
                retention[1] = num_running[which_d];
                next_retention.set(retention);
                if (next_retention.get()) { // a server departure
                    num_running[which_d] --;
                    step++;
                } else { // a dummy departure
                    dummy_running[which_d] --;
                }

                retention[0] = exp(-w(Q[which_d]));
                retention[1] = 1.0 - retention[0];

                // threshold = dist(e2);

                next_retention.set(retention);

                if (next_retention.get() == 1) { // retention
                // if (threshold > retention[0]) {
                    if (Q[which_d] > 0) { // if there are jobs in the queue, admit
                        Q[which_d] --;
                        num_running[which_d] ++;
                    } else { // otherwise, put a dummy job into the system
                        dummy_running[which_d] ++;
                    }
                }
            break;
            case 2:
                /*
                 * Randomized Timer:
                 * When clock ticks for type-j job, if it fits, place type-j job into the system.
                */
                next_e.set(clock_rate);
                which_e = next_e.get();

                // try to place type-which_e job into the system
                total_running[which_e] ++;
                if (valid_schedule(total_running)) {
                    if (Q[which_e] > 0) {
                        Q[which_e] --;
                        num_running[which_e] ++;
                    } else {
                        dummy_running[which_e] ++;
                    }
                }
            break;
            default:
            break;
        }
    }
}

int main() {
    std::cin >> n >> K >> steps;

    std::cout << "E[Q1]";
    for (int i = 2; i <= K; ++ i) {
        std::cout << ",E[Q" <<  i << "]";
    }
    std::cout << std::endl;

    lambda.resize(K, 0.0);
    mu.resize(K, 0.0);
    clock_rate.resize(K, 0.0);

    Q.resize(K, 0);
    sum_Q.resize(K, 0.0);
    samples.resize(K, 0);

    for (int i = 0; i < K; ++ i) std::cin >> lambda[i];
    for (int i = 0; i < K; ++ i) std::cin >> mu[i];
    for (int i = 0; i < K; ++ i) std::cin >> clock_rate[i];

    total_clock_rate = std::reduce(clock_rate.begin(), clock_rate.end());
    total_arrival_rate = std::reduce(lambda.begin(), lambda.end());

    num.resize(n, std::vector<long long>(K, 0));

    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < K; ++ j)
            std::cin >> num[i][j];

    simulate(steps);

    std::cout << sum_Q / samples << std::endl;

    // long long total_samples = 0;
    // std::vector<double> mu_star;
    // mu_star.resize(K, 0.0);
    // for (auto const & [key, value]: stats_map) {
    //     for (auto k : key) {
    //         std::cout << k << "/";
    //     }
    //     for (long long i = 0; i < K; ++ i) {
    //         mu_star[i] += double(key[i]) * value;
    //     }
    //     std::cout << " " << value << std::endl;
    //     total_samples += value;
    // }
    // for (long long i = 0; i < K; ++ i) {
    //     std::cout << mu_star[i] / double(total_samples) << "/";
    // }
    return 0;
}
