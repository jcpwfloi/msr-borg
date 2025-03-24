#include "generation.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <ranges>
#include <queue>

long long steps = 1e5;

std::vector<double> lambda;
std::vector<double> mu;

std::vector<long long> Q;
std::vector<double> sum_Q;
std::vector<long long> samples;

std::vector<std::vector<long long> > num;

int n, K;
double total_arrival_rate;

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


std::tuple <int, double> next_event(std::vector<double> p) {
    double total_rate = 0;
    for (auto i : p) total_rate += i;
    Exponential next_t = {total_rate};
    DiscreteDistribution which_e = {p};
    return {which_e.get(), next_t.get()};
}

void max_weight(std::vector<long long> &u, std::vector<long long> Q) {
    long long largest = -1, weight;
    std::vector<long long> result;
    for (auto c : num) {
        weight = std::inner_product(c.begin(), c.end(), Q.begin(), 0ll);
        if (weight > largest) {
            largest = weight;
            result = c;
        }
    }
    element_wise_min(u, result, Q);
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

void simulate(long long steps) {
    std::vector<double> p, rates;
    std::vector<long long> num_running;
    std::queue<long long> next_in_line;
    DiscreteDistribution next_e {{}};
    DiscreteDistribution next_a {lambda};
    DiscreteDistribution next_d {{}};
    DiscreteDistribution next_p {{}};
    long long which_arrival = 0;
    long long which_d = 0;
    long long which_e = 0;

    rates.resize(K, 0.0);
    num_running.resize(K, 0);

    for (long long step = 0, n_arrival = 0; step < steps;) {
        // iterate next event: {arrival, transition, departure}
        multiply(rates, mu, num_running);
        //rates = mu * num_running;
        double total_departure_rate = std::reduce(rates.begin(), rates.end());
        p = {total_arrival_rate, total_departure_rate};
        next_p.set(p);
        which_e = next_p.get();
        switch (which_e) {
            case 0:
                which_arrival = next_a.get();
                if (step > 0.2 * steps) {
                    sum_Q[which_arrival] += Q[which_arrival] + num_running[which_arrival];
                    samples[which_arrival]++;
                    if (step % (steps / 100) == 0) {
                        std::cout << sum_Q / samples << std::endl;
                    }
                }
                n_arrival ++;
                num_running[which_arrival] ++;
                if (!valid_schedule(num_running)) {
                    num_running[which_arrival] --;
                    Q[which_arrival] ++;
                    next_in_line.push(which_arrival);
                }
            break;
            case 1:
                next_d.set(rates);
                which_d = next_d.get();
                if (num_running[which_d] > 0) {
                    step ++;
                    num_running[which_d] --;
                }
                while (!next_in_line.empty()) {
                    which_arrival = next_in_line.front();
                    num_running[which_arrival] ++;
                    if (valid_schedule(num_running)) {
                        Q[which_arrival] --;
                        next_in_line.pop();
                    } else {
                        num_running[which_arrival]--;
                        break;
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

    Q.resize(K, 0);
    sum_Q.resize(K, 0.0);
    samples.resize(K, 0);

    for (int i = 0; i < K; ++ i) std::cin >> lambda[i];
    for (int i = 0; i < K; ++ i) std::cin >> mu[i];

    total_arrival_rate = std::reduce(lambda.begin(), lambda.end());

    num.resize(n, std::vector<long long>(K, 0));

    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < K; ++ j)
            std::cin >> num[i][j];

    simulate(steps);

    std::cout << sum_Q / samples << std::endl;
    return 0;
}
