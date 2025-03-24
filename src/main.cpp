#include "generation.hpp"
#include <cmath>
#include <queue>
#include <iostream>
#include <algorithm>
#include <tuple>

long long steps = 1e5;

const int n_max = 200;
double lambda, mu = 1.0;
double alpha[2];

double current_time = 0.0;

int msr_state = 0;
int *schedules;

long long Q = 0;
double sum_Q;
long long samples = 0;

double dtmc[n_max][n_max];
double dtmc_row_rate[n_max];
double dtmc_out_rate;
long long num[n_max];

int n = 2;

std::tuple <int, double> next_event(std::vector<double> p) {
    double total_rate = 0;
    for (auto i : p) total_rate += i;
    Exponential next_t = {total_rate};
    DiscreteDistribution which_e = {p};
    return {which_e.get(), next_t.get()};
}

void simulate(long long steps) {
    for (int i = 0; i < n; ++ i) {
        dtmc_row_rate[i] = 0;
        for (int j = 0; j < n; ++ j) {
            dtmc_row_rate[i] += dtmc[i][j];
        }
    }
    std::vector<double> p;
    DiscreteDistribution next_e {{}};
    for (long long step = 0; step < steps;) {
        // iterate next event: {arrival, transition, departure}
        long long num_running = std::min(num[msr_state], Q);
        p = {lambda, dtmc_row_rate[msr_state], mu * num_running};
        auto [which_e, elapsed] = next_event(p);
        switch (which_e) {
            case 0:
                Q++;
                if (step > 0.1 * steps) {
                    sum_Q += Q;
                    samples++;
                    if (step % (steps / 100) == 0) {
                        std::cout << sum_Q / samples << std::endl;
                    }
                }
            break;
            case 1:
                // p = std::vector<double>(dtmc[msr_state], dtmc[msr_state] + n);
                // next_e.set(p);
                next_e.set(dtmc[msr_state], dtmc[msr_state] + n);
                msr_state = next_e.get();
            break;
            case 2:
                if (Q > 0) {
                    step ++;
                    Q--;
                }
            break;
            default:
            break;
        }
    }
}

int main() {
    int type_of_program = 0;
    std::cin >> type_of_program;
    std::cout << "E[Q]" << std::endl;

    switch (type_of_program) {
        case 2:
        std::cin >> n >> lambda;
        for (int i = 0; i < n; ++ i)
            for (int j = 0; j < n; ++ j)
                std::cin >> dtmc[i][j];
        for (int i = 0; i < n; ++ i) std::cin >> num[i];
        std::cin >> steps;

        simulate(steps);
        break;
        default:
        std::cin >> lambda >> alpha[0] >> alpha[1] >> num[0] >> num[1] >> steps;

        DiscreteDistribution initial_state = {{alpha[1] / (alpha[0] + alpha[1]), alpha[0] / (alpha[0] + alpha[1])}};
        msr_state = initial_state.get();

        dtmc[0][0] = dtmc[1][1] = 0.0;
        dtmc[0][1] = alpha[0];
        dtmc[1][0] = alpha[1];
        
        simulate(steps);
    }

    std::cout << sum_Q / samples << std::endl;
    return 0;
}