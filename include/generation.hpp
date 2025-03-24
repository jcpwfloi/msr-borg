#pragma once
#include <cmath>
#include <fstream>
#include <math.h>
#include <numeric>
#include <random>

class RandomGenerator {
protected:
  std::random_device rd;
  std::default_random_engine generator;
  RandomGenerator(unsigned long seed = 0) {
    if (!seed)
      seed = rd();
    generator = std::default_random_engine(seed);
  }
};

class InverseTransform : public RandomGenerator {
protected:
  std::uniform_real_distribution<double> uniform_distribution{0.0, 1.0};
  double uniform() { return uniform_distribution(generator); }

public:
  InverseTransform(unsigned long seed = 0) : RandomGenerator(seed) {}
  virtual double get() = 0;
};

class Exponential : public RandomGenerator {
private:
  std::exponential_distribution<double> distribution;

public:
  Exponential(double lambda, unsigned long seed = 0)
      : RandomGenerator(seed), distribution{lambda} {}
  double get() { return distribution(generator); }
  void set(double lambda) {
    distribution = std::exponential_distribution<double>(lambda);
  }
};

class Normal : public RandomGenerator {
private:
  std::normal_distribution<double> distribution;

public:
  Normal(double mean, double delta, unsigned long seed = 0)
      : RandomGenerator(seed), distribution{mean, delta} {}
  double get() { return distribution(generator); }
};

class IntegerTruncatedNormal : public Normal {
private:
  int l, r;

public:
  IntegerTruncatedNormal(double mean = 0.0, double delta = 0.0, int l = 0,
                         int r = 0, unsigned long seed = 0)
      : Normal(mean, delta, seed), l{l}, r{r} {}
  int get() {
    int result = round(Normal::get());
    while (result < l || result > r) {
      result = round(Normal::get());
    }
    return result;
  }
};

class Zipf : public RandomGenerator {
private:
  std::discrete_distribution<> distribution;

public:
  Zipf(int N, double alpha = 2.0, unsigned long seed = 0)
      : RandomGenerator(seed) {
    std::vector<double> prob;
    for (int i = 0; i < N; ++i)
      prob.push_back(pow(i + 1, -alpha));
    distribution = std::discrete_distribution<>(prob.begin(), prob.end());
  }
  int get() { return distribution(generator); }
};

class BoundedPareto : public InverseTransform {
  double L, H, alpha;

public:
  BoundedPareto(double L, double H, double alpha, unsigned long seed = 0)
      : InverseTransform(seed), L(L), H(H), alpha(alpha) {}
  double get();
};

class Pareto : public InverseTransform {
  double alpha;

public:
  Pareto(double alpha, unsigned long seed = 0)
      : InverseTransform(seed), alpha(alpha) {}
  double get();
};

class PoissonProcess : public Exponential {
  double last_arrival = 0.0;

public:
  PoissonProcess(double lambda, unsigned long seed = 0)
      : Exponential{lambda, seed} {}
  double get() { return last_arrival += Exponential::get(); }
};

class DiscreteDistribution : public RandomGenerator {
  std::discrete_distribution<size_t> dist;
  std::vector<double> weight;

public:
  DiscreteDistribution(std::vector<double> weight, unsigned long seed = 0)
      : RandomGenerator(seed), weight(weight) {
        dist = std::discrete_distribution<size_t>(weight.begin(), weight.end());
      }

  size_t get() { return dist(generator); }
  void set(std::vector<double> weight) {
    dist = std::discrete_distribution<size_t>(weight.begin(), weight.end());
  }
  void set(double* beg, double* end) {
    dist = std::discrete_distribution<size_t>(beg, end);
  }
};

class SequenceGenerator : RandomGenerator {
  PoissonProcess *A;
  std::discrete_distribution<size_t> dist;

public:
  SequenceGenerator(std::vector<double> lambdas, unsigned long seed = 0)
      : RandomGenerator(seed), dist{lambdas.begin(), lambdas.end()} {
    double lambda = std::reduce(lambdas.begin(), lambdas.end());
    A = new PoissonProcess(lambda);
  }

  std::pair<double, int> get() { return {A->get(), dist(generator)}; }
};