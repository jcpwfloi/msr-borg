#pragma once
#include "miniconf.h"
#include <exception>
#include <iostream>
#include <fstream>

#define getNumber_(var) var = conf[#var].getNumber()
#define getInt_(var) var = conf[#var].getInt()

class Config {
  miniconf::Config conf;

public:
  double alphal, alphah;
  std::vector<double> lambda;
  std::vector<std::vector<int> > schedules;
  std::vector<std::vector<double> > alpha;
  std::vector<std::vector<int> > mw_schedules;
  int steps = 10000000;
  int n;

  Config() {
    conf.option("maxweight").required(false).defaultValue("").description("MaxWeight config file");
    conf.option("states").shortflag("n").defaultValue(2).required(true).description("Number of MAMS states");
    conf.option("verbose").shortflag("v").defaultValue(false).required(true).description("Verbose output");
    conf.option("steps").defaultValue(10000000).required(true).description("Steps");
  }

  void parse(int argc, char **argv) {
    bool success = conf.parse(argc, argv);

    n = conf["states"].getInt();

    steps = conf["steps"].getInt();

    bool verbose = conf["verbose"].getBoolean();

    if (verbose) conf.print();

    lambda.resize(n);
    schedules.resize(n);
    alpha.resize(n);
    for (int i = 0; i < n; ++ i) {
        schedules[i].resize(n);
        alpha[i].resize(n);
    }

    for (int i = 0; i < n; ++ i)
        std::cin >> lambda[i];
    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < n; ++ j)
            std::cin >> schedules[i][j];
    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < n; ++ j)
            std::cin >> alpha[i][j];

    std::string maxweight_file = conf["maxweight"].getString();

    if (!maxweight_file.empty()) {
        try {
            std::fstream fs;
            fs.open(maxweight_file, std::fstream::in);
            int cur = -1;
            int val;
            while (fs >> val) {
                cur ++;
                mw_schedules.push_back({});
                mw_schedules[cur].resize(n, 0);
                mw_schedules[cur][0] = val;
                for (int i = 1; i < n; ++ i) {
                    fs >> mw_schedules[cur][i];
                }
            }
            fs.close();
        } catch (std::exception e) {
            std::cerr << "MaxWeight file invalid" << std::endl;
        }
    }

    if (verbose) {
        std::cout << "Arrival Vector:\n";
        for (auto i : lambda) std::cout << i << " ";
        std::cout << "\nCandidate Set:\n";
        for (auto i : schedules) {
            for (auto j: i) {
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Transition Rates:\n";
        for (auto i : alpha) {
            for (auto j: i) {
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }
  }
};