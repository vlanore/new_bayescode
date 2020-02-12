#pragma once

#include "ChainComponent.hpp"
#include "bayes_utils/src/logging.hpp"

class ConsoleLogger : public ChainComponent {
    logger_t logger{stdout_logger("chain_logger")};
    bool verbose;

  public:

    ConsoleLogger(bool v = true) : verbose(v) {}
    void start() override { logger->info("Started"); }
    void move(int i) override { if (verbose) {logger->info("Move {}", i);} }
    void savepoint(int i) override { if (verbose) {logger->info("Savepoint {}", i);} }
    void end() override { logger->info("Ended"); }
};
