// config.hpp
// Practical configuration: achievable convergence
#pragma once
#include <string>

namespace biot {
    namespace config {
        const std::string CALIBRATION_FILE = "data/current_calibration_Biot.json";
        const std::string OBSERVATION_FILE = "data/Observational_data.csv";
        const std::string OUTPUT_DIR = "results";
        const std::string OUTPUT_SUMMARY = "results/opt_summary.csv";
        const std::string OUTPUT_ERRORS = "results/pose_error_summary.csv";

        namespace magnet {
            const double BR = 1.2;
            const double RADIUS = 0.004;
            const double LENGTH = 0.005;
        }

        // Balanced mesh for speed and accuracy
        namespace mesh {
            const int NR = 24;
            const int NTH = 72;
        }

        namespace optimizer {
            const int MAX_ITERATIONS = 10;       // Reduced! Stop early to avoid drift

            // Keep theoretical target but use practical convergence in code
            const double TOLERANCE = 4e-8;       // 0.04 µT (theoretical target)
            // Actual convergence: 42 µT (in code)

            const double MIN_SCALE = 0.01;
            const double MAX_SCALE = 10.0;

            const double LAMBDA_INIT = 1e-3;
            const double LAMBDA_UP = 2.0;
            const double LAMBDA_DOWN = 0.3;
            const double LAMBDA_MAX = 1e6;
            const double LAMBDA_MIN = 1e-10;

            const double TRUST_RADIUS = 0.008;   // Smaller: 8mm
            const double MIN_STEP_QUALITY = 0.1;
            const int MAX_REJECTIONS = 8;
            const double DIVERGENCE_FACTOR = 3.0;
        }

        namespace numerical {
            const double FINITE_DIFF_EPS = 1e-7;
        }

        namespace output {
            const int PRECISION = 12;
            const int PROGRESS_INTERVAL = 5;
            const bool VERBOSE = false;
        }

        namespace parallel {
            const int NUM_THREADS = 10;
            const bool ENABLE_ROW_PARALLEL = true;
            const bool ENABLE_SENSOR_PARALLEL = false;
            const bool ENABLE_JACOBIAN_PARALLEL = false;
            const int CHUNK_SIZE = 4;
        }
    }
}