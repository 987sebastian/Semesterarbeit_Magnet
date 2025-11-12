// benchmark_parallel.cpp
// Multi-core performance benchmark for Biot-Savart tracking

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "geometry.hpp"
#include "biot_field.hpp"
#include "pose_optimizer.hpp"
#include "io_utils.hpp"
#include "config.hpp"

using namespace biot;

struct BenchmarkResult {
    int num_threads;
    int total_rows;
    int converged;
    double total_time_s;
    double avg_time_ms;
    double throughput_hz;
    double speedup;
    double efficiency;
};

BenchmarkResult run_benchmark(int num_threads, const CylinderGeom& geom, const DiscGrid& disc,
    const CalibrationData& cal_data, const ObservationData& obs_data) {
    BenchmarkResult result;
    result.num_threads = num_threads;
    result.total_rows = static_cast<int>(obs_data.mag_positions.size());
    result.converged = 0;

#ifdef USE_OPENMP
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
#endif

    auto start_time = std::chrono::high_resolution_clock::now();

    int local_converged = 0;

#ifdef USE_OPENMP
#pragma omp parallel reduction(+:local_converged)
    {
        PoseOptimizer opt(geom, disc, config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE);

#pragma omp for schedule(dynamic, config::parallel::CHUNK_SIZE)
        for (int row = 0; row < result.total_rows; ++row) {
            Eigen::Vector3d p_ref = obs_data.mag_positions[row];
            Eigen::Vector3d u_ref = obs_data.mag_directions[row];
            if (u_ref.norm() > 0) u_ref.normalize();
            else u_ref << 0, 0, 1;

            std::vector<Eigen::Vector3d> sens_cal, B_cal;
            apply_calibration(obs_data.sensor_positions[row], obs_data.B_measured[row],
                cal_data, sens_cal, B_cal);

            PoseResult res = opt.estimate_pose(sens_cal, B_cal, p_ref, u_ref, 1.0);

            if (res.converged) {
                local_converged++;
            }
        }
    }
    result.converged = local_converged;
#else
    PoseOptimizer opt(geom, disc, config::optimizer::MAX_ITERATIONS,
        config::optimizer::TOLERANCE);

    for (int row = 0; row < result.total_rows; ++row) {
        Eigen::Vector3d p_ref = obs_data.mag_positions[row];
        Eigen::Vector3d u_ref = obs_data.mag_directions[row];
        if (u_ref.norm() > 0) u_ref.normalize();
        else u_ref << 0, 0, 1;

        std::vector<Eigen::Vector3d> sens_cal, B_cal;
        apply_calibration(obs_data.sensor_positions[row], obs_data.B_measured[row],
            cal_data, sens_cal, B_cal);

        PoseResult res = opt.optimize(sens_cal, B_cal, p_ref, u_ref, 1.0);

        if (res.converged) {
            result.converged++;
        }
    }
#endif

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    result.total_time_s = duration.count() / 1000.0;
    result.avg_time_ms = (duration.count() * 1.0) / result.total_rows;
    result.throughput_hz = result.total_rows / result.total_time_s;

    return result;
}

void save_benchmark_results(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to save: " << filename << std::endl;
        return;
    }

    ofs << "num_threads,total_rows,converged,total_time_s,avg_time_ms,throughput_hz,speedup,efficiency\n";

    for (const auto& r : results) {
        ofs << r.num_threads << ","
            << r.total_rows << ","
            << r.converged << ","
            << std::fixed << std::setprecision(3) << r.total_time_s << ","
            << std::setprecision(3) << r.avg_time_ms << ","
            << std::setprecision(2) << r.throughput_hz << ","
            << std::setprecision(3) << r.speedup << ","
            << std::setprecision(3) << r.efficiency << "\n";
    }

    ofs.close();
    std::cout << "[OK] Benchmark results saved to: " << filename << std::endl;
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "  Multi-Core Performance Benchmark\n";
    std::cout << "  Biot-Savart Magnetic Tracking\n";
    std::cout << "============================================\n\n";

#ifdef USE_OPENMP
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP: Enabled\n";
    std::cout << "Max available threads: " << max_threads << "\n\n";
#else
    std::cout << "OpenMP: NOT AVAILABLE\n";
    std::cout << "Single-threaded mode only\n\n";
    return -1;
#endif

    CylinderGeom geom(1.2, 0.004, 0.005);
    std::cout << "Magnet: Br=" << geom.Br << " T, R=" << geom.R * 1000
        << " mm, L=" << geom.L * 1000 << " mm\n";

    CalibrationData cal_data;
    if (!load_calibration(config::CALIBRATION_FILE, cal_data)) {
        std::cerr << "[ERROR] Failed to load calibration!" << std::endl;
        return -1;
    }
    std::cout << "[OK] Calibration: " << cal_data.transformations.size() << " sensors\n";

    ObservationData obs_data;
    if (!load_observations(config::OBSERVATION_FILE, obs_data)) {
        std::cerr << "[ERROR] Failed to load observations!" << std::endl;
        return -1;
    }
    std::cout << "[OK] Observations: " << obs_data.mag_positions.size() << " rows\n";

    DiscGrid disc(config::mesh::NR, config::mesh::NTH);
    std::cout << "Mesh: Nr=" << disc.Nr << ", Nth=" << disc.Nth
        << " (total: " << disc.Nr * disc.Nth << " points)\n\n";

    std::vector<int> thread_counts;

    int max_test_threads = (std::min)(16, max_threads);
    for (int i = 1; i <= max_test_threads; ++i) {
        thread_counts.push_back(i);
    }

    std::cout << "Testing thread counts: ";
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        std::cout << thread_counts[i];
        if (i < thread_counts.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    std::cout << "============================================\n";
    std::cout << "  Running Benchmarks\n";
    std::cout << "============================================\n\n";

    std::vector<BenchmarkResult> results;
    double baseline_time = 0.0;

    for (size_t idx = 0; idx < thread_counts.size(); ++idx) {
        int num_threads = thread_counts[idx];
        std::cout << "Testing with " << num_threads << " thread(s)... " << std::flush;

        BenchmarkResult result = run_benchmark(num_threads, geom, disc, cal_data, obs_data);

        if (num_threads == 1) {
            baseline_time = result.total_time_s;
            result.speedup = 1.0;
            result.efficiency = 1.0;
        }
        else {
            result.speedup = baseline_time / result.total_time_s;
            result.efficiency = result.speedup / num_threads;
        }

        results.push_back(result);

        std::cout << "Done! "
            << "Time: " << std::fixed << std::setprecision(1) << result.total_time_s << " s, "
            << "Throughput: " << std::setprecision(1) << result.throughput_hz << " Hz, "
            << "Speedup: " << std::setprecision(2) << result.speedup << "x\n";
    }

    std::cout << "\n============================================\n";
    std::cout << "  Benchmark Results Summary\n";
    std::cout << "============================================\n\n";

    std::cout << std::left << std::setw(10) << "Threads"
        << std::setw(12) << "Time (s)"
        << std::setw(15) << "Avg (ms)"
        << std::setw(15) << "Throughput"
        << std::setw(12) << "Speedup"
        << std::setw(12) << "Efficiency" << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(10) << r.num_threads
            << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_s
            << std::setw(15) << std::setprecision(3) << r.avg_time_ms
            << std::setw(15) << std::setprecision(1) << r.throughput_hz << " Hz"
            << std::setw(12) << std::setprecision(2) << r.speedup << "x"
            << std::setw(12) << std::setprecision(1) << (r.efficiency * 100) << "%\n";
    }

    std::cout << std::string(76, '-') << "\n\n";

    auto optimal = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.throughput_hz < b.throughput_hz;
        });

    std::cout << "Optimal configuration:\n";
    std::cout << "  Threads: " << optimal->num_threads << "\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << optimal->throughput_hz << " Hz\n";
    std::cout << "  Speedup: " << std::setprecision(2) << optimal->speedup << "x over single-threaded\n";
    std::cout << "  Efficiency: " << std::setprecision(1) << (optimal->efficiency * 100) << "%\n\n";

    std::cout << "Scaling Analysis:\n";
    if (results.size() >= 2) {
        double speedup_2 = results[1].speedup;
        double speedup_4 = results.size() >= 4 ? results[3].speedup : 0;
        double speedup_8 = results.size() >= 8 ? results[7].speedup : 0;

        std::cout << "  2 threads:  " << std::fixed << std::setprecision(2) << speedup_2 << "x speedup "
            << "(" << std::setprecision(0) << (speedup_2 / 2.0 * 100) << "% efficient)\n";

        if (speedup_4 > 0) {
            std::cout << "  4 threads:  " << std::fixed << std::setprecision(2) << speedup_4 << "x speedup "
                << "(" << std::setprecision(0) << (speedup_4 / 4.0 * 100) << "% efficient)\n";
        }

        if (speedup_8 > 0) {
            std::cout << "  8 threads:  " << std::fixed << std::setprecision(2) << speedup_8 << "x speedup "
                << "(" << std::setprecision(0) << (speedup_8 / 8.0 * 100) << "% efficient)\n";
        }
    }

    std::cout << "\n";

    save_benchmark_results(results, "results/benchmark_parallel.csv");

    std::cout << "============================================\n";
    std::cout << "  Benchmark Complete!\n";
    std::cout << "============================================\n\n";

    return 0;
}