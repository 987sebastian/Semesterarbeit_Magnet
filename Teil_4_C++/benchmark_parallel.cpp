// benchmark_parallel.cpp
// Multi-core performance benchmark for Biot-Savart tracking

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#define NOMINMAX
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
#include "cpu_topology.hpp"

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

        PoseResult res = opt.estimate_pose(sens_cal, B_cal, p_ref, u_ref, 1.0);

        if (res.converged) {
            local_converged++;
        }
    }
#endif

    result.converged = local_converged;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_s = std::chrono::duration<double>(end_time - start_time).count();
    result.avg_time_ms = (result.total_time_s * 1000.0) / result.total_rows;
    result.throughput_hz = result.total_rows / result.total_time_s;

    return result;
}

// Performance scaling analysis / 性能扩展分析
void analyze_scaling(const std::vector<BenchmarkResult>& results, const CPUTopology& cpu_topo) {
    if (results.size() < 2) return;

    std::cout << "\n============================================\n";
    std::cout << "  PERFORMANCE SCALING ANALYSIS\n";
    std::cout << "============================================\n\n";

    // Baseline performance / 基准性能
    double baseline_time = results[0].total_time_s;

    // Find 6-thread and 12-thread results / 查找6线程和12线程结果
    const BenchmarkResult* six_thread = nullptr;
    const BenchmarkResult* twelve_thread = nullptr;

    for (const auto& r : results) {
        if (r.num_threads == 6) six_thread = &r;
        if (r.num_threads == 12) twelve_thread = &r;
    }

    // Amdahl's Law analysis / Amdahl定律分析
    if (six_thread) {
        double actual_speedup_6 = baseline_time / six_thread->total_time_s;
        // Solve for parallel fraction: S = 1 / ((1-p) + p/n)
        // Where S=speedup, p=parallel fraction, n=cores
        double p = (actual_speedup_6 - 1.0) * 6.0 / (actual_speedup_6 * (6.0 - 1.0));
        double theoretical_max = 1.0 / (1.0 - p);

        std::cout << "Amdahl's Law Estimation:\n";
        std::cout << "  Parallel fraction:     " << std::fixed << std::setprecision(1)
            << p * 100.0 << "%\n";
        std::cout << "  Serial fraction:       " << (1.0 - p) * 100.0 << "%\n";
        std::cout << "  Theoretical max speedup: " << std::setprecision(1)
            << theoretical_max << "x\n\n";
    }

    // Physical cores efficiency / 物理核心效率
    std::cout << "Core Architecture Impact:\n";

    if (six_thread) {
        std::cout << "  6 threads (6 P-cores, no HT):\n";
        std::cout << "    Throughput: " << std::setprecision(1) << six_thread->throughput_hz << " Hz\n";
        std::cout << "    Efficiency: " << six_thread->efficiency * 100.0 << "%\n";
        std::cout << "    Per-core:   " << six_thread->throughput_hz / 6.0 << " Hz/thread\n\n";
    }

    if (twelve_thread) {
        std::cout << "  12 threads (6 P-cores with HT):\n";
        std::cout << "    Throughput: " << std::setprecision(1) << twelve_thread->throughput_hz << " Hz\n";
        std::cout << "    Efficiency: " << twelve_thread->efficiency * 100.0 << "%\n";
        std::cout << "    Per-core:   " << twelve_thread->throughput_hz / 12.0 << " Hz/thread\n\n";

        if (six_thread) {
            double ht_overhead = (twelve_thread->throughput_hz / 12.0) / (six_thread->throughput_hz / 6.0);
            std::cout << "  Hyper-Threading efficiency: " << std::setprecision(1)
                << ht_overhead * 100.0 << "%\n";
            std::cout << "  (Per-thread throughput with HT vs without HT)\n\n";
        }
    }

    // Memory bandwidth analysis / 内存带宽分析
    std::cout << "Memory Bandwidth Analysis:\n";
    // Estimate data transfer per sample / 估算每样本数据传输
    // Grid points: 16×48×3×2 coordinates (plus, minus faces)
    // Each coordinate is double (8 bytes)
    size_t grid_size = 16 * 48 * 3 * 2 * sizeof(double);
    // Sensor data: 41 sensors × 3 coordinates × 2 (position + field)
    size_t sensor_size = 41 * 3 * 2 * sizeof(double);
    size_t total_bytes = grid_size + sensor_size;

    if (six_thread) {
        double bandwidth_6 = (total_bytes * six_thread->throughput_hz) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  6 threads:  " << std::setprecision(2) << bandwidth_6 << " GB/s\n";
    }

    if (twelve_thread) {
        double bandwidth_12 = (total_bytes * twelve_thread->throughput_hz) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  12 threads: " << std::setprecision(2) << bandwidth_12 << " GB/s\n";

        if (six_thread) {
            double bandwidth_scaling = bandwidth_12 / (total_bytes * six_thread->throughput_hz / (1024.0 * 1024.0 * 1024.0));
            std::cout << "  Bandwidth scaling (12/6): " << std::setprecision(2)
                << bandwidth_scaling << "x\n";
            std::cout << "  (< 2.0x suggests memory bottleneck)\n";
        }
    }

    std::cout << "\nNote: Grid size = " << grid_size / 1024 << " KB, "
        << "Sensor data = " << sensor_size << " bytes\n";
}

void save_benchmark_results(const std::string& filename, const std::vector<BenchmarkResult>& results) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to save: " << filename << std::endl;
        return;
    }

    ofs << "num_threads,core_type,total_rows,converged,total_time_s,avg_time_ms,throughput_hz,speedup,efficiency\n";

    for (const auto& r : results) {
        // Annotate core type / 标注核心类型
        std::string core_type;
        if (r.num_threads <= 6) {
            core_type = "P-core";
        }
        else {
            core_type = "P-core+HT";
        }

        ofs << r.num_threads << ","
            << core_type << ","
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
    // Detect CPU topology / 检测CPU拓扑
    CPUTopology cpu_topo = detect_cpu_topology();
    print_cpu_topology(cpu_topo);
    std::cout << "\n";

    int max_threads = cpu_topo.performance_cores_threads;

    std::cout << "Benchmark Configuration:\n";
    std::cout << "  Tested range:  1 to " << max_threads << " threads\n";
    std::cout << "  Core type:     P-cores only (E-cores disabled)\n";
    std::cout << "  Scheduling:    Dynamic with chunk size " << config::parallel::CHUNK_SIZE << "\n";

#ifdef _WIN32
    // Bind to performance cores / 绑定到性能核心
    if (cpu_topo.has_hybrid_architecture) {
        DWORD_PTR process_mask = 0;
        for (int i = 0; i < max_threads; ++i) {
            process_mask |= (1ULL << i);
        }
        SetProcessAffinityMask(GetCurrentProcess(), process_mask);
        std::cout << "  CPU affinity:  Bound to P-cores (0-" << (max_threads - 1) << ")\n";
    }
#endif
    std::cout << "\n";
#else
    std::cout << "OpenMP: NOT AVAILABLE\n";
    std::cout << "Single-threaded mode only\n\n";
    return -1;
#endif

    CylinderGeom geom(config::magnet::BR, config::magnet::RADIUS, config::magnet::LENGTH);
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

    int max_test_threads = (std::min)(max_threads, max_threads);
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
        << std::setw(15) << "Core Type"
        << std::setw(12) << "Time (s)"
        << std::setw(15) << "Avg (ms)"
        << std::setw(15) << "Throughput"
        << std::setw(12) << "Speedup"
        << std::setw(12) << "Efficiency" << "\n";
    std::cout << std::string(91, '-') << "\n";

    for (const auto& r : results) {
        std::string core_type = (r.num_threads <= 6) ? "P-core" : "P-core+HT";

        std::cout << std::left << std::setw(10) << r.num_threads
            << std::setw(15) << core_type
            << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_s
            << std::setw(15) << std::setprecision(3) << r.avg_time_ms
            << std::setw(15) << std::setprecision(1) << r.throughput_hz << " Hz"
            << std::setw(12) << std::setprecision(2) << r.speedup << "x"
            << std::setw(12) << std::setprecision(1) << (r.efficiency * 100) << "%\n";
    }

    std::cout << std::string(91, '-') << "\n\n";

    auto optimal = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.throughput_hz < b.throughput_hz;
        });

    auto best_efficiency = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.efficiency < b.efficiency;
        });

    std::cout << "Optimal Throughput:\n";
    std::cout << "  Threads: " << optimal->num_threads << "\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << optimal->throughput_hz << " Hz\n";
    std::cout << "  Speedup: " << std::setprecision(2) << optimal->speedup << "x\n";
    std::cout << "  Efficiency: " << std::setprecision(1) << (optimal->efficiency * 100) << "%\n\n";

    std::cout << "Best Efficiency:\n";
    std::cout << "  Threads: " << best_efficiency->num_threads << "\n";
    std::cout << "  Efficiency: " << std::setprecision(1) << (best_efficiency->efficiency * 100) << "%\n";
    std::cout << "  Throughput: " << std::setprecision(1) << best_efficiency->throughput_hz << " Hz\n\n";

    // Perform scaling analysis / 执行性能扩展分析
    analyze_scaling(results, cpu_topo);

    // Save results / 保存结果
    save_benchmark_results("results/benchmark_parallel.csv", results);

    std::cout << "\n[OK] Benchmark complete!\n";
    return 0;
}