// main.cpp
// Biot-Savart pose optimization with random noise initialization and CPU topology detection
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <random>

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h min/max macro conflicts
#include <Windows.h>
#include <direct.h>
#endif

#include "geometry.hpp"
#include "biot_field.hpp"
#include "io_utils.hpp"
#include "pose_optimizer.hpp"
#include "config.hpp"
#include "cpu_topology.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace biot;

// Thread-safe random noise generation using C++11
inline Eigen::Vector3d generate_position_noise(int seed, double sigma_mm) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, sigma_mm * 0.001);
    return Eigen::Vector3d(dist(gen), dist(gen), dist(gen));
}

inline Eigen::Vector3d generate_direction_noise(int seed, double sigma_deg) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double angle_rad = dist(gen) * sigma_deg * M_PI / 180.0;
    Eigen::Vector3d axis(dist(gen), dist(gen), dist(gen));
    axis.normalize();

    return axis * angle_rad;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "============================================" << std::endl;
    std::cout << "  Biot-Savart Pose Optimization" << std::endl;
    std::cout << "  Random Noise Initialization + Parallel Processing" << std::endl;
    std::cout << "============================================\n" << std::endl;

#ifdef USE_OPENMP
    // Auto-detect CPU topology
    CPUTopology cpu_topo = detect_cpu_topology();
    print_cpu_topology(cpu_topo);
    std::cout << "\n";

    // Use performance cores threads
    int num_threads = cpu_topo.performance_cores_threads;

    // Allow manual override from config
    if (config::parallel::NUM_THREADS > 0) {
        num_threads = config::parallel::NUM_THREADS;
        std::cout << "[Override] Using user-specified threads: " << num_threads << "\n\n";
    }

    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);

    std::cout << "OpenMP Configuration:\n";
    std::cout << "  Active threads: " << num_threads << "\n";
    std::cout << "  Dynamic adjustment: Disabled\n";

#ifdef _WIN32
    // Bind to performance cores on Windows
    if (cpu_topo.has_hybrid_architecture) {
        std::cout << "  CPU affinity: Binding to P-cores (0-"
            << (cpu_topo.performance_cores_threads - 1) << ")\n";

        DWORD_PTR process_mask = 0;
        for (int i = 0; i < cpu_topo.performance_cores_threads; ++i) {
            process_mask |= (1ULL << i);
        }

        SetProcessAffinityMask(GetCurrentProcess(), process_mask);
    }
#endif
    std::cout << "\n";
#else
    std::cout << "OpenMP: NOT AVAILABLE" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();

    CylinderGeom geom(config::magnet::BR, config::magnet::RADIUS, config::magnet::LENGTH);
    std::cout << "Magnet: Br=" << geom.Br << " T, R=" << geom.R * 1000
        << " mm, L=" << geom.L * 1000 << " mm\n" << std::endl;

    CalibrationData cal_data;
    if (!load_calibration(config::CALIBRATION_FILE, cal_data)) {
        std::cerr << "[ERROR] Failed to load calibration!" << std::endl;
        return -1;
    }
    std::cout << "[OK] Calibration: " << cal_data.transformations.size() << " sensors\n" << std::endl;

    ObservationData obs_data;
    if (!load_observations(config::OBSERVATION_FILE, obs_data)) {
        std::cerr << "[ERROR] Failed to load observations!" << std::endl;
        return -1;
    }
    std::cout << "[OK] Observations: " << obs_data.mag_positions.size() << " rows\n" << std::endl;

    const int total_rows = static_cast<int>(obs_data.mag_positions.size());

    _mkdir(config::OUTPUT_DIR.c_str());

    DiscGrid disc(config::mesh::NR, config::mesh::NTH);
    std::cout << "Mesh: Nr=" << disc.Nr << ", Nth=" << disc.Nth
        << " (total: " << disc.Nr * disc.Nth << " points)\n" << std::endl;

    // Noise parameters for initial guess
    const double pos_noise_mm = 5.0;   
    const double dir_noise_deg = 3.0;   

    std::cout << "Initial Guess Strategy: Ground truth + Gaussian noise\n";
    std::cout << "  Position noise: σ = " << pos_noise_mm << " mm\n";
    std::cout << "  Direction noise: σ = " << dir_noise_deg << " deg\n" << std::endl;

    std::cout << "Processing " << total_rows << " samples...\n" << std::endl;

    std::vector<PoseResult> results(total_rows);
    std::vector<Eigen::Vector3d> position_errors(total_rows);
    std::vector<double> direction_errors(total_rows);
    std::vector<double> processing_times(total_rows);

    int converged_count = 0;

#ifdef USE_OPENMP
#pragma omp parallel reduction(+:converged_count)
#endif
    {
        PoseOptimizer opt(geom, disc, config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE);

#ifdef USE_OPENMP
#pragma omp for schedule(dynamic, config::parallel::CHUNK_SIZE)
#endif
        for (int i = 0; i < total_rows; ++i) {
            auto iter_start = std::chrono::high_resolution_clock::now();

            Eigen::Vector3d p_ref = obs_data.mag_positions[i];
            Eigen::Vector3d u_ref = obs_data.mag_directions[i];
            if (u_ref.norm() > 0) u_ref.normalize();
            else u_ref << 0, 0, 1;

            // Generate thread-safe random seed
#ifdef USE_OPENMP
            int thread_id = omp_get_thread_num();
#else
            int thread_id = 0;
#endif
            int seed = i * 1000 + thread_id;

            // Add position noise: ground truth + Gaussian noise
            Eigen::Vector3d p_init = p_ref + generate_position_noise(seed, pos_noise_mm);

            // Add direction noise: ground truth + small rotation
            Eigen::Vector3d rot_vec = generate_direction_noise(seed + 123, dir_noise_deg);
            double angle = rot_vec.norm();

            Eigen::Vector3d u_init;
            if (angle > 1e-10) {
                Eigen::Vector3d axis = rot_vec.normalized();

                // Rodrigues rotation formula
                u_init = u_ref * std::cos(angle)
                    + axis.cross(u_ref) * std::sin(angle)
                    + axis * (axis.dot(u_ref)) * (1.0 - std::cos(angle));
                u_init.normalize();
            }
            else {
                u_init = u_ref;
            }

            std::vector<Eigen::Vector3d> sensors_cal, B_cal;
            apply_calibration(obs_data.sensor_positions[i], obs_data.B_measured[i],
                cal_data, sensors_cal, B_cal);

            PoseResult res = opt.estimate_pose(sensors_cal, B_cal, p_init, u_init, 1.0);

            auto iter_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();

            results[i] = res;
            position_errors[i] = res.position - p_ref;

            double dot = u_ref.dot(res.direction);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            direction_errors[i] = std::acos(dot) * 180.0 / M_PI;

            processing_times[i] = elapsed_ms;

            if (res.converged) {
                converged_count++;
            }

            if ((i + 1) % config::output::PROGRESS_INTERVAL == 0 || i == total_rows - 1) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    std::cout << "Progress: " << (i + 1) << "/" << total_rows
                        << " (" << std::fixed << std::setprecision(1)
                        << (100.0 * (i + 1) / total_rows) << "%)" << std::endl;
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // Compute statistics
    double sum_pos_error = 0.0, sum_dir_error = 0.0, sum_rmse = 0.0, sum_time = 0.0;
    double max_pos_error = 0.0, max_dir_error = 0.0;
    int total_iterations = 0;

    for (int i = 0; i < total_rows; ++i) {
        double pos_err = position_errors[i].norm();
        sum_pos_error += pos_err;
        sum_dir_error += direction_errors[i];
        sum_rmse += results[i].rmse;
        sum_time += processing_times[i];
        total_iterations += results[i].iterations;

        if (pos_err > max_pos_error) max_pos_error = pos_err;
        if (direction_errors[i] > max_dir_error) max_dir_error = direction_errors[i];
    }

    double mean_pos_error = sum_pos_error / total_rows;
    double mean_dir_error = sum_dir_error / total_rows;
    double mean_rmse = sum_rmse / total_rows;
    double mean_time = sum_time / total_rows;
    double mean_iterations = (double)total_iterations / total_rows;

    // Prepare data for saving
    std::vector<int> row_indices(total_rows);
    std::vector<Eigen::Vector3d> positions(total_rows);
    std::vector<Eigen::Vector3d> directions(total_rows);
    std::vector<double> scales(total_rows);
    std::vector<double> rmse_values(total_rows);
    std::vector<int> nfev_values(total_rows);
    std::vector<int> status_values(total_rows);
    std::vector<double> pos_errors_norm(total_rows);
    std::vector<double> dir_errors_deg(total_rows);

    for (int i = 0; i < total_rows; ++i) {
        row_indices[i] = i;
        positions[i] = results[i].position;
        directions[i] = results[i].direction;
        scales[i] = results[i].scale;
        rmse_values[i] = results[i].rmse;
        nfev_values[i] = results[i].iterations;
        status_values[i] = results[i].converged ? 1 : 0;
        pos_errors_norm[i] = position_errors[i].norm();
        dir_errors_deg[i] = direction_errors[i];
    }

    // Save results
    save_results(config::OUTPUT_SUMMARY, row_indices, positions, directions, scales,
        rmse_values, nfev_values, status_values, "biot");
    save_error_summary(config::OUTPUT_ERRORS, row_indices, pos_errors_norm, dir_errors_deg);

    // Print summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "  OPTIMIZATION SUMMARY" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Total samples:    " << total_rows << std::endl;
    std::cout << "Converged:        " << converged_count << " ("
        << std::fixed << std::setprecision(1)
        << (100.0 * converged_count / total_rows) << "%)" << std::endl;
    std::cout << "Total time:       " << total_duration.count() << " seconds" << std::endl;
    std::cout << "Avg time/sample:  " << std::setprecision(2) << mean_time << " ms" << std::endl;
    std::cout << "Avg iterations:   " << std::setprecision(1) << mean_iterations << std::endl;

    std::cout << "\n============================================" << std::endl;
    std::cout << "  ACCURACY METRICS" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Mean RMSE:        " << mean_rmse << " T ("
        << std::fixed << std::setprecision(2) << mean_rmse * 1e6 << " µT)" << std::endl;

    std::cout << "\nPosition Error:" << std::endl;
    std::cout << "  Mean:  " << std::setprecision(1) << mean_pos_error * 1000 << " mm" << std::endl;
    std::cout << "  Max:   " << max_pos_error * 1000 << " mm" << std::endl;

    std::cout << "\nDirection Error:" << std::endl;
    std::cout << "  Mean:  " << mean_dir_error << " degrees" << std::endl;
    std::cout << "  Max:   " << max_dir_error << " degrees" << std::endl;

    // Real-time performance assessment
    double throughput = 1000.0 / mean_time;
    std::cout << "\n============================================" << std::endl;
    std::cout << "  REAL-TIME PERFORMANCE" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " Hz" << std::endl;

    if (throughput >= 50.0) {
        std::cout << "Status: ✓ Meets >50Hz real-time requirement" << std::endl;
    }
    else {
        std::cout << "Status: ✗ Below 50Hz real-time requirement" << std::endl;
        std::cout << "  (Need " << std::setprecision(1) << (50.0 / throughput)
            << "x speedup)" << std::endl;
    }

    std::cout << "\n[OK] Results saved to:" << std::endl;
    std::cout << "  - " << config::OUTPUT_SUMMARY << std::endl;
    std::cout << "  - " << config::OUTPUT_ERRORS << std::endl;

    return 0;
}
