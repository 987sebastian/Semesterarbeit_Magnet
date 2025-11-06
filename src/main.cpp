// main.cpp
#include <iostream>
#include <Windows.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <atomic>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "geometry.hpp"
#include "biot_field.hpp"
#include "io_utils.hpp"
#include "pose_optimizer.hpp"
#include "config.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace biot;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "============================================" << std::endl;
    std::cout << "  Biot-Savart Pose Optimization" << std::endl;
    std::cout << "  Analytical Jacobian + 4e-8T tolerance" << std::endl;
    std::cout << "============================================\n" << std::endl;

#ifdef USE_OPENMP
    int num_threads = config::parallel::NUM_THREADS;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
    std::cout << "OpenMP: " << num_threads << " threads" << std::endl;
#else
    std::cout << "OpenMP: NOT AVAILABLE" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();

    CylinderGeom geom(1.2, 0.004, 0.005);
    std::cout << "Magnet: Br=" << geom.Br << " T, R=" << geom.R * 1000
        << " mm, L=" << geom.L * 1000 << " mm\n" << std::endl;

    CalibrationData cal_data;
    if (!load_calibration("data/current_calibration_Biot.json", cal_data)) {
        std::cerr << "[ERROR] Failed to load calibration!" << std::endl;
        return -1;
    }
    std::cout << "[OK] Calibration: " << cal_data.transformations.size() << " sensors\n" << std::endl;

    ObservationData obs_data;
    if (!load_observations("data/Observational_data.csv", obs_data)) {
        std::cerr << "[ERROR] Failed to load observations!" << std::endl;
        return -1;
    }
    std::cout << "[OK] Observations: " << obs_data.mag_positions.size() << " rows\n" << std::endl;

    const int total_rows = static_cast<int>(obs_data.mag_positions.size());

#ifdef _WIN32
    _mkdir("results");
#else
    mkdir("results", 0755);
#endif

    DiscGrid disc(config::mesh::NR, config::mesh::NTH);
    std::cout << "Discretization: Nr=" << disc.Nr << ", Nth=" << disc.Nth
        << " (total: " << disc.Nr * disc.Nth << " points)" << std::endl;
    std::cout << "Optimizer: max_iter=" << config::optimizer::MAX_ITERATIONS
        << ", tol=" << config::optimizer::TOLERANCE << " T\n" << std::endl;

    std::cout << "Processing " << total_rows << " rows...\n" << std::endl;

    std::vector<int> row_indices(total_rows);
    std::vector<Eigen::Vector3d> positions(total_rows);
    std::vector<Eigen::Vector3d> directions(total_rows);
    std::vector<double> scales(total_rows);
    std::vector<double> pos_errors(total_rows);
    std::vector<double> dir_errors(total_rows);
    std::vector<double> rmse_values(total_rows);
    std::vector<int> nfev_values(total_rows);
    std::vector<int> status_values(total_rows);

    int success_count = 0;

    // Use atomic counter for proper progress tracking
    std::atomic<int> completed_count(0);
    auto last_print = std::chrono::high_resolution_clock::now();

#ifdef USE_OPENMP
    if (config::parallel::ENABLE_ROW_PARALLEL) {
#pragma omp parallel reduction(+:success_count)
        {
            PoseOptimizer opt(geom, disc, config::optimizer::MAX_ITERATIONS,
                config::optimizer::TOLERANCE);

#pragma omp for schedule(dynamic, config::parallel::CHUNK_SIZE)
            for (int row = 0; row < total_rows; ++row) {
                Eigen::Vector3d p_ref = obs_data.mag_positions[row];
                Eigen::Vector3d u_ref = obs_data.mag_directions[row];
                if (u_ref.norm() > 0) u_ref.normalize();
                else u_ref << 0, 0, 1;

                std::vector<Eigen::Vector3d> sens_cal, B_cal;
                apply_calibration(obs_data.sensor_positions[row], obs_data.B_measured[row],
                    cal_data, sens_cal, B_cal);

                PoseResult res = opt.estimate_pose(sens_cal, B_cal, p_ref, u_ref, 1.0);

                row_indices[row] = row;
                positions[row] = res.position;
                directions[row] = res.direction;
                scales[row] = res.scale;
                rmse_values[row] = res.rmse;
                nfev_values[row] = res.iterations;
                status_values[row] = res.status;
                pos_errors[row] = (res.position - p_ref).norm();
                dir_errors[row] = PoseOptimizer::compute_direction_error_deg(res.direction, u_ref);

                if (res.converged) success_count++;

                // Atomic progress update
                int current = ++completed_count;

                // Print progress every 2 seconds using master thread only
                if (omp_get_thread_num() == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration<double>(now - last_print).count() >= 2.0) {
                        double progress = 100.0 * current / total_rows;
                        std::cout << "\rProgress: " << current << "/" << total_rows
                            << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                            << std::flush;
                        last_print = now;
                    }
                }
            }
        }
    }
    else {
#endif
        PoseOptimizer opt(geom, disc, config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE);

        for (int row = 0; row < total_rows; ++row) {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() >= 2.0) {
                double progress = 100.0 * (row + 1) / total_rows;
                std::cout << "\rProgress: " << (row + 1) << "/" << total_rows
                    << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                    << std::flush;
                last_print = now;
            }

            Eigen::Vector3d p_ref = obs_data.mag_positions[row];
            Eigen::Vector3d u_ref = obs_data.mag_directions[row];
            if (u_ref.norm() > 0) u_ref.normalize();
            else u_ref << 0, 0, 1;

            std::vector<Eigen::Vector3d> sens_cal, B_cal;
            apply_calibration(obs_data.sensor_positions[row], obs_data.B_measured[row],
                cal_data, sens_cal, B_cal);

            PoseResult res = opt.estimate_pose(sens_cal, B_cal, p_ref, u_ref, 1.0);

            row_indices[row] = row;
            positions[row] = res.position;
            directions[row] = res.direction;
            scales[row] = res.scale;
            rmse_values[row] = res.rmse;
            nfev_values[row] = res.iterations;
            status_values[row] = res.status;
            pos_errors[row] = (res.position - p_ref).norm();
            dir_errors[row] = PoseOptimizer::compute_direction_error_deg(res.direction, u_ref);

            if (res.converged) success_count++;
        }
#ifdef USE_OPENMP
    }
#endif

    std::cout << "\r" << std::string(60, ' ') << "\r";

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n============================================" << std::endl;
    std::cout << "Complete!" << std::endl;
    std::cout << "============================================\n" << std::endl;

    save_results("results/opt_summary.csv", row_indices, positions, directions, scales,
        rmse_values, nfev_values, status_values, "biot");
    save_error_summary("results/pose_error_summary.csv", row_indices, pos_errors, dir_errors);

    // Statistics
    double mean_pos = 0.0, mean_dir = 0.0, max_pos = 0.0, max_dir = 0.0;
    double total_iters = 0.0, mean_rmse = 0.0;

    for (size_t i = 0; i < pos_errors.size(); ++i) {
        mean_pos += pos_errors[i];
        mean_dir += dir_errors[i];
        mean_rmse += rmse_values[i];
        total_iters += nfev_values[i];
        if (pos_errors[i] > max_pos) max_pos = pos_errors[i];
        if (dir_errors[i] > max_dir) max_dir = dir_errors[i];
    }
    mean_pos /= total_rows;
    mean_dir /= total_rows;
    mean_rmse /= total_rows;
    double avg_iters = total_iters / total_rows;

    std::cout << "Total: " << total_rows << ", Converged: " << success_count
        << " (" << std::fixed << std::setprecision(1)
        << (100.0 * success_count / total_rows) << "%)" << std::endl;
    std::cout << "Time: " << duration.count() << " s, Throughput: "
        << std::setprecision(1) << (total_rows / (double)duration.count()) << " Hz" << std::endl;
    std::cout << "\nAverage iterations: " << std::setprecision(1) << avg_iters << std::endl;
    std::cout << "Average RMSE: " << std::scientific << std::setprecision(2)
        << mean_rmse << " T (" << (mean_rmse * 1e6) << " uT)" << std::endl;
    std::cout << "\nPosition error: mean=" << std::fixed << std::setprecision(1)
        << mean_pos * 1000 << " mm, max=" << max_pos * 1000 << " mm" << std::endl;
    std::cout << "Direction error: mean=" << mean_dir << " deg, max="
        << max_dir << " deg" << std::endl;
    std::cout << "\n============================================" << std::endl;

    return 0;
}