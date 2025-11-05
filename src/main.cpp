// main.cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

#ifdef _WIN32
#include <Windows.h>
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
using namespace biot::config;

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "============================================" << std::endl;
    std::cout << "  Biot-Savart Pose Optimization / 位姿优化  " << std::endl;
    std::cout << "  Analytical Jacobian + OpenMP / 解析雅可比+并行" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // OpenMP configuration / OpenMP配置
#ifdef USE_OPENMP
    int num_threads = config::parallel::NUM_THREADS;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);

    std::cout << "OpenMP parallel computing / OpenMP并行计算:" << std::endl;
    std::cout << "  Max threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "  Using threads: " << num_threads << std::endl;
    std::cout << "  Analytical Jacobian: ENABLED / 解析雅可比:已启用" << std::endl;
    std::cout << std::endl;
#else
    std::cout << "OpenMP: NOT AVAILABLE (sequential mode)" << std::endl;
    std::cout << "Analytical Jacobian: ENABLED / 解析雅可比:已启用" << std::endl;
    std::cout << std::endl;
#endif

    // Start timer / 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // Define magnet geometry / 定义磁体几何
    CylinderGeom geom(1.2, 0.004, 0.005);
    std::cout << "Magnet geometry / 磁体几何参数:" << std::endl;
    std::cout << "  Remanence (Br) = " << geom.Br << " T" << std::endl;
    std::cout << "  Radius (R) = " << geom.R * 1000 << " mm" << std::endl;
    std::cout << "  Length (L) = " << geom.L * 1000 << " mm\n" << std::endl;

    // Load calibration / 加载标定
    CalibrationData cal_data;
    std::string cal_path = "data/current_calibration_Biot.json";
    std::cout << "Loading calibration from: " << cal_path << std::endl;
    if (!load_calibration(cal_path, cal_data)) {
        std::cerr << "[ERROR] Failed to load calibration!" << std::endl;
        return -1;
    }
    std::cout << std::endl;

    // Load observations / 加载观测数据
    ObservationData obs_data;
    std::string obs_path = "data/Observational_data.csv";
    std::cout << "Loading observations from: " << obs_path << std::endl;
    if (!load_observations(obs_path, obs_data)) {
        std::cerr << "[ERROR] Failed to load observations!" << std::endl;
        return -1;
    }
    std::cout << std::endl;

    const int total_rows = static_cast<int>(obs_data.mag_positions.size());

    // Create results directory / 创建结果目录
#ifdef _WIN32
    _mkdir("results");
#else
    mkdir("results", 0755);
#endif

    // Create optimizer / 创建优化器
    DiscGrid disc(16, 48);
    std::cout << "Discretization / 离散化参数:" << std::endl;
    std::cout << "  Radial divisions (Nr) = " << disc.Nr << std::endl;
    std::cout << "  Angular divisions (Nth) = " << disc.Nth << std::endl;
    std::cout << "  Total mesh points per disk = " << disc.Nr * disc.Nth << "\n" << std::endl;

    std::cout << "Optimizer parameters / 优化器参数:" << std::endl;
    std::cout << "  Max iterations = " << config::optimizer::MAX_ITERATIONS << std::endl;
    std::cout << "  Tolerance = " << config::optimizer::TOLERANCE << " T" << std::endl;
    std::cout << std::endl;

    std::cout << "============================================" << std::endl;
    std::cout << "Starting optimization / 开始优化..." << std::endl;
    std::cout << "Processing " << total_rows << " observation rows" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // Storage for results / 存储结果
    std::vector<int> row_indices(total_rows);
    std::vector<Eigen::Vector3d> positions(total_rows);
    std::vector<Eigen::Vector3d> directions(total_rows);
    std::vector<double> scales(total_rows);
    std::vector<double> pos_errors(total_rows);
    std::vector<double> dir_errors(total_rows);
    std::vector<double> rmse_values(total_rows);
    std::vector<int> nfev_values(total_rows);
    std::vector<int> status_values(total_rows);

    // Process counters / 处理计数器
    int success_count = 0;
    int failure_count = 0;
    std::vector<int> status_histogram(10, 0);

    // Progress tracking / 进度跟踪
    auto last_print_time = std::chrono::high_resolution_clock::now();
    const double print_interval_sec = 2.0;

    // Single frame performance test / 单帧性能测试
    std::cout << "=== Performance Test: Single Frame ===" << std::endl;
    {
        PoseOptimizer test_optimizer(geom, disc,
            config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE);

        int row = 0;
        Eigen::Vector3d p_ref = obs_data.mag_positions[row];
        Eigen::Vector3d u_ref = obs_data.mag_directions[row];
        if (u_ref.norm() > 0) u_ref.normalize();
        else u_ref << 0, 0, 1;

        const auto& sensors = obs_data.sensor_positions[row];
        const auto& B_meas = obs_data.B_measured[row];

        std::vector<Eigen::Vector3d> sensors_cal, B_meas_cal;
        apply_calibration(sensors, B_meas, cal_data, sensors_cal, B_meas_cal);

        auto frame_start = std::chrono::high_resolution_clock::now();
        PoseResult result = test_optimizer.estimate_pose(
            sensors_cal, B_meas_cal, p_ref, u_ref, 1.0
        );
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        double frame_time_ms = frame_duration.count() / 1000.0;

        std::cout << "Single frame time: " << std::fixed << std::setprecision(2)
            << frame_time_ms << " ms" << std::endl;
        std::cout << "RMSE: " << result.rmse * 1e6 << " uT" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << std::endl;
    }

    // Batch Processing / 批量处理
#ifdef USE_OPENMP
    if (config::parallel::ENABLE_ROW_PARALLEL) {
#pragma omp parallel
        {
            PoseOptimizer local_optimizer(geom, disc,
                config::optimizer::MAX_ITERATIONS,
                config::optimizer::TOLERANCE);

#pragma omp for schedule(dynamic, config::parallel::CHUNK_SIZE) nowait
            for (int row = 0; row < total_rows; ++row) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - last_print_time).count();
#pragma omp critical
                {
                    if (elapsed >= print_interval_sec) {
                        int completed = row + 1;
                        double progress = 100.0 * completed / total_rows;
                        std::cout << "\rProgress: " << completed << "/" << total_rows
                            << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                            << std::flush;
                        last_print_time = now;
                    }
                }

                Eigen::Vector3d p_ref = obs_data.mag_positions[row];
                Eigen::Vector3d u_ref = obs_data.mag_directions[row];
                if (u_ref.norm() > 0) u_ref.normalize();
                else u_ref << 0, 0, 1;

                const auto& sensors = obs_data.sensor_positions[row];
                const auto& B_meas = obs_data.B_measured[row];

                std::vector<Eigen::Vector3d> sensors_cal, B_meas_cal;
                apply_calibration(sensors, B_meas, cal_data, sensors_cal, B_meas_cal);

                PoseResult result = local_optimizer.estimate_pose(
                    sensors_cal, B_meas_cal, p_ref, u_ref, 1.0
                );

                double pos_err = PoseOptimizer::compute_position_error(result.position, p_ref);
                double dir_err = PoseOptimizer::compute_direction_error_deg(result.direction, u_ref);

                row_indices[row] = row;
                positions[row] = result.position;
                directions[row] = result.direction;
                scales[row] = result.scale;
                pos_errors[row] = pos_err;
                dir_errors[row] = dir_err;
                rmse_values[row] = result.rmse;
                nfev_values[row] = result.iterations;
                status_values[row] = result.status;

#pragma omp atomic
                if (result.converged) success_count++;
                else failure_count++;

#pragma omp critical
                {
                    int hist_idx = result.status + 2;
                    if (hist_idx >= 0 && hist_idx < 10) {
                        status_histogram[hist_idx]++;
                    }
                }
            }
        }
    }
    else {
#endif
        // Sequential processing / 串行处理
        PoseOptimizer optimizer(geom, disc,
            config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE);

        for (int row = 0; row < total_rows; ++row) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_print_time).count();
            if (elapsed >= print_interval_sec) {
                int completed = row + 1;
                double progress = 100.0 * completed / total_rows;
                std::cout << "\rProgress: " << completed << "/" << total_rows
                    << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                    << std::flush;
                last_print_time = now;
            }

            Eigen::Vector3d p_ref = obs_data.mag_positions[row];
            Eigen::Vector3d u_ref = obs_data.mag_directions[row];
            if (u_ref.norm() > 0) u_ref.normalize();
            else u_ref << 0, 0, 1;

            const auto& sensors = obs_data.sensor_positions[row];
            const auto& B_meas = obs_data.B_measured[row];

            std::vector<Eigen::Vector3d> sensors_cal, B_meas_cal;
            apply_calibration(sensors, B_meas, cal_data, sensors_cal, B_meas_cal);

            PoseResult result = optimizer.estimate_pose(
                sensors_cal, B_meas_cal, p_ref, u_ref, 1.0
            );

            double pos_err = PoseOptimizer::compute_position_error(result.position, p_ref);
            double dir_err = PoseOptimizer::compute_direction_error_deg(result.direction, u_ref);

            row_indices[row] = row;
            positions[row] = result.position;
            directions[row] = result.direction;
            scales[row] = result.scale;
            pos_errors[row] = pos_err;
            dir_errors[row] = dir_err;
            rmse_values[row] = result.rmse;
            nfev_values[row] = result.iterations;
            status_values[row] = result.status;

            if (result.converged) success_count++;
            else failure_count++;

            int hist_idx = result.status + 2;
            if (hist_idx >= 0 && hist_idx < 10) {
                status_histogram[hist_idx]++;
            }
        }
#ifdef USE_OPENMP
    }
#endif

    std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;

    // End timer / 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_seconds = duration.count() / 1000.0;

    std::cout << "\n============================================" << std::endl;
    std::cout << "Optimization complete / 优化完成！" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // Save results / 保存结果
    std::string output_path = "results/opt_summary.csv";
    std::string error_path = "results/pose_error_summary.csv";

    save_results(output_path, row_indices, positions, directions, scales, rmse_values, nfev_values, status_values, "biot");
    save_error_summary(error_path, row_indices, pos_errors, dir_errors);

    // Summary statistics / 统计摘要
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary Statistics / 统计摘要" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  Total rows processed: " << total_rows << std::endl;
    std::cout << "  Successful optimizations: " << success_count << std::endl;
    std::cout << "  Failed to converge: " << failure_count << std::endl;
    std::cout << "  Total elapsed time: " << std::fixed << std::setprecision(2)
        << total_seconds << " seconds" << std::endl;

    // Status distribution / 状态分布
    std::cout << "\n--- Status Distribution / 状态分布 ---" << std::endl;
    std::cout << "  status = -2 (Failed): " << status_histogram[0] << std::endl;
    std::cout << "  status = -1 (Matrix fail): " << status_histogram[1] << std::endl;
    std::cout << "  status =  0 (Not converged): " << status_histogram[2] << std::endl;
    std::cout << "  status =  1 (Perfect <0.04 uT): " << status_histogram[3] << std::endl;
    std::cout << "  status =  2 (Excellent <0.4 uT): " << status_histogram[4] << std::endl;
    std::cout << "  status =  3 (Good <4 uT): " << status_histogram[5] << std::endl;
    std::cout << "  status =  4 (Acceptable <20 uT): " << status_histogram[6] << std::endl;

    // Performance metrics / 性能指标
    double avg_time_per_row = total_seconds / total_rows;
    double throughput_hz = 1.0 / avg_time_per_row;
    std::cout << "\n--- Performance / 性能 ---" << std::endl;
    std::cout << "  Average time per row: " << std::fixed << std::setprecision(1)
        << avg_time_per_row * 1000 << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << throughput_hz << " Hz" << std::endl;

#ifdef USE_OPENMP
    std::cout << "  Parallel threads used: " << num_threads << std::endl;
#endif

    // Error statistics / 误差统计
    if (!pos_errors.empty()) {
        double mean_pos_err = 0.0, mean_dir_err = 0.0, mean_rmse = 0.0;
        double max_pos_err = pos_errors[0], max_dir_err = dir_errors[0];
        double min_pos_err = pos_errors[0], min_dir_err = dir_errors[0];

        for (size_t i = 0; i < pos_errors.size(); ++i) {
            mean_pos_err += pos_errors[i];
            mean_dir_err += dir_errors[i];
            mean_rmse += rmse_values[i];
            if (pos_errors[i] > max_pos_err) max_pos_err = pos_errors[i];
            if (pos_errors[i] < min_pos_err) min_pos_err = pos_errors[i];
            if (dir_errors[i] > max_dir_err) max_dir_err = dir_errors[i];
            if (dir_errors[i] < min_dir_err) min_dir_err = dir_errors[i];
        }

        mean_pos_err /= pos_errors.size();
        mean_dir_err /= dir_errors.size();
        mean_rmse /= rmse_values.size();

        std::cout << "\n--- Position Errors / 位置误差 ---" << std::endl;
        std::cout << "  Mean: " << mean_pos_err * 1000 << " mm" << std::endl;
        std::cout << "  Min:  " << min_pos_err * 1000 << " mm" << std::endl;
        std::cout << "  Max:  " << max_pos_err * 1000 << " mm" << std::endl;

        std::cout << "\n--- Direction Errors / 方向误差 ---" << std::endl;
        std::cout << "  Mean: " << mean_dir_err << " deg" << std::endl;
        std::cout << "  Min:  " << min_dir_err << " deg" << std::endl;
        std::cout << "  Max:  " << max_dir_err << " deg" << std::endl;

        std::cout << "\n--- RMSE / 均方根误差 ---" << std::endl;
        std::cout << "  Mean: " << mean_rmse * 1e6 << " uT" << std::endl;
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "Program finished / 程序完成" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}