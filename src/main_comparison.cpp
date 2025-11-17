// main_comparison.cpp
// Model comparison with Earth calibration
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <Windows.h>
#include <direct.h>
#include <cstdlib>

#include "geometry.hpp"
#include "io_utils.hpp"
#include "model_adapters.hpp"
#include "pose_optimizer_generic.hpp"
#include "config.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace biot;
using namespace adapters;
using namespace generic;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void save_detailed_results(
    const std::string& filename,
    const std::vector<int>& row_indices,
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Vector3d>& directions,
    const std::vector<double>& scales,
    const std::vector<double>& rmse_values,
    const std::vector<int>& status_values,
    const std::vector<double>& pos_errors,
    const std::vector<double>& dir_errors,
    const std::vector<double>& time_ms
) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return;
    }

    ofs << std::fixed << std::setprecision(12);
    ofs << "row_index,pos_x,pos_y,pos_z,dir_x,dir_y,dir_z,scale,rmse,status,pos_error,dir_error,time_ms\n";

    for (size_t i = 0; i < row_indices.size(); ++i) {
        ofs << row_indices[i] << ","
            << positions[i](0) << "," << positions[i](1) << "," << positions[i](2) << ","
            << directions[i](0) << "," << directions[i](1) << "," << directions[i](2) << ","
            << scales[i] << "," << rmse_values[i] << ","
            << status_values[i] << ","
            << pos_errors[i] << "," << dir_errors[i] << "," << time_ms[i] << "\n";
    }

    ofs.close();
    std::cout << "[OK] Saved: " << filename << std::endl;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "============================================" << std::endl;
    std::cout << "  Biot-Savart vs Dipole Comparison" << std::endl;
    std::cout << "  Earth Calibration + Corrected Test Data" << std::endl;
    std::cout << "============================================\n" << std::endl;

#ifdef USE_OPENMP
    int num_threads = config::parallel::NUM_THREADS;
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
    std::cout << "OpenMP: " << num_threads << " threads (Performance cores only)" << std::endl;
    std::cout << "Note: Using 6 performance cores as specified\n" << std::endl;
#endif

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
    std::cout << "Discretization: Nr=" << disc.Nr << ", Nth=" << disc.Nth
        << " (total: " << disc.Nr * disc.Nth << " points)\n" << std::endl;

    // =================================================================
    // BIOT-SAVART MODEL / BIOT-SAVART模型
    // =================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing BIOT-SAVART Model" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<int> biot_row_indices(total_rows);
    std::vector<Eigen::Vector3d> biot_positions(total_rows);
    std::vector<Eigen::Vector3d> biot_directions(total_rows);
    std::vector<double> biot_scales(total_rows);
    std::vector<double> biot_rmse(total_rows);
    std::vector<int> biot_status(total_rows);
    std::vector<double> biot_pos_errors(total_rows);
    std::vector<double> biot_dir_errors(total_rows);
    std::vector<double> biot_time_ms(total_rows);

    auto biot_start = std::chrono::high_resolution_clock::now();
    int biot_converged = 0;

#ifdef USE_OPENMP
#pragma omp parallel reduction(+:biot_converged)
#endif
    {
        BiotSavartAdapter biot_adapter(geom, disc);
        GenericPoseOptimizer optimizer(
            &biot_adapter,
            config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE
        );

#ifdef USE_OPENMP
#pragma omp for schedule(dynamic, config::parallel::CHUNK_SIZE)
#endif
        for (int row = 0; row < total_rows; ++row) {
            auto row_start = std::chrono::high_resolution_clock::now();

            Eigen::Vector3d p_ref = obs_data.mag_positions[row];
            Eigen::Vector3d u_ref = obs_data.mag_directions[row];
            // Add noise (thread-safe random numbers) / 添加噪声(线程安全的随机数)
            unsigned int seed = row + omp_get_thread_num() * 1000;
            std::srand(seed);

            Eigen::Vector3d p_init = p_ref + Eigen::Vector3d::Random() * 0.01;  // ±10mm
            Eigen::Vector3d u_init = u_ref + Eigen::Vector3d::Random() * 0.1;
            u_init.normalize();
            double scale_init = 1.0;
            if (u_ref.norm() > 0) u_ref.normalize();
            else u_ref << 0, 0, 1;

            std::vector<Eigen::Vector3d> sens_cal, B_cal;
            apply_calibration(obs_data.sensor_positions[row], obs_data.B_measured[row],
                cal_data, sens_cal, B_cal);

            PoseResult res = optimizer.estimate_pose(sens_cal, B_cal, p_ref, u_ref, 1.0);

            auto row_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(row_end - row_start).count();

            biot_row_indices[row] = row;
            biot_positions[row] = res.position;
            biot_directions[row] = res.direction;
            biot_scales[row] = res.scale;
            biot_rmse[row] = res.rmse;
            biot_status[row] = res.converged ? 1 : 0;
            biot_time_ms[row] = elapsed_ms;

            if (res.converged) biot_converged++;

            Eigen::Vector3d pos_err = res.position - p_ref;
            biot_pos_errors[row] = pos_err.norm();

            double dot = u_ref.dot(res.direction);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            biot_dir_errors[row] = std::acos(dot) * 180.0 / M_PI;
        }
    }

    auto biot_end = std::chrono::high_resolution_clock::now();
    auto biot_duration = std::chrono::duration_cast<std::chrono::seconds>(biot_end - biot_start);

    save_detailed_results("results/biot_savart_results.csv",
        biot_row_indices, biot_positions, biot_directions, biot_scales,
        biot_rmse, biot_status, biot_pos_errors, biot_dir_errors, biot_time_ms);

    // =================================================================
    // DIPOLE MODEL / DIPOLE模型
    // =================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing DIPOLE Model" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<int> dipole_row_indices(total_rows);
    std::vector<Eigen::Vector3d> dipole_positions(total_rows);
    std::vector<Eigen::Vector3d> dipole_directions(total_rows);
    std::vector<double> dipole_scales(total_rows);
    std::vector<double> dipole_rmse(total_rows);
    std::vector<int> dipole_status(total_rows);
    std::vector<double> dipole_pos_errors(total_rows);
    std::vector<double> dipole_dir_errors(total_rows);
    std::vector<double> dipole_time_ms(total_rows);

    auto dipole_start = std::chrono::high_resolution_clock::now();
    int dipole_converged = 0;

#ifdef USE_OPENMP
#pragma omp parallel reduction(+:dipole_converged)
#endif
    {
        DipoleAdapter dipole_adapter(geom);
        GenericPoseOptimizer optimizer(
            &dipole_adapter,
            config::optimizer::MAX_ITERATIONS,
            config::optimizer::TOLERANCE
        );

#ifdef USE_OPENMP
#pragma omp for schedule(dynamic, config::parallel::CHUNK_SIZE)
#endif
        for (int row = 0; row < total_rows; ++row) {
            auto row_start = std::chrono::high_resolution_clock::now();

            Eigen::Vector3d p_ref = obs_data.mag_positions[row];
            Eigen::Vector3d u_ref = obs_data.mag_directions[row];
            if (u_ref.norm() > 0) u_ref.normalize();
            else u_ref << 0, 0, 1;

            std::vector<Eigen::Vector3d> sens_cal, B_cal;
            apply_calibration(obs_data.sensor_positions[row], obs_data.B_measured[row],
                cal_data, sens_cal, B_cal);

            PoseResult res = optimizer.estimate_pose(sens_cal, B_cal, p_ref, u_ref, 1.0);

            auto row_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(row_end - row_start).count();

            dipole_row_indices[row] = row;
            dipole_positions[row] = res.position;
            dipole_directions[row] = res.direction;
            dipole_scales[row] = res.scale;
            dipole_rmse[row] = res.rmse;
            dipole_status[row] = res.converged ? 1 : 0;
            dipole_time_ms[row] = elapsed_ms;

            if (res.converged) dipole_converged++;

            Eigen::Vector3d pos_err = res.position - p_ref;
            dipole_pos_errors[row] = pos_err.norm();

            double dot = u_ref.dot(res.direction);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            dipole_dir_errors[row] = std::acos(dot) * 180.0 / M_PI;
        }
    }

    auto dipole_end = std::chrono::high_resolution_clock::now();
    auto dipole_duration = std::chrono::duration_cast<std::chrono::seconds>(dipole_end - dipole_start);

    save_detailed_results("results/dipole_results.csv",
        dipole_row_indices, dipole_positions, dipole_directions, dipole_scales,
        dipole_rmse, dipole_status, dipole_pos_errors, dipole_dir_errors, dipole_time_ms);

    // =================================================================
    // SUMMARY / 总结
    // =================================================================
    std::cout << "\n============================================" << std::endl;
    std::cout << "  COMPARISON SUMMARY" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // Compute statistics / 计算统计数据
    double biot_mean_pos = 0.0, biot_mean_dir = 0.0, biot_max_pos = 0.0, biot_max_dir = 0.0;
    double biot_mean_rmse = 0.0, biot_mean_time = 0.0;
    for (int i = 0; i < total_rows; ++i) {
        biot_mean_pos += biot_pos_errors[i];
        biot_mean_dir += biot_dir_errors[i];
        biot_mean_rmse += biot_rmse[i];
        biot_mean_time += biot_time_ms[i];
        if (biot_pos_errors[i] > biot_max_pos) biot_max_pos = biot_pos_errors[i];
        if (biot_dir_errors[i] > biot_max_dir) biot_max_dir = biot_dir_errors[i];
    }
    biot_mean_pos /= total_rows;
    biot_mean_dir /= total_rows;
    biot_mean_rmse /= total_rows;
    biot_mean_time /= total_rows;

    double dipole_mean_pos = 0.0, dipole_mean_dir = 0.0, dipole_max_pos = 0.0, dipole_max_dir = 0.0;
    double dipole_mean_rmse = 0.0, dipole_mean_time = 0.0;
    for (int i = 0; i < total_rows; ++i) {
        dipole_mean_pos += dipole_pos_errors[i];
        dipole_mean_dir += dipole_dir_errors[i];
        dipole_mean_rmse += dipole_rmse[i];
        dipole_mean_time += dipole_time_ms[i];
        if (dipole_pos_errors[i] > dipole_max_pos) dipole_max_pos = dipole_pos_errors[i];
        if (dipole_dir_errors[i] > dipole_max_dir) dipole_max_dir = dipole_dir_errors[i];
    }
    dipole_mean_pos /= total_rows;
    dipole_mean_dir /= total_rows;
    dipole_mean_rmse /= total_rows;
    dipole_mean_time /= total_rows;

    std::cout << "BIOT-SAVART:" << std::endl;
    std::cout << "  Converged: " << biot_converged << "/" << total_rows
        << " (" << std::fixed << std::setprecision(1) << (100.0 * biot_converged / total_rows) << "%)" << std::endl;
    std::cout << "  Time: " << biot_duration.count() << " s, Avg: "
        << std::setprecision(2) << biot_mean_time << " ms" << std::endl;
    std::cout << "  RMSE: " << std::scientific << std::setprecision(2) << biot_mean_rmse
        << " T (" << (biot_mean_rmse * 1e6) << " µT)" << std::endl;
    std::cout << "  Position: mean=" << std::fixed << std::setprecision(1) << biot_mean_pos * 1000
        << " mm, max=" << biot_max_pos * 1000 << " mm" << std::endl;
    std::cout << "  Direction: mean=" << biot_mean_dir << " deg, max=" << biot_max_dir << " deg\n" << std::endl;

    std::cout << "DIPOLE:" << std::endl;
    std::cout << "  Converged: " << dipole_converged << "/" << total_rows
        << " (" << std::fixed << std::setprecision(1) << (100.0 * dipole_converged / total_rows) << "%)" << std::endl;
    std::cout << "  Time: " << dipole_duration.count() << " s, Avg: "
        << std::setprecision(2) << dipole_mean_time << " ms" << std::endl;
    std::cout << "  RMSE: " << std::scientific << std::setprecision(2) << dipole_mean_rmse
        << " T (" << (dipole_mean_rmse * 1e6) << " µT)" << std::endl;
    std::cout << "  Position: mean=" << std::fixed << std::setprecision(1) << dipole_mean_pos * 1000
        << " mm, max=" << dipole_max_pos * 1000 << " mm" << std::endl;
    std::cout << "  Direction: mean=" << dipole_mean_dir << " deg, max=" << dipole_max_dir << " deg\n" << std::endl;

    std::cout << "SPEEDUP: Dipole is " << std::setprecision(1) << (biot_mean_time / dipole_mean_time) << "x faster" << std::endl;
    std::cout << "============================================" << std::endl;

    // Save text summary / 保存文本摘要
    std::ofstream summary("results/model_comparison_summary.txt");
    if (summary.is_open()) {
        summary << "Model Comparison Summary (Earth Calibration)\n";
        summary << "=============================================\n\n";
        summary << "BIOT-SAVART:\n  Converged: " << biot_converged << "/" << total_rows << "\n";
        summary << "  Time: " << biot_duration.count() << " s\n  Avg time: " << biot_mean_time << " ms\n";
        summary << "  RMSE: " << biot_mean_rmse << " T (" << (biot_mean_rmse * 1e6) << " µT)\n";
        summary << "  Position: " << biot_mean_pos * 1000 << " mm (max " << biot_max_pos * 1000 << " mm)\n";
        summary << "  Direction: " << biot_mean_dir << " deg (max " << biot_max_dir << " deg)\n\n";
        summary << "DIPOLE:\n  Converged: " << dipole_converged << "/" << total_rows << "\n";
        summary << "  Time: " << dipole_duration.count() << " s\n  Avg time: " << dipole_mean_time << " ms\n";
        summary << "  RMSE: " << dipole_mean_rmse << " T (" << (dipole_mean_rmse * 1e6) << " µT)\n";
        summary << "  Position: " << dipole_mean_pos * 1000 << " mm (max " << dipole_max_pos * 1000 << " mm)\n";
        summary << "  Direction: " << dipole_mean_dir << " deg (max " << dipole_max_dir << " deg)\n\n";
        summary << "SPEEDUP: " << (biot_mean_time / dipole_mean_time) << "x\n";
        summary.close();
    }

    return 0;
}
