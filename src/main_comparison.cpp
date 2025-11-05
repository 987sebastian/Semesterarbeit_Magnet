// main_comparison.cpp
// Model Comparison: Biot-Savart vs Dipole
// 模型对比：Biot-Savart vs Dipole

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include "config.hpp"
#include "io_utils.hpp"
#include "geometry.hpp"
#include "model_adapters.hpp"
#include "pose_optimizer_generic.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace biot;
using namespace generic;
using namespace adapters;

// Structure to hold comparison results
// 保存对比结果的结构体
struct ComparisonResult {
    std::string model_name;
    double mean_rmse;
    double mean_pos_error;
    double mean_dir_error;
    double mean_scale;
    double mean_time_ms;
    int total_samples;
    int converged_samples;
    double convergence_rate;
};

// Run optimization for one model
// 为一个模型运行优化
ComparisonResult run_model_comparison(
    const std::string& model_name,
    const MagneticFieldModel* model,
    const ObservationData& obs_data,
    const std::string& output_file
) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "Testing Model: " << model_name << std::endl;
    std::cout << "============================================\n" << std::endl;

    GenericPoseOptimizer optimizer(model, config::optimizer::MAX_ITERATIONS, config::optimizer::TOLERANCE);

    int n_rows = static_cast<int>(obs_data.mag_positions.size());
    std::vector<double> rmse_values(n_rows);
    std::vector<double> pos_errors(n_rows);
    std::vector<double> dir_errors(n_rows);
    std::vector<double> scales(n_rows);
    std::vector<double> times_ms(n_rows);
    std::vector<int> status_values(n_rows);
    int success_count = 0;

    auto start_all = std::chrono::high_resolution_clock::now();

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, config::parallel::CHUNK_SIZE) if(config::parallel::ENABLE_ROW_PARALLEL)
#endif
    for (int i = 0; i < n_rows; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        PoseResult result = optimizer.estimate_pose(
            obs_data.sensor_positions[i],
            obs_data.B_measured[i],
            obs_data.mag_positions[i],
            obs_data.mag_directions[i],
            1.0
        );

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        rmse_values[i] = result.rmse;
        scales[i] = result.scale;
        times_ms[i] = elapsed_ms;
        status_values[i] = result.status;

        double pos_err = GenericPoseOptimizer::compute_position_error(
            result.position, obs_data.mag_positions[i]
        );
        double dir_err = GenericPoseOptimizer::compute_direction_error_deg(
            result.direction, obs_data.mag_directions[i]
        );

        pos_errors[i] = pos_err;
        dir_errors[i] = dir_err;

        if (result.converged) {
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            success_count++;
        }

#ifndef USE_OPENMP
        if ((i + 1) % config::output::PROGRESS_INTERVAL == 0 || i == n_rows - 1) {
            std::cout << "  Progress: " << (i + 1) << "/" << n_rows << " rows processed" << std::endl;
        }
#endif
    }

    auto end_all = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_all - start_all).count();

    // Compute statistics
    double mean_rmse = 0.0, mean_pos_err = 0.0, mean_dir_err = 0.0, mean_scale = 0.0, mean_time = 0.0;
    for (int i = 0; i < n_rows; ++i) {
        mean_rmse += rmse_values[i];
        mean_pos_err += pos_errors[i];
        mean_dir_err += dir_errors[i];
        mean_scale += scales[i];
        mean_time += times_ms[i];
    }
    mean_rmse /= n_rows;
    mean_pos_err /= n_rows;
    mean_dir_err /= n_rows;
    mean_scale /= n_rows;
    mean_time /= n_rows;

    // Save results
    std::ofstream out(output_file);
    out << std::setprecision(config::output::PRECISION);
    out << "row_index,pos_x,pos_y,pos_z,dir_x,dir_y,dir_z,scale,rmse,status,pos_error,dir_error,time_ms\n";

    for (int i = 0; i < n_rows; ++i) {
        // Note: We save initial values since we're comparing models, not re-optimizing
        out << i << ","
            << obs_data.mag_positions[i](0) << ","
            << obs_data.mag_positions[i](1) << ","
            << obs_data.mag_positions[i](2) << ","
            << obs_data.mag_directions[i](0) << ","
            << obs_data.mag_directions[i](1) << ","
            << obs_data.mag_directions[i](2) << ","
            << scales[i] << ","
            << rmse_values[i] << ","
            << status_values[i] << ","
            << pos_errors[i] << ","
            << dir_errors[i] << ","
            << times_ms[i] << "\n";
    }
    out.close();

    // Print summary
    std::cout << "\n--- " << model_name << " Summary ---" << std::endl;
    std::cout << "  Total samples: " << n_rows << std::endl;
    std::cout << "  Converged: " << success_count << " ("
        << (100.0 * success_count / n_rows) << "%)" << std::endl;
    std::cout << "  Mean RMSE: " << mean_rmse * 1e6 << " µT" << std::endl;
    std::cout << "  Mean position error: " << mean_pos_err * 1000 << " mm" << std::endl;
    std::cout << "  Mean direction error: " << mean_dir_err << "°" << std::endl;
    std::cout << "  Mean scale factor: " << mean_scale << std::endl;
    std::cout << "  Mean time per sample: " << mean_time << " ms" << std::endl;
    std::cout << "  Total time: " << total_time << " s" << std::endl;
    std::cout << "  Throughput: " << (n_rows / total_time) << " Hz" << std::endl;

    ComparisonResult comp_result;
    comp_result.model_name = model_name;
    comp_result.mean_rmse = mean_rmse;
    comp_result.mean_pos_error = mean_pos_err;
    comp_result.mean_dir_error = mean_dir_err;
    comp_result.mean_scale = mean_scale;
    comp_result.mean_time_ms = mean_time;
    comp_result.total_samples = n_rows;
    comp_result.converged_samples = success_count;
    comp_result.convergence_rate = 100.0 * success_count / n_rows;

    return comp_result;
}

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Model Comparison: Biot-Savart vs Dipole" << std::endl;
    std::cout << "  模型对比：Biot-Savart vs Dipole" << std::endl;
    std::cout << "============================================\n" << std::endl;

#ifdef USE_OPENMP
    int num_threads = config::parallel::NUM_THREADS;
    omp_set_num_threads(num_threads);
    std::cout << "OpenMP parallel computing / OpenMP并行计算:" << std::endl;
    std::cout << "  Max threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "  Using threads: " << num_threads << std::endl;
#else
    std::cout << "Sequential execution (OpenMP not enabled)" << std::endl;
#endif

    // Geometry
    CylinderGeom geom(config::magnet::BR, config::magnet::RADIUS, config::magnet::LENGTH);
    std::cout << "\nMagnet geometry / 磁体几何参数:" << std::endl;
    std::cout << "  Remanence (Br) = " << geom.Br << " T" << std::endl;
    std::cout << "  Radius (R) = " << geom.R * 1000 << " mm" << std::endl;
    std::cout << "  Length (L) = " << geom.L * 1000 << " mm" << std::endl;

    // Load calibration
    CalibrationData cal_data;
    if (!load_calibration(config::CALIBRATION_FILE, cal_data)) {
        std::cerr << "[ERROR] Failed to load calibration file: "
            << config::CALIBRATION_FILE << std::endl;
        return 1;
    }
    std::cout << "\n[OK] Loaded calibration for " << cal_data.transformations.size()
        << " sensors." << std::endl;

    // Load observations
    ObservationData obs_data;
    if (!load_observations(config::OBSERVATION_FILE, obs_data)) {
        std::cerr << "[ERROR] Failed to load observations file: "
            << config::OBSERVATION_FILE << std::endl;
        return 1;
    }
    std::cout << "[OK] Loaded " << obs_data.mag_positions.size()
        << " observation rows." << std::endl;

    // Discretization
    DiscGrid disc(config::mesh::NR, config::mesh::NTH);
    std::cout << "\nDiscretization / 离散化参数:" << std::endl;
    std::cout << "  Radial divisions (Nr) = " << disc.Nr << std::endl;
    std::cout << "  Angular divisions (Nth) = " << disc.Nth << std::endl;
    std::cout << "  Total mesh points per disk = " << disc.Nr * disc.Nth << std::endl;

    // Create models
    BiotSavartAdapter biot_model(geom, disc);
    DipoleAdapter dipole_model(geom);

    // Run comparisons
    std::vector<ComparisonResult> results;

    results.push_back(run_model_comparison(
        "Biot-Savart",
        &biot_model,
        obs_data,
        "results/biot_savart_results.csv"
    ));

    results.push_back(run_model_comparison(
        "Dipole",
        &dipole_model,
        obs_data,
        "results/dipole_results.csv"
    ));

    // Print comparison table
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPARISON SUMMARY / 对比总结" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    std::cout << std::left << std::setw(20) << "Metric"
        << std::setw(20) << "Biot-Savart"
        << std::setw(20) << "Dipole"
        << std::setw(20) << "Winner" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    auto print_metric = [](const std::string& name, double val1, double val2, const std::string& unit, bool lower_is_better) {
        std::cout << std::left << std::setw(20) << name
            << std::setw(20) << (std::to_string(val1) + " " + unit)
            << std::setw(20) << (std::to_string(val2) + " " + unit);
        if (lower_is_better) {
            std::cout << std::setw(20) << (val1 < val2 ? "Biot-Savart ✓" : "Dipole ✓");
        }
        else {
            std::cout << std::setw(20) << (val1 > val2 ? "Biot-Savart ✓" : "Dipole ✓");
        }
        std::cout << std::endl;
        };

    print_metric("RMSE", results[0].mean_rmse * 1e6, results[1].mean_rmse * 1e6, "µT", true);
    print_metric("Position Error", results[0].mean_pos_error * 1000, results[1].mean_pos_error * 1000, "mm", true);
    print_metric("Direction Error", results[0].mean_dir_error, results[1].mean_dir_error, "°", true);
    print_metric("Speed", results[0].mean_time_ms, results[1].mean_time_ms, "ms", true);
    print_metric("Convergence Rate", results[0].convergence_rate, results[1].convergence_rate, "%", false);

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "RECOMMENDATION / 建议:" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // Simple recommendation logic
    bool biot_better_accuracy = (results[0].mean_pos_error < results[1].mean_pos_error);
    bool biot_better_rmse = (results[0].mean_rmse < results[1].mean_rmse);
    bool dipole_faster = (results[1].mean_time_ms < results[0].mean_time_ms);

    if (biot_better_accuracy && biot_better_rmse) {
        std::cout << "✓ RECOMMENDATION: Use Biot-Savart Model" << std::endl;
        std::cout << "  Reason: Better accuracy in both RMSE and position error" << std::endl;
        if (dipole_faster) {
            std::cout << "  Note: Dipole is faster but less accurate" << std::endl;
        }
    }
    else if (!biot_better_accuracy && !biot_better_rmse) {
        std::cout << "✓ RECOMMENDATION: Use Dipole Model" << std::endl;
        std::cout << "  Reason: Better accuracy and faster computation" << std::endl;
    }
    else {
        std::cout << "⚠ TRADE-OFF: Choose based on application requirements" << std::endl;
        std::cout << "  - For accuracy: Use " << (biot_better_accuracy ? "Biot-Savart" : "Dipole") << std::endl;
        std::cout << "  - For speed: Use " << (dipole_faster ? "Dipole" : "Biot-Savart") << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Results saved to:" << std::endl;
    std::cout << "  - results/biot_savart_results.csv" << std::endl;
    std::cout << "  - results/dipole_results.csv" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}