// io_utils.hpp
// Input/Output utilities / 输入输出工具
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>

namespace biot {

    // Calibration data structure / 标定数据结构
    struct CalibrationData {
        std::map<int, Eigen::Matrix4d> transformations;  // Sensor transformations / 传感器变换矩阵
    };

    // Observation data structure / 观测数据结构
    struct ObservationData {
        std::vector<Eigen::Vector3d> mag_positions;   // Initial magnet positions / 初始磁体位置
        std::vector<Eigen::Vector3d> mag_directions;  // Initial magnet directions / 初始磁体方向
        std::vector<std::vector<Eigen::Vector3d>> sensor_positions;  // Calibrated sensor positions / 标定后传感器位置
        std::vector<std::vector<Eigen::Vector3d>> B_measured;        // Measured B fields / 测量磁场
    };

    // Load calibration from JSON / 从JSON加载标定数据
    bool load_calibration(const std::string& json_path, CalibrationData& cal_data);

    // Load observations from CSV / 从CSV加载观测数据
    bool load_observations(const std::string& csv_path, ObservationData& obs_data);

    // Apply calibration to raw sensor data / 对原始传感器数据应用标定
    void apply_calibration(
        const std::vector<Eigen::Vector3d>& raw_positions,
        const std::vector<Eigen::Vector3d>& raw_B_fields,
        const CalibrationData& cal_data,
        std::vector<Eigen::Vector3d>& cal_positions,
        std::vector<Eigen::Vector3d>& cal_B_fields
    );

    // Save results to CSV / 保存结果到CSV
    bool save_results(
        const std::string& output_path,
        const std::vector<int>& row_indices,
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& directions,
        const std::vector<double>& scales,
        const std::vector<double>& rmse_values,
        const std::vector<int>& nfev_values,
        const std::vector<int>& status_values,
        const std::string& model_name
    );

    // Save error summary to CSV / 保存误差摘要到CSV
    bool save_error_summary(
        const std::string& output_path,
        const std::vector<int>& row_indices,
        const std::vector<double>& pos_errors,
        const std::vector<double>& dir_errors
    );

} // namespace biot