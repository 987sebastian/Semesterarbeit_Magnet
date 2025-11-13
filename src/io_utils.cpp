// io_utils.cpp
// Implementation of I/O utilities with zeroing support / I/O工具实现（带归零功能）
#include "io_utils.hpp"
#include "csv_parser.hpp"
#include "sensors_data.hpp"
#include "config.hpp"
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

using json = nlohmann::json;

namespace biot {

    bool load_calibration(const std::string& json_path, CalibrationData& cal_data) {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open calibration file: " << json_path << std::endl;
            return false;
        }

        try {
            json j;
            file >> j;

            // Parse each sensor's transformation and zeroing / 解析每个传感器的变换和归零数据
            for (auto& [key, value] : j.items()) {
                int sensor_id = std::stoi(key);

                // Parse transformation matrix / 解析变换矩阵
                auto& trans = value["transformation"];
                Eigen::Matrix4d T;
                for (int i = 0; i < 4; ++i) {
                    for (int k = 0; k < 4; ++k) {
                        T(i, k) = trans[i][k];
                    }
                }
                cal_data.transformations[sensor_id] = T;

                // Parse zeroing vector if exists / 解析归零向量（如果存在）
                if (value.contains("zeroing")) {
                    auto& zero = value["zeroing"];
                    Eigen::Vector3d z;
                    z << zero[0], zero[1], zero[2];
                    cal_data.zeroing[sensor_id] = z;
                }
            }

            std::cout << "[OK] Loaded calibration for " << cal_data.transformations.size()
                << " sensors";
            if (!cal_data.zeroing.empty()) {
                std::cout << " (with zeroing data for " << cal_data.zeroing.size() << " sensors)";
            }
            std::cout << std::endl;

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing calibration JSON: " << e.what() << std::endl;
            return false;
        }
    }

    bool load_observations(const std::string& csv_path, ObservationData& obs_data) {
        std::vector<std::vector<std::string>> rows;

        if (!CSVParser::parse_file(csv_path, rows, true)) {
            std::cerr << "Error: Cannot open observations file: " << csv_path << std::endl;
            return false;
        }

        std::vector<Eigen::Vector3d> sensor_positions_fixed;
        sensor_positions_fixed = get_sensor_positions(config::SENSOR_POSITIONS_FILE);

        const int num_sensors = static_cast<int>(sensor_positions_fixed.size());
        const int expected_cols = num_sensors * 3 + 6;

        std::cout << "[INFO] Using " << num_sensors << " sensors, expecting "
            << expected_cols << " columns in CSV" << std::endl;

        for (const auto& tokens : rows) {
            if (tokens.size() < static_cast<size_t>(expected_cols)) {
                std::cerr << "Warning: Row has insufficient columns (" << tokens.size() << ")" << std::endl;
                continue;
            }

            std::vector<Eigen::Vector3d> B_fields;
            B_fields.reserve(num_sensors);

            for (int i = 0; i < num_sensors; ++i) {
                double bx = std::stod(tokens[i * 3 + 0]);
                double by = std::stod(tokens[i * 3 + 1]);
                double bz = std::stod(tokens[i * 3 + 2]);
                B_fields.emplace_back(bx, by, bz);
            }

            int offset = num_sensors * 3;
            double px = std::stod(tokens[offset + 0]);
            double py = std::stod(tokens[offset + 1]);
            double pz = std::stod(tokens[offset + 2]);
            double ux = std::stod(tokens[offset + 3]);
            double uy = std::stod(tokens[offset + 4]);
            double uz = std::stod(tokens[offset + 5]);

            obs_data.mag_positions.emplace_back(px, py, pz);
            obs_data.mag_directions.emplace_back(ux, uy, uz);
            obs_data.sensor_positions.push_back(sensor_positions_fixed);
            obs_data.B_measured.push_back(B_fields);
        }

        std::cout << "[OK] Loaded " << obs_data.mag_positions.size()
            << " observations" << std::endl;

        return !obs_data.mag_positions.empty();
    }

    void apply_calibration(
        const std::vector<Eigen::Vector3d>& raw_positions,
        const std::vector<Eigen::Vector3d>& raw_B_fields,
        const CalibrationData& cal_data,
        std::vector<Eigen::Vector3d>& cal_positions,
        std::vector<Eigen::Vector3d>& cal_B_fields
    ) {
        cal_positions.clear();
        cal_B_fields.clear();
        cal_positions.reserve(raw_positions.size());
        cal_B_fields.reserve(raw_B_fields.size());

        for (size_t i = 0; i < raw_positions.size(); ++i) {
            if (cal_data.transformations.find(i) == cal_data.transformations.end()) {
                std::cerr << "Warning: No calibration for sensor " << i << std::endl;
                cal_positions.push_back(raw_positions[i]);
                cal_B_fields.push_back(raw_B_fields[i]);
                continue;
            }

            const Eigen::Matrix4d& T = cal_data.transformations.at(i);

            // Apply transformation to position / 对位置应用变换
            Eigen::Vector4d r_homo;
            r_homo << raw_positions[i], 1.0;
            Eigen::Vector4d r_cal_homo = T * r_homo;
            cal_positions.emplace_back(r_cal_homo.head<3>());

            // Apply rotation to B field / 对磁场应用旋转
            Eigen::Matrix3d R = T.block<3, 3>(0, 0);
            Eigen::Vector3d B_transformed = R * raw_B_fields[i];

            // Apply zeroing (subtract earth field) / 应用归零（减去地磁场）
            if (cal_data.zeroing.find(i) != cal_data.zeroing.end()) {
                B_transformed -= cal_data.zeroing.at(i);
            }

            cal_B_fields.push_back(B_transformed);
        }
    }

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
    ) {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
            return false;
        }

        file << std::fixed << std::setprecision(10);
        file << "row,model,px,py,pz,ux,uy,uz,k,cost,nfev,status\n";

        for (size_t i = 0; i < row_indices.size(); ++i) {
            file << row_indices[i] << ","
                << model_name << ","
                << positions[i](0) << "," << positions[i](1) << "," << positions[i](2) << ","
                << directions[i](0) << "," << directions[i](1) << "," << directions[i](2) << ","
                << scales[i] << "," << rmse_values[i] << ","
                << nfev_values[i] << "," << status_values[i] << "\n";
        }

        file.close();
        std::cout << "[OK] Results saved to: " << output_path << std::endl;
        return true;
    }

    bool save_error_summary(
        const std::string& output_path,
        const std::vector<int>& row_indices,
        const std::vector<double>& pos_errors,
        const std::vector<double>& dir_errors
    ) {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open error summary file: " << output_path << std::endl;
            return false;
        }

        file << std::fixed << std::setprecision(10);
        file << "row,pos_error_m,dir_error_deg\n";

        for (size_t i = 0; i < row_indices.size(); ++i) {
            file << row_indices[i] << ","
                << pos_errors[i] << ","
                << dir_errors[i] << "\n";
        }

        file.close();
        std::cout << "[OK] Error summary saved to: " << output_path << std::endl;
        return true;
    }

} // namespace biot