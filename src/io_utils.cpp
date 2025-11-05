// io_utils.cpp
// Implementation of I/O utilities / I/O工具实现
#include "io_utils.hpp"
#include "csv_parser.hpp"
#include "sensors_data.hpp"
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

            // Parse each sensor's transformation / 解析每个传感器的变换
            for (auto& [key, value] : j.items()) {
                int sensor_id = std::stoi(key);
                auto& trans = value["transformation"];

                Eigen::Matrix4d T;
                for (int i = 0; i < 4; ++i) {
                    for (int k = 0; k < 4; ++k) {
                        T(i, k) = trans[i][k];
                    }
                }

                cal_data.transformations[sensor_id] = T;
            }

            std::cout << "[OK] Loaded calibration for " << cal_data.transformations.size()
                << " sensors." << std::endl;
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

        // Get fixed sensor positions / 获取固定传感器位置
        auto sensor_positions_fixed = get_sensor_positions();

        for (const auto& tokens : rows) {
            // 检查列数：41个传感器×3个磁场分量 + 6个磁体参数 = 129列
            if (tokens.size() < 129) {
                std::cerr << "Warning: Row has insufficient columns (" << tokens.size() << ")" << std::endl;
                std::cerr << "Expected: 129 columns (41 sensors × 3 B-field + 6 magnet params)" << std::endl;
                continue;
            }

            // 解析41个传感器的磁场测量值 (x0,y0,z0, x1,y1,z1, ..., x40,y40,z40)
            // 注意：这里 x,y,z 是磁场 Bx, By, Bz，不是位置！
            std::vector<Eigen::Vector3d> sensors_B;
            sensors_B.reserve(41);

            bool parse_error = false;
            for (int i = 0; i < 41; ++i) {
                int base_idx = i * 3;

                double Bx, By, Bz;

                if (!CSVParser::to_double(tokens[base_idx], Bx) ||
                    !CSVParser::to_double(tokens[base_idx + 1], By) ||
                    !CSVParser::to_double(tokens[base_idx + 2], Bz)) {
                    parse_error = true;
                    break;
                }

                sensors_B.emplace_back(Bx, By, Bz);
            }

            if (parse_error) {
                std::cerr << "Warning: Failed to parse B-field measurements" << std::endl;
                continue;
            }

            // 解析磁体初始位姿 (最后6列)
            double mag_x, mag_y, mag_z, mag_mx, mag_my, mag_mz;

            if (!CSVParser::to_double(tokens[123], mag_x) ||
                !CSVParser::to_double(tokens[124], mag_y) ||
                !CSVParser::to_double(tokens[125], mag_z) ||
                !CSVParser::to_double(tokens[126], mag_mx) ||
                !CSVParser::to_double(tokens[127], mag_my) ||
                !CSVParser::to_double(tokens[128], mag_mz)) {
                std::cerr << "Warning: Failed to parse magnet pose" << std::endl;
                continue;
            }

            Eigen::Vector3d mag_pos(mag_x, mag_y, mag_z);
            Eigen::Vector3d mag_dir(mag_mx, mag_my, mag_mz);

            obs_data.mag_positions.push_back(mag_pos);
            obs_data.mag_directions.push_back(mag_dir);
            obs_data.sensor_positions.push_back(sensor_positions_fixed);  // 使用固定位置
            obs_data.B_measured.push_back(sensors_B);  // 磁场测量值
        }

        std::cout << "[OK] Loaded " << obs_data.mag_positions.size()
            << " observation rows." << std::endl;
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
            cal_B_fields.push_back(R * raw_B_fields[i]);
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

        // Set precision / 设置精度
        file << std::fixed << std::setprecision(10);

        // Write header matching dipole format / 写表头（与dipole格式匹配）
        file << "row,model,px,py,pz,ux,uy,uz,k,cost,nfev,status\n";

        // Write data / 写数据
        for (size_t i = 0; i < row_indices.size(); ++i) {
            file << row_indices[i] << ","
                << model_name << ","
                << positions[i](0) << "," << positions[i](1) << "," << positions[i](2) << ","
                << directions[i](0) << "," << directions[i](1) << "," << directions[i](2) << ","
                << scales[i] << ","
                << rmse_values[i] << ","
                << nfev_values[i] << ","
                << status_values[i] << "\n";
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

        // Write header / 写表头
        file << "row,pos_err_norm,dir_err_deg\n";

        // Write data / 写数据
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