// io_utils.cpp
// Implementation of I/O utilities / I/O工具实现
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

            // Parse each sensor's transformation / 解析每个传感器的变换
            // Support string format sensor IDs like "00", "01", etc. / 支持字符串格式传感器ID如"00"、"01"等
            for (auto& [key, value] : j.items()) {
                int sensor_id = std::stoi(key);  // Convert string "00" to int 0 / 将字符串"00"转换为整数0
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

        // Try to get sensor positions from config file, fallback to default / 尝试从配置文件获取传感器位置，失败则使用默认
        std::vector<Eigen::Vector3d> sensor_positions_fixed;

        // Load sensor positions from file or use default / 从文件加载传感器位置或使用默认
        sensor_positions_fixed = get_sensor_positions(config::SENSOR_POSITIONS_FILE);

        const int num_sensors = static_cast<int>(sensor_positions_fixed.size());
        const int expected_cols = num_sensors * 3 + 6;  // sensors × 3 + 6 magnet params / 传感器×3 + 6个磁体参数

        std::cout << "[INFO] Using " << num_sensors << " sensors, expecting "
            << expected_cols << " columns in CSV" << std::endl;

        for (const auto& tokens : rows) {
            // Check column count / 检查列数
            if (tokens.size() < static_cast<size_t>(expected_cols)) {
                std::cerr << "Warning: Row has insufficient columns (" << tokens.size() << ")" << std::endl;
                std::cerr << "Expected: " << expected_cols << " columns ("
                    << num_sensors << " sensors × 3 B-field + 6 magnet params)" << std::endl;
                continue;
            }

            // Parse B-field measurements from sensors / 解析传感器的磁场测量值
            // (x0,y0,z0, x1,y1,z1, ..., x[n-1],y[n-1],z[n-1])
            // Note: x,y,z here are B-field Bx, By, Bz, not positions! / 注意：这里x,y,z是磁场Bx,By,Bz，不是位置！
            std::vector<Eigen::Vector3d> sensors_B;
            sensors_B.reserve(num_sensors);

            bool parse_error = false;
            for (int i = 0; i < num_sensors; ++i) {
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

            // Parse initial magnet pose (last 6 columns) / 解析磁体初始位姿（最后6列）
            const int mag_start_idx = num_sensors * 3;
            double mag_x, mag_y, mag_z, mag_mx, mag_my, mag_mz;

            if (!CSVParser::to_double(tokens[mag_start_idx], mag_x) ||
                !CSVParser::to_double(tokens[mag_start_idx + 1], mag_y) ||
                !CSVParser::to_double(tokens[mag_start_idx + 2], mag_z) ||
                !CSVParser::to_double(tokens[mag_start_idx + 3], mag_mx) ||
                !CSVParser::to_double(tokens[mag_start_idx + 4], mag_my) ||
                !CSVParser::to_double(tokens[mag_start_idx + 5], mag_mz)) {
                std::cerr << "Warning: Failed to parse magnet pose" << std::endl;
                continue;
            }

            Eigen::Vector3d mag_pos(mag_x, mag_y, mag_z);
            Eigen::Vector3d mag_dir(mag_mx, mag_my, mag_mz);

            obs_data.mag_positions.push_back(mag_pos);
            obs_data.mag_directions.push_back(mag_dir);
            obs_data.sensor_positions.push_back(sensor_positions_fixed);  // Use fixed positions / 使用固定位置
            obs_data.B_measured.push_back(sensors_B);  // Magnetic field measurements / 磁场测量值
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
            // The calibration matrix must be multiplied with each B-field value
            // 校准矩阵必须乘以每个磁场值
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
            std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
            return false;
        }

        // Set precision / 设置精度
        file << std::fixed << std::setprecision(6);

        // Write header / 写表头
        file << "row,pos_error_mm,dir_error_deg\n";

        // Write data / 写数据
        for (size_t i = 0; i < row_indices.size(); ++i) {
            file << row_indices[i] << ","
                << pos_errors[i] * 1000.0 << ","  // Convert to mm / 转换为毫米
                << dir_errors[i] << "\n";
        }

        return true;
    }

} // namespace biot