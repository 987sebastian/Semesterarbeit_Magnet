// sensors_data.hpp
// Sensor positions management - auto-load from file
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

namespace biot {

    // Load sensor positions from CSV file
    // Format: x,y,z (one sensor per line)
    inline std::vector<Eigen::Vector3d> load_sensor_positions_from_file(const std::string& filename) {
        std::vector<Eigen::Vector3d> sensors;

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Cannot open sensor positions file: " << filename << std::endl;
            std::cerr << "Using default 41-sensor configuration." << std::endl;

            // Fallback to hardcoded 41 sensors
            sensors.reserve(41);

            // 5x5 grid
            sensors.emplace_back(0.005, 0.005, 0);
            sensors.emplace_back(0.04, 0.005, 0);
            sensors.emplace_back(0.075, 0.005, 0);
            sensors.emplace_back(0.11, 0.005, 0);
            sensors.emplace_back(0.145, 0.005, 0);

            sensors.emplace_back(0.005, 0.04, 0);
            sensors.emplace_back(0.04, 0.04, 0);
            sensors.emplace_back(0.075, 0.04, 0);
            sensors.emplace_back(0.11, 0.04, 0);
            sensors.emplace_back(0.145, 0.04, 0);

            sensors.emplace_back(0.005, 0.075, 0);
            sensors.emplace_back(0.04, 0.075, 0);
            sensors.emplace_back(0.075, 0.075, 0);
            sensors.emplace_back(0.11, 0.075, 0);
            sensors.emplace_back(0.145, 0.075, 0);

            sensors.emplace_back(0.005, 0.11, 0);
            sensors.emplace_back(0.04, 0.11, 0);
            sensors.emplace_back(0.075, 0.11, 0);
            sensors.emplace_back(0.11, 0.11, 0);
            sensors.emplace_back(0.145, 0.11, 0);

            sensors.emplace_back(0.005, 0.145, 0);
            sensors.emplace_back(0.04, 0.145, 0);
            sensors.emplace_back(0.075, 0.145, 0);
            sensors.emplace_back(0.11, 0.145, 0);
            sensors.emplace_back(0.145, 0.145, 0);

            // 4x4 offset grid
            sensors.emplace_back(0.0225, 0.0225, 0);
            sensors.emplace_back(0.0575, 0.0225, 0);
            sensors.emplace_back(0.0925, 0.0225, 0);
            sensors.emplace_back(0.1275, 0.0225, 0);

            sensors.emplace_back(0.0225, 0.0575, 0);
            sensors.emplace_back(0.0575, 0.0575, 0);
            sensors.emplace_back(0.0925, 0.0575, 0);
            sensors.emplace_back(0.1275, 0.0575, 0);

            sensors.emplace_back(0.0225, 0.0925, 0);
            sensors.emplace_back(0.0575, 0.0925, 0);
            sensors.emplace_back(0.0925, 0.0925, 0);
            sensors.emplace_back(0.1275, 0.0925, 0);

            sensors.emplace_back(0.0225, 0.1275, 0);
            sensors.emplace_back(0.0575, 0.1275, 0);
            sensors.emplace_back(0.0925, 0.1275, 0);
            sensors.emplace_back(0.1275, 0.1275, 0);

            return sensors;
        }

        // Read from file
        std::string line;
        int line_number = 0;

        while (std::getline(file, line)) {
            line_number++;

            // Skip empty lines
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string x_str, y_str, z_str;

            if (!std::getline(ss, x_str, ',') ||
                !std::getline(ss, y_str, ',') ||
                !std::getline(ss, z_str, ',')) {
                std::cerr << "Warning: Invalid format at line " << line_number
                    << " in " << filename << std::endl;
                continue;
            }

            try {
                double x = std::stod(x_str);
                double y = std::stod(y_str);
                double z = std::stod(z_str);

                sensors.emplace_back(x, y, z);
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse line " << line_number
                    << " in " << filename << ": " << e.what() << std::endl;
            }
        }

        file.close();

        std::cout << "[OK] Loaded " << sensors.size() << " sensor positions from "
            << filename << std::endl;

        return sensors;
    }

    // Get sensor positions (auto-load or use default)
    inline std::vector<Eigen::Vector3d> get_sensor_positions(const std::string& filename = "") {
        if (filename.empty()) {
            // Use default 41-sensor configuration
            std::vector<Eigen::Vector3d> sensors;
            sensors.reserve(41);

            // 5x5 grid
            sensors.emplace_back(0.005, 0.005, 0);
            sensors.emplace_back(0.04, 0.005, 0);
            sensors.emplace_back(0.075, 0.005, 0);
            sensors.emplace_back(0.11, 0.005, 0);
            sensors.emplace_back(0.145, 0.005, 0);

            sensors.emplace_back(0.005, 0.04, 0);
            sensors.emplace_back(0.04, 0.04, 0);
            sensors.emplace_back(0.075, 0.04, 0);
            sensors.emplace_back(0.11, 0.04, 0);
            sensors.emplace_back(0.145, 0.04, 0);

            sensors.emplace_back(0.005, 0.075, 0);
            sensors.emplace_back(0.04, 0.075, 0);
            sensors.emplace_back(0.075, 0.075, 0);
            sensors.emplace_back(0.11, 0.075, 0);
            sensors.emplace_back(0.145, 0.075, 0);

            sensors.emplace_back(0.005, 0.11, 0);
            sensors.emplace_back(0.04, 0.11, 0);
            sensors.emplace_back(0.075, 0.11, 0);
            sensors.emplace_back(0.11, 0.11, 0);
            sensors.emplace_back(0.145, 0.11, 0);

            sensors.emplace_back(0.005, 0.145, 0);
            sensors.emplace_back(0.04, 0.145, 0);
            sensors.emplace_back(0.075, 0.145, 0);
            sensors.emplace_back(0.11, 0.145, 0);
            sensors.emplace_back(0.145, 0.145, 0);

            // 4x4 offset grid
            sensors.emplace_back(0.0225, 0.0225, 0);
            sensors.emplace_back(0.0575, 0.0225, 0);
            sensors.emplace_back(0.0925, 0.0225, 0);
            sensors.emplace_back(0.1275, 0.0225, 0);

            sensors.emplace_back(0.0225, 0.0575, 0);
            sensors.emplace_back(0.0575, 0.0575, 0);
            sensors.emplace_back(0.0925, 0.0575, 0);
            sensors.emplace_back(0.1275, 0.0575, 0);

            sensors.emplace_back(0.0225, 0.0925, 0);
            sensors.emplace_back(0.0575, 0.0925, 0);
            sensors.emplace_back(0.0925, 0.0925, 0);
            sensors.emplace_back(0.1275, 0.0925, 0);

            sensors.emplace_back(0.0225, 0.1275, 0);
            sensors.emplace_back(0.0575, 0.1275, 0);
            sensors.emplace_back(0.0925, 0.1275, 0);
            sensors.emplace_back(0.1275, 0.1275, 0);

            return sensors;
        }
        else {
            // Load from file
            return load_sensor_positions_from_file(filename);
        }
    }

} // namespace biot