#pragma once
// sensors_data.hpp
// Hardcoded 41 sensor positions / 硬编码的41个传感器位置
#pragma once

#include <Eigen/Dense>
#include <vector>

namespace biot {

    // Get the 41 fixed sensor positions [m] / 获取41个固定传感器位置 [m]
    inline std::vector<Eigen::Vector3d> get_sensor_positions() {
        std::vector<Eigen::Vector3d> sensors;
        sensors.reserve(41);

        // 5x5 grid / 5×5网格
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

        // 4x4 offset grid / 4×4偏移网格
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

} // namespace biot