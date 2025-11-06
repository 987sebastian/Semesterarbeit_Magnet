// pose_optimizer_generic.hpp
// Generic Pose Optimizer supporting both Biot-Savart and Dipole models
// 通用位姿优化器 - 支持Biot-Savart和Dipole两种模型

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "geometry.hpp"

namespace generic {

    // Pose estimation result / 位姿估计结果
    struct PoseResult {
        Eigen::Vector3d position;    // Estimated magnet center / 估计的磁体中心
        Eigen::Vector3d direction;   // Estimated unit direction / 估计的单位方向
        double scale;                 // Estimated scale factor / 估计的尺度因子
        double rmse;                  // Root mean square error (cost) / 均方根误差
        int iterations;               // Number of iterations (nfev) / 迭代次数
        bool converged;               // Convergence flag / 收敛标志
        int status;                   // Status: 1=converged, 0=not converged / 状态
    };

    // Abstract base class for magnetic field models
    // 磁场模型的抽象基类
    class MagneticFieldModel {
    public:
        virtual ~MagneticFieldModel() = default;

        // Compute B field for multiple sensors with gradients
        // 为多个传感器计算磁场及梯度
        virtual void compute_fields_with_gradients(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale,
            std::vector<Eigen::Vector3d>& B_fields,
            Eigen::MatrixXd& jacobian
        ) const = 0;

        // Get model name / 获取模型名称
        virtual std::string get_model_name() const = 0;
    };

    // Generic pose optimizer that works with any MagneticFieldModel
    // 适用于任何磁场模型的通用位姿优化器
    class GenericPoseOptimizer {
    public:
        GenericPoseOptimizer(
            const MagneticFieldModel* model,
            int max_iterations = 50,
            double tolerance = 4e-8
        );

        // Estimate pose from sensor measurements / 从传感器测量估计位姿
        PoseResult estimate_pose(
            const std::vector<Eigen::Vector3d>& sensors,
            const std::vector<Eigen::Vector3d>& B_meas,
            const Eigen::Vector3d& p_init,
            const Eigen::Vector3d& u_init,
            double scale_init = 1.0
        );

        // Compute position and direction errors / 计算位置和方向误差
        static double compute_position_error(
            const Eigen::Vector3d& p_est,
            const Eigen::Vector3d& p_ref
        );

        static double compute_direction_error_deg(
            const Eigen::Vector3d& u_est,
            const Eigen::Vector3d& u_ref
        );

    private:
        const MagneticFieldModel* model_;
        int max_iterations_;
        double tolerance_;

        // Compute residuals and Jacobian / 计算残差和雅可比
        void compute_residuals_and_jacobian(
            const std::vector<Eigen::Vector3d>& sensors,
            const std::vector<Eigen::Vector3d>& B_meas,
            const Eigen::Vector3d& p,
            const Eigen::Vector3d& u,
            double scale,
            Eigen::VectorXd& residuals,
            Eigen::MatrixXd& jacobian
        );
    };

} // namespace generic