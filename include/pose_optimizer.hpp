#pragma once
// pose_optimizer.hpp
// Pose optimization using Gauss-Newton / 使用高斯-牛顿法的位姿优化
#pragma once

#include <Eigen/Dense>
#include <vector>
#include "biot_field.hpp"
#include "geometry.hpp"

namespace biot {

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

    // Pose optimizer / 位姿优化器
    class PoseOptimizer {
    public:
        PoseOptimizer(
            const CylinderGeom& geom,
            const DiscGrid& disc = DiscGrid(),
            int max_iterations = 20,
            double tolerance = 1e-8
        );

        // Estimate pose from sensor measurements / 从传感器测量估计位姿
        // sensors: sensor positions [m] / 传感器位置 [m]
        // B_meas: measured B fields [T] / 测量磁场 [T]
        // p_init: initial position guess / 初始位置猜测
        // u_init: initial direction guess (need not be unit) / 初始方向猜测（无需单位化）
        // scale_init: initial scale guess / 初始尺度猜测
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
        CylinderGeom geom_;
        BiotSavartModel model_;
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

        // Numerical Jacobian using finite differences / 使用有限差分的数值雅可比
        void numerical_jacobian(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& p,
            const Eigen::Vector3d& u,
            double scale,
            const std::vector<Eigen::Vector3d>& B_pred,
            Eigen::MatrixXd& J
        );
    };

} // namespace biot