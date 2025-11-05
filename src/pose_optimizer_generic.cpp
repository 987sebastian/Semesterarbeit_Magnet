// pose_optimizer_generic.cpp
// Generic Pose Optimizer Implementation
// 通用位姿优化器实现

#include "pose_optimizer_generic.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <limits>

namespace generic {

    GenericPoseOptimizer::GenericPoseOptimizer(
        const MagneticFieldModel* model,
        int max_iterations,
        double tolerance
    )
        : model_(model)
        , max_iterations_(max_iterations)
        , tolerance_(tolerance)
    {
    }

    void GenericPoseOptimizer::compute_residuals_and_jacobian(
        const std::vector<Eigen::Vector3d>& sensors,
        const std::vector<Eigen::Vector3d>& B_meas,
        const Eigen::Vector3d& p,
        const Eigen::Vector3d& u,
        double scale,
        Eigen::VectorXd& residuals,
        Eigen::MatrixXd& jacobian
    ) {
        size_t n_sensors = sensors.size();
        residuals.resize(n_sensors * 3);
        jacobian.resize(n_sensors * 3, 7);

        // Normalize direction / 归一化方向
        Eigen::Vector3d u_norm = u.normalized();

        // Compute predicted fields and Jacobian / 计算预测场和雅可比
        std::vector<Eigen::Vector3d> B_pred;
        model_->compute_fields_with_gradients(sensors, p, u_norm, scale, B_pred, jacobian);

        // Compute residuals / 计算残差
        for (size_t i = 0; i < n_sensors; ++i) {
            Eigen::Vector3d diff = B_pred[i] - B_meas[i];
            residuals.segment<3>(i * 3) = diff;
        }
    }

    PoseResult GenericPoseOptimizer::estimate_pose(
        const std::vector<Eigen::Vector3d>& sensors,
        const std::vector<Eigen::Vector3d>& B_meas,
        const Eigen::Vector3d& p_init,
        const Eigen::Vector3d& u_init,
        double scale_init
    ) {
        PoseResult result;
        result.converged = false;
        result.status = 0;

        Eigen::Vector3d p = p_init;
        Eigen::Vector3d u = u_init.normalized();
        double scale = scale_init;

        // Levenberg-Marquardt parameters / LM参数
        double lambda = config::optimizer::LAMBDA_INIT;
        const double lambda_up = config::optimizer::LAMBDA_UP;
        const double lambda_down = config::optimizer::LAMBDA_DOWN;
        const double lambda_max = config::optimizer::LAMBDA_MAX;
        const double lambda_min = config::optimizer::LAMBDA_MIN;

        const double trust_radius = config::optimizer::TRUST_RADIUS;
        const double min_step_quality = config::optimizer::MIN_STEP_QUALITY;
        const int max_rejections = config::optimizer::MAX_REJECTIONS;
        const double divergence_factor = config::optimizer::DIVERGENCE_FACTOR;

        double cost_prev = std::numeric_limits<double>::max();
        double cost_initial = 0.0;
        int num_rejected = 0;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            Eigen::VectorXd residuals;
            Eigen::MatrixXd J;
            compute_residuals_and_jacobian(sensors, B_meas, p, u, scale, residuals, J);

            double cost = residuals.squaredNorm();
            double rmse = std::sqrt(cost / sensors.size());

            if (iter == 0) {
                cost_initial = cost;
            }

            result.rmse = rmse;
            result.iterations = iter + 1;

#ifndef USE_OPENMP
            if (config::output::VERBOSE && (iter < 3 || iter % 2 == 0)) {
                std::cout << "  Iter " << iter << ": rmse=" << rmse * 1e6 << " uT, lambda=" << lambda << std::endl;
            }
#endif

            // Check convergence / 检查收敛
            if (rmse < tolerance_) {
                result.converged = true;
                result.status = 1;
                break;
            }
            if (rmse < tolerance_ * 10) {
                result.converged = true;
                result.status = 2;
                break;
            }

            // Levenberg-Marquardt step / LM步骤
            Eigen::MatrixXd JtJ = J.transpose() * J;
            Eigen::VectorXd Jtr = J.transpose() * residuals;

            // Add damping / 添加阻尼
            Eigen::MatrixXd A = JtJ + lambda * Eigen::MatrixXd::Identity(7, 7);
            Eigen::VectorXd delta = A.ldlt().solve(-Jtr);

            // Trust region / 信赖域
            double step_norm = delta.norm();
            if (step_norm > trust_radius) {
                delta *= (trust_radius / step_norm);
                step_norm = trust_radius;
            }

            // Apply update / 应用更新
            Eigen::Vector3d p_new = p + delta.segment<3>(0);
            Eigen::Vector3d u_new = u + delta.segment<3>(3);
            u_new.normalize();
            double scale_new = scale + delta(6);

            // Clamp scale / 限制scale范围
            scale_new = std::max(config::optimizer::MIN_SCALE,
                std::min(config::optimizer::MAX_SCALE, scale_new));

            // Evaluate new cost / 评估新成本
            Eigen::VectorXd residuals_new;
            Eigen::MatrixXd J_new;
            compute_residuals_and_jacobian(sensors, B_meas, p_new, u_new, scale_new, residuals_new, J_new);
            double cost_new = residuals_new.squaredNorm();

            // Gain ratio / 增益比
            double predicted_reduction = -delta.dot(Jtr) + 0.5 * delta.dot(JtJ * delta);
            double actual_reduction = cost - cost_new;
            double rho = (predicted_reduction > 1e-12) ? (actual_reduction / predicted_reduction) : -1.0;

            if (rho > min_step_quality && cost_new < cost) {
                // Accept step / 接受步骤
                p = p_new;
                u = u_new;
                scale = scale_new;
                cost_prev = cost;
                num_rejected = 0;

                if (rho > 0.75) {
                    lambda = std::max(lambda * 0.1, lambda_min);
                }
                else {
                    lambda = std::max(lambda * lambda_down, lambda_min);
                }

                // Early convergence check / 早期收敛检查
                double cost_improvement = (cost_prev - cost_new) / (cost_initial + 1e-12);
                if (iter > 2 && step_norm < 1e-6 && cost_improvement < 1e-6) {
                    double rmse_new = std::sqrt(cost_new / sensors.size());
                    if (rmse_new < tolerance_ * 100) {
                        result.converged = true;
                        result.status = 3;
                    }
                    break;
                }
            }
            else {
                // Reject step / 拒绝步骤
                lambda = std::min(lambda * lambda_up, lambda_max);
                num_rejected++;

                if (num_rejected > max_rejections) {
                    double rmse_current = std::sqrt(cost / sensors.size());
                    if (rmse_current < tolerance_ * 100) {
                        result.converged = true;
                        result.status = 4;
                    }
                    break;
                }
            }

            // Divergence check / 发散检查
            if (iter > 0 && cost_new > cost_prev * divergence_factor) {
                result.status = 0;
                break;
            }
        }

        // Final result / 最终结果
        result.position = p;
        result.direction = u.normalized();
        result.scale = scale;

        Eigen::VectorXd residuals_final;
        Eigen::MatrixXd J_final;
        compute_residuals_and_jacobian(sensors, B_meas, p, u, scale, residuals_final, J_final);
        double cost_final = residuals_final.squaredNorm();
        double rmse_final = std::sqrt(cost_final / sensors.size());
        result.rmse = rmse_final;

        // Set convergence status / 设置收敛状态
        if (rmse_final < tolerance_) {
            result.converged = true;
            if (result.status == 0) result.status = 1;
        }
        else if (rmse_final < tolerance_ * 10) {
            result.converged = true;
            if (result.status == 0) result.status = 2;
        }
        else if (rmse_final < tolerance_ * 100) {
            result.converged = true;
            if (result.status == 0) result.status = 3;
        }
        else if (rmse_final < tolerance_ * 500) {
            result.converged = true;
            if (result.status == 0) result.status = 4;
        }
        else {
            result.converged = false;
            if (result.status == 0) result.status = -2;
        }

        return result;
    }

    double GenericPoseOptimizer::compute_position_error(
        const Eigen::Vector3d& p_est,
        const Eigen::Vector3d& p_ref
    ) {
        return (p_est - p_ref).norm();
    }

    double GenericPoseOptimizer::compute_direction_error_deg(
        const Eigen::Vector3d& u_est,
        const Eigen::Vector3d& u_ref
    ) {
        double cos_angle = u_est.dot(u_ref) / (u_est.norm() * u_ref.norm());
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        double angle_rad = std::acos(cos_angle);
        return angle_rad * 180.0 / M_PI;
    }

} // namespace generic
