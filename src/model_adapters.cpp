// model_adapters.cpp
// Model Adapters Implementation
// 模型适配器实现

#include "model_adapters.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace adapters {

    void BiotSavartAdapter::compute_fields_with_gradients(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale,
        std::vector<Eigen::Vector3d>& B_fields,
        Eigen::MatrixXd& jacobian
    ) const {
        size_t n_sensors = sensors.size();
        B_fields.resize(n_sensors);
        jacobian.resize(n_sensors * 3, 7);

        // Compute results with gradients / 计算结果及梯度
        auto results = model_.compute_B_batch_with_gradients(sensors, center, u_hat, scale);

        // Extract fields and build Jacobian / 提取场并构建雅可比
        double u_mag = u_hat.norm();
        if (u_mag < 1e-10) u_mag = 1.0;

        for (size_t i = 0; i < n_sensors; ++i) {
            B_fields[i] = results[i].B;

            // Position derivatives ∂B/∂p (3×3)
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;

            // Direction derivatives ∂B/∂u (3×3)
            // Account for normalization: ∂B/∂u_normalized = ∂B/∂u * ∂u_normalized/∂u
            Eigen::Vector3d u_norm = u_hat / u_mag;
            Eigen::Matrix3d du_norm_du = (Eigen::Matrix3d::Identity() - u_norm * u_norm.transpose()) / u_mag;
            Eigen::Matrix3d dB_du_normalized = results[i].dB_du * du_norm_du;
            jacobian.block<3, 3>(i * 3, 3) = dB_du_normalized;

            // Scale derivative ∂B/∂scale (3×1)
            jacobian.block<3, 1>(i * 3, 6) = results[i].dB_dscale;
        }
    }

    void DipoleAdapter::compute_fields_with_gradients(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale,
        std::vector<Eigen::Vector3d>& B_fields,
        Eigen::MatrixXd& jacobian
    ) const {
        size_t n_sensors = sensors.size();
        B_fields.resize(n_sensors);
        jacobian.resize(n_sensors * 3, 7);

        // Compute results with gradients / 计算结果及梯度
        auto results = model_.compute_B_batch_with_gradients(sensors, center, u_hat, scale);

        // Extract fields and build Jacobian / 提取场并构建雅可比
        double u_mag = u_hat.norm();
        if (u_mag < 1e-10) u_mag = 1.0;

        for (size_t i = 0; i < n_sensors; ++i) {
            B_fields[i] = results[i].B;

            // Position derivatives ∂B/∂p (3×3)
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;

            // Direction derivatives ∂B/∂u (3×3)
            // Account for normalization
            Eigen::Vector3d u_norm = u_hat / u_mag;
            Eigen::Matrix3d du_norm_du = (Eigen::Matrix3d::Identity() - u_norm * u_norm.transpose()) / u_mag;
            Eigen::Matrix3d dB_du_normalized = results[i].dB_du * du_norm_du;
            jacobian.block<3, 3>(i * 3, 3) = dB_du_normalized;

            // Scale derivative ∂B/∂scale (3×1)
            // For dipole: ∂B/∂scale = ∂B/∂m * ∂m/∂scale = ∂B/∂m * m_base
            jacobian.block<3, 1>(i * 3, 6) = results[i].dB_dm;
        }
    }

} // namespace adapters