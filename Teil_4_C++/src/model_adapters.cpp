// model_adapters.cpp
#include "model_adapters.hpp"
#include "biot_field.hpp"
#include "dipole_field.hpp"

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

        Eigen::Vector3d u_norm = u_hat.normalized();

        // Compute with full analytical gradients (position, direction, and scale)
        // results already contain dB_dp, dB_du, and dB_dscale
        auto results = model_.compute_B_batch_with_gradients(sensors, center, u_norm, scale);

        for (size_t i = 0; i < n_sensors; ++i) {
            B_fields[i] = results[i].B;

            // Block 1: Position Gradient (Analytical dB_dp)
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;

            // Block 2: Direction Gradient (Analytical dB_du/G_rot)
            // This replaces the slow 3-call numerical gradient
            jacobian.block<3, 3>(i * 3, 3) = results[i].dB_du;

            // Block 3: Scale Gradient (Analytical dB_dscale)
            // This replaces the slow 1-call numerical gradient
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

        Eigen::Vector3d u_norm = u_hat.normalized();

        // Dipole has full analytical gradients
        auto results = model_.compute_B_batch_with_gradients(sensors, center, u_norm, scale);

        double u_mag = u_norm.norm();
        if (u_mag < 1e-10) u_mag = 1.0;

        for (size_t i = 0; i < n_sensors; ++i) {
            B_fields[i] = results[i].B;
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;

            // Direction with normalization
            Eigen::Matrix3d du_norm_du = (Eigen::Matrix3d::Identity() - u_norm * u_norm.transpose()) / u_mag;
            jacobian.block<3, 3>(i * 3, 3) = results[i].dB_du * du_norm_du;

            // Scale (dipole uses dB_dm)
            jacobian.block<3, 1>(i * 3, 6) = results[i].dB_dm;
        }
    }

} // namespace adapters