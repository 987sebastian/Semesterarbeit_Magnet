// model_adapters.cpp
#include "model_adapters.hpp"

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

        // Compute with analytical position gradient
        auto results = model_.compute_B_batch_with_gradients(sensors, center, u_norm, scale);

        for (size_t i = 0; i < n_sensors; ++i) {
            B_fields[i] = results[i].B;
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;
        }

        // Numerical gradients for direction (3 calls)
        const double eps = 1e-7;
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d u_plus = u_norm;
            u_plus(j) += eps;
            u_plus.normalize();

            auto B_plus = model_.compute_B_batch(sensors, center, u_plus, scale);

            for (size_t i = 0; i < n_sensors; ++i) {
                jacobian.block<3, 1>(i * 3, 3 + j) = (B_plus[i] - B_fields[i]) / eps;
            }
        }

        // Numerical gradient for scale (1 call)
        double scale_plus = scale + eps;
        auto B_scale = model_.compute_B_batch(sensors, center, u_norm, scale_plus);

        for (size_t i = 0; i < n_sensors; ++i) {
            jacobian.block<3, 1>(i * 3, 6) = (B_scale[i] - B_fields[i]) / eps;
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