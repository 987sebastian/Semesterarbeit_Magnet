#pragma once
// model_adapters.hpp
// Adapters for Biot-Savart and Dipole models to work with generic optimizer
// Biot-Savart��Dipoleģ��������

#pragma once

#include "pose_optimizer_generic.hpp"
#include "biot_field.hpp"
#include "dipole_field.hpp"

namespace adapters {

    // Biot-Savart model adapter / Biot-Savartģ��������
    class BiotSavartAdapter : public generic::MagneticFieldModel {
    public:
        BiotSavartAdapter(const biot::CylinderGeom& geom, const biot::DiscGrid& disc)
            : model_(geom, disc)
        {
        }

        void compute_fields_with_gradients(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale,
            std::vector<Eigen::Vector3d>& B_fields,
            Eigen::MatrixXd& jacobian
        ) const override;

        std::string get_model_name() const override {
            return "Biot-Savart";
        }

    private:
        biot::BiotSavartModel model_;
    };

    // Dipole model adapter / Dipoleģ��������
    class DipoleAdapter : public generic::MagneticFieldModel {
    public:
        DipoleAdapter(const biot::CylinderGeom& geom)
            : model_(geom)
        {
        }

        void compute_fields_with_gradients(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale,
            std::vector<Eigen::Vector3d>& B_fields,
            Eigen::MatrixXd& jacobian
        ) const override;

        std::string get_model_name() const override {
            return "Dipole";
        }

    private:
        dipole::DipoleModel model_;
    };

} // namespace adapters