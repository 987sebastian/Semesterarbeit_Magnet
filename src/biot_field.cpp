// biot_field.cpp

#include "biot_field.hpp"
#include "config.hpp"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace biot {

    EndDisks::EndDisks(const CylinderGeom& geom, const DiscGrid& disc) {
        build(geom, disc);
    }

    void EndDisks::build(const CylinderGeom& geom, const DiscGrid& disc) {
        const int Nr = disc.Nr;
        const int Nth = disc.Nth;
        const double R = geom.R;
        const double L = geom.L;

        std::vector<double> r_edges(Nr + 1);
        std::vector<double> th_edges(Nth + 1);

        for (int i = 0; i <= Nr; ++i) {
            r_edges[i] = (i * R) / Nr;
        }
        for (int j = 0; j <= Nth; ++j) {
            th_edges[j] = (j * 2.0 * M_PI) / Nth;
        }

        std::vector<double> rc(Nr), thc(Nth);
        for (int i = 0; i < Nr; ++i) {
            rc[i] = 0.5 * (r_edges[i] + r_edges[i + 1]);
        }
        for (int j = 0; j < Nth; ++j) {
            thc[j] = 0.5 * (th_edges[j] + th_edges[j + 1]);
        }

        const int Npts = Nr * Nth;
        p_local_plus.reserve(Npts);
        p_local_minus.reserve(Npts);
        dS.reserve(Npts);

        for (int i = 0; i < Nr; ++i) {
            double dr = r_edges[i + 1] - r_edges[i];
            for (int j = 0; j < Nth; ++j) {
                double dth = th_edges[j + 1] - th_edges[j];
                double r = rc[i];
                double th = thc[j];

                double x = r * std::cos(th);
                double y = r * std::sin(th);
                double area = r * dr * dth;

                p_local_plus.emplace_back(x, y, L / 2.0);
                p_local_minus.emplace_back(x, y, -L / 2.0);
                dS.push_back(area);
            }
        }
    }

    BiotSavartModel::BiotSavartModel(const CylinderGeom& geom, const DiscGrid& disc)
        : geom_(geom), mesh_(geom, disc) {
    }

    void BiotSavartModel::build_frame(
        const Eigen::Vector3d& u_hat,
        Eigen::Vector3d& e1,
        Eigen::Vector3d& e2,
        Eigen::Vector3d& e3
    ) const {
        double uz = u_hat(2);
        double alpha = std::acos(std::clamp(uz, -1.0, 1.0));
        double beta = std::atan2(u_hat(1), u_hat(0));

        e3 = u_hat;
        e2 << std::cos(alpha) * std::cos(beta),
            std::cos(alpha)* std::sin(beta),
            -std::sin(alpha);
        e1 << -std::sin(beta), std::cos(beta), 0.0;
    }

    Eigen::Vector3d BiotSavartModel::kernel(const Eigen::Vector3d& r) const {
        double r_norm = r.norm();
        if (r_norm < 1e-30) {
            return Eigen::Vector3d::Zero();
        }
        return r / (r_norm * r_norm * r_norm);
    }

    Eigen::Matrix3d BiotSavartModel::kernel_gradient(const Eigen::Vector3d& r) const {
        // ∂K/∂r = I/|r|³ - 3(r⊗r^T)/|r|⁵
        double r_norm = r.norm();
        if (r_norm < 1e-30) {
            return Eigen::Matrix3d::Zero();
        }
        double r3 = r_norm * r_norm * r_norm;
        double r5 = r3 * r_norm * r_norm;
        return Eigen::Matrix3d::Identity() / r3 - 3.0 * (r * r.transpose()) / r5;
    }

    void BiotSavartModel::compute_rotation_gradients(
        const Eigen::Vector3d& u_hat,
        const Eigen::Vector3d& e1,
        const Eigen::Vector3d& e2,
        Eigen::Matrix3d dR_du[3]
    ) const {
        double ux = u_hat(0);
        double uy = u_hat(1);
        double uz = u_hat(2);
        double rho = std::sqrt(ux * ux + uy * uy);
        double rho_safe = std::max(rho, 1e-10);

        double alpha = std::acos(std::clamp(uz, -1.0, 1.0));
        double sin_alpha = std::sin(alpha);
        double cos_alpha = uz;
        double sin_beta = uy / rho_safe;
        double cos_beta = ux / rho_safe;

        if (rho < 1e-8) {
            for (int i = 0; i < 3; ++i) {
                dR_du[i].setZero();
            }
            return;
        }

        double dalpha_duz = -1.0 / std::max(sin_alpha, 1e-10);
        Eigen::Vector3d dalpha_du(0, 0, dalpha_duz);
        Eigen::Vector3d dbeta_du(-uy / (rho * rho), ux / (rho * rho), 0.0);

        for (int i = 0; i < 3; ++i) {
            dR_du[i].col(2).setZero();
            dR_du[i](i, 2) = 1.0;

            double da = dalpha_du(i);
            double db = dbeta_du(i);

            dR_du[i].col(1) <<
                -sin_alpha * cos_beta * da - cos_alpha * sin_beta * db,
                -sin_alpha * sin_beta * da + cos_alpha * cos_beta * db,
                -cos_alpha * da;

            dR_du[i].col(0) << -cos_beta * db, -sin_beta * db, 0.0;
        }
    }

    Eigen::Vector3d BiotSavartModel::compute_B(
        const Eigen::Vector3d& sensor,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        const double M0 = geom_.Br / MU0;
        Eigen::Vector3d e1, e2, e3;
        build_frame(u_hat, e1, e2, e3);

        Eigen::Matrix3d R;
        R.col(0) = e1;
        R.col(1) = e2;
        R.col(2) = e3;

        Eigen::Vector3d B = Eigen::Vector3d::Zero();
        const size_t Npts = mesh_.dS.size();
        const double coeff = MU0 / (4.0 * M_PI) * M0 * scale;

        for (size_t k = 0; k < Npts; ++k) {
            Eigen::Vector3d p_world_plus = center + R * mesh_.p_local_plus[k];
            Eigen::Vector3d r_plus = sensor - p_world_plus;
            B += coeff * mesh_.dS[k] * kernel(r_plus);

            Eigen::Vector3d p_world_minus = center + R * mesh_.p_local_minus[k];
            Eigen::Vector3d r_minus = sensor - p_world_minus;
            B -= coeff * mesh_.dS[k] * kernel(r_minus);
        }

        return B;
    }

    BFieldResult BiotSavartModel::compute_B_with_gradients(
        const Eigen::Vector3d& sensor,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        const double M0 = geom_.Br / MU0;
        Eigen::Vector3d e1, e2, e3;
        build_frame(u_hat, e1, e2, e3);

        Eigen::Matrix3d R;
        R.col(0) = e1;
        R.col(1) = e2;
        R.col(2) = e3;

        Eigen::Matrix3d dR_du[3];
        compute_rotation_gradients(u_hat, e1, e2, dR_du);

        BFieldResult result;
        result.B.setZero();
        result.dB_dp.setZero();
        result.dB_du.setZero();
        result.dB_dscale.setZero();

        const size_t Npts = mesh_.dS.size();
        const double coeff = MU0 / (4.0 * M_PI) * M0 * scale;

        for (size_t k = 0; k < Npts; ++k) {
            const double area = mesh_.dS[k];

            // Plus disk
            Eigen::Vector3d p_local_plus = mesh_.p_local_plus[k];
            Eigen::Vector3d p_world_plus = center + R * p_local_plus;
            Eigen::Vector3d r_plus = sensor - p_world_plus;

            Eigen::Vector3d K_plus = kernel(r_plus);
            Eigen::Matrix3d dK_dr_plus = kernel_gradient(r_plus);

            result.B += coeff * area * K_plus;
            result.dB_dp += coeff * area * dK_dr_plus * (-Eigen::Matrix3d::Identity());

            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3d dr_du = -(dR_du[i] * p_local_plus);
                result.dB_du.col(i) += coeff * area * (dK_dr_plus * dr_du);
            }

            result.dB_dscale += (MU0 / (4.0 * M_PI) * M0) * area * K_plus;

            // Minus disk
            Eigen::Vector3d p_local_minus = mesh_.p_local_minus[k];
            Eigen::Vector3d p_world_minus = center + R * p_local_minus;
            Eigen::Vector3d r_minus = sensor - p_world_minus;

            Eigen::Vector3d K_minus = kernel(r_minus);
            Eigen::Matrix3d dK_dr_minus = kernel_gradient(r_minus);

            result.B -= coeff * area * K_minus;
            result.dB_dp -= coeff * area * dK_dr_minus * (-Eigen::Matrix3d::Identity());

            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3d dr_du = -(dR_du[i] * p_local_minus);
                result.dB_du.col(i) -= coeff * area * (dK_dr_minus * dr_du);
            }

            result.dB_dscale -= (MU0 / (4.0 * M_PI) * M0) * area * K_minus;
        }

        return result;
    }

    std::vector<Eigen::Vector3d> BiotSavartModel::compute_B_batch(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        const int N = static_cast<int>(sensors.size());
        std::vector<Eigen::Vector3d> B_fields(N);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) if(N > 4)
#endif
        for (int i = 0; i < N; ++i) {
            B_fields[i] = compute_B(sensors[i], center, u_hat, scale);
        }

        return B_fields;
    }

    std::vector<BFieldResult> BiotSavartModel::compute_B_batch_with_gradients(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        const int N = static_cast<int>(sensors.size());
        std::vector<BFieldResult> results(N);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) if(N > 4)
#endif
        for (int i = 0; i < N; ++i) {
            results[i] = compute_B_with_gradients(sensors[i], center, u_hat, scale);
        }

        return results;
    }

} // namespace biot