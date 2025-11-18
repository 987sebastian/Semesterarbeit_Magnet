// include/motion_predictor.hpp
#pragma once
#include <Eigen/Dense>
#include <deque>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace biot {

    class MotionPredictor {
    public:
        MotionPredictor(
            size_t history_size = 3,
            double max_position_jump = 0.010,   // 10mm max jump per frame
            double max_direction_change = 5.0,  // 5 degrees max change
            double damping_factor = 0.3         // Blend with previous (30% new)
        ) : max_history_(history_size),
            max_pos_jump_(max_position_jump),
            max_dir_change_(max_direction_change* M_PI / 180.0),
            damping_(damping_factor) {
        }

        void update(const Eigen::Vector3d& position,
            const Eigen::Vector3d& direction,
            double scale) {
            history_pos_.push_back(position);
            history_dir_.push_back(direction.normalized());
            history_scale_.push_back(scale);

            if (history_pos_.size() > max_history_) {
                history_pos_.pop_front();
                history_dir_.pop_front();
                history_scale_.pop_front();
            }
        }

        // Predict with adaptive step limiting
        void predict(Eigen::Vector3d& p_init,
            Eigen::Vector3d& u_init,
            double& scale_init) const {
            if (history_pos_.empty()) return;

            // === Strategy 1: Use last known pose (safest) ===
            Eigen::Vector3d p_last = history_pos_.back();
            Eigen::Vector3d u_last = history_dir_.back();
            double s_last = history_scale_.back();

            if (history_pos_.size() < 2) {
                // Not enough history - just use last pose
                p_init = p_last;
                u_init = u_last;
                scale_init = s_last;
                return;
            }

            // === Strategy 2: Predict with velocity + limit jump ===
            Eigen::Vector3d v_pos = history_pos_.back() - history_pos_[history_pos_.size() - 2];
            Eigen::Vector3d p_pred = p_last + v_pos;  // Linear extrapolation

            // CRITICAL: Limit position jump
            Eigen::Vector3d dp = p_pred - p_last;
            double jump_distance = dp.norm();
            if (jump_distance > max_pos_jump_) {
                dp = dp * (max_pos_jump_ / jump_distance);  // Clip to max jump
                p_pred = p_last + dp;
            }

            // Predict direction with angular velocity
            Eigen::Vector3d u_prev = history_dir_[history_dir_.size() - 2];
            Eigen::Vector3d u_last_norm = u_last.normalized();
            Eigen::Vector3d axis = u_prev.cross(u_last_norm);
            double sin_angle = axis.norm();

            Eigen::Vector3d u_pred = u_last_norm;  // Default: no change
            if (sin_angle > 1e-6) {
                axis.normalize();
                double angle = std::asin(std::min(1.0, sin_angle));

                // CRITICAL: Limit angular change
                if (angle > max_dir_change_) {
                    angle = max_dir_change_;
                }

                // Apply rotation using Rodrigues' formula
                double cos_a = std::cos(angle);
                double sin_a = std::sin(angle);
                u_pred = u_last_norm * cos_a + axis.cross(u_last_norm) * sin_a +
                    axis * (axis.dot(u_last_norm)) * (1.0 - cos_a);
                u_pred.normalize();
            }

            // Predict scale with damping
            double s_pred = s_last;
            if (history_scale_.size() >= 2) {
                double ds = history_scale_.back() - history_scale_[history_scale_.size() - 2];
                s_pred = s_last + ds * 0.5;  // Half the velocity
                s_pred = std::max(0.5, std::min(2.0, s_pred));  // Clip [0.5, 2.0]
            }

            // === Strategy 3: Blend prediction with last pose (damping) ===
            p_init = (1.0 - damping_) * p_last + damping_ * p_pred;
            u_init = ((1.0 - damping_) * u_last_norm + damping_ * u_pred).normalized();
            scale_init = (1.0 - damping_) * s_last + damping_ * s_pred;
        }

        // Accessors
        bool has_history() const {
            return !history_pos_.empty();
        }

        size_t history_size() const {
            return history_pos_.size();
        }

        Eigen::Vector3d get_last_position() const {
            if (history_pos_.empty()) {
                return Eigen::Vector3d::Zero();
            }
            return history_pos_.back();
        }

        Eigen::Vector3d get_last_direction() const {
            if (history_dir_.empty()) {
                return Eigen::Vector3d(0, 0, 1);  // Default z-axis
            }
            return history_dir_.back();
        }

        double get_last_scale() const {
            if (history_scale_.empty()) {
                return 1.0;
            }
            return history_scale_.back();
        }

        // Get velocity for debugging/analysis
        Eigen::Vector3d get_velocity() const {
            if (history_pos_.size() < 2) {
                return Eigen::Vector3d::Zero();
            }
            return history_pos_.back() - history_pos_[history_pos_.size() - 2];
        }

        // Clear history (e.g., after tracking loss)
        void reset() {
            history_pos_.clear();
            history_dir_.clear();
            history_scale_.clear();
        }

    private:
        std::deque<Eigen::Vector3d> history_pos_;
        std::deque<Eigen::Vector3d> history_dir_;
        std::deque<double> history_scale_;
        size_t max_history_;
        double max_pos_jump_;      // Maximum position jump per frame [m]
        double max_dir_change_;    // Maximum direction change per frame [rad]
        double damping_;           // Damping factor for blending [0,1]
    };

} // namespace biot