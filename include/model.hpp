#pragma once
#include <array>
#include <vector>
#include <cmath>

/** Model of a uniformly magnetized cylinder using end-surface magnetic charges.
 *  采用端面表面磁荷的均匀磁化圆柱模型。
 *  Orientation is given by a direction vector u (not angles). 方向用向量u表示。
 *  Analytic Jacobian provided for params [cx,cy,cz, ux,uy,uz, scale].
 */
namespace model {
constexpr double MU0 = 1.2566370614359173e-06;
constexpr double Br  = 1.2;
constexpr double R   = 0.0050000000000000001;
constexpr double L   = 0.0040000000000000001;
constexpr int NPTS   = 768;

extern const std::array<std::array<double,3>, NPTS> P_PLUS_LOCAL;
extern const std::array<std::array<double,3>, NPTS> P_MINUS_LOCAL;
extern const std::array<double, NPTS> D_S;

struct Result {
  std::vector<double> B;       // size 3N
  std::vector<double> J;       // size (3N)*7 row-major blocks
};

Result compute_B_and_J(const std::vector<double>& sensors_xyz,
                       const std::array<double,3>& center,
                       const std::array<double,3>& u_raw,
                       double scale);
} // namespace model