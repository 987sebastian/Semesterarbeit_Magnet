#pragma once
// geometry.hpp
// Geometry definitions for cylindrical magnet / 柱形磁体几何定义

namespace biot {

    // Constants / 常数
    constexpr double MU0 = 4.0 * 3.14159265358979323846 * 1e-7; // Vacuum permeability / 真空磁导率

    // Cylinder geometry / 柱形磁体几何
    struct CylinderGeom {
        double Br;  // Remanence [T] / 剩磁
        double R;   // Radius [m] / 半径
        double L;   // Length [m] / 长度

        CylinderGeom(double br = 1.35, double r = 0.004, double L = 0.005)
            : Br(br), R(r), L(L) {
        }
    };

    // Discretization parameters / 离散化参数
    struct DiscGrid {
        int Nr;   // Radial divisions / 径向剖分数
        int Nth;  // Angular divisions / 角向剖分数

        DiscGrid(int nr = 16, int nth = 48) : Nr(nr), Nth(nth) {}
    };

} // namespace biot