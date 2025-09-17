import os
import numpy as np
from .Discretization import DiscGrid, build_end_disks
from .Geometry import CylinderGeom


def write_autogen(geom: CylinderGeom, disc: DiscGrid, outdir: str = "autogen"):
    os.makedirs(outdir, exist_ok=True)
    p_plus, p_minus, dS = build_end_disks(geom, disc)
    N = p_plus.shape[0]
    MU0 = 4 * np.pi * 1e-7

    # ---------------- header ----------------
    hpp = []
    hpp.append('#pragma once')
    hpp.append('#include <array>')
    hpp.append('#include <vector>')
    hpp.append('#include <cmath>')
    hpp.append('')
    hpp.append('/** Model of a uniformly magnetized cylinder using end-surface magnetic charges.')
    hpp.append(' *  采用端面表面磁荷的均匀磁化圆柱模型。')
    hpp.append(' *  Orientation is given by a direction vector u (not angles). 方向用向量u表示。')
    hpp.append(' *  Analytic Jacobian provided for params [cx,cy,cz, ux,uy,uz, scale].')
    hpp.append(' */')
    hpp.append('namespace model {')
    hpp.append(f'constexpr double MU0 = {MU0:.17g};')
    hpp.append(f'constexpr double Br  = {geom.Br:.17g};')
    hpp.append(f'constexpr double R   = {geom.R:.17g};')
    hpp.append(f'constexpr double L   = {geom.l:.17g};')
    hpp.append(f'constexpr int NPTS   = {N};')
    hpp.append('')
    hpp.append('extern const std::array<std::array<double,3>, NPTS> P_PLUS_LOCAL;')
    hpp.append('extern const std::array<std::array<double,3>, NPTS> P_MINUS_LOCAL;')
    hpp.append('extern const std::array<double, NPTS> D_S;')
    hpp.append('')
    hpp.append('struct Result {')
    hpp.append('  std::vector<double> B;       // size 3N')
    hpp.append('  std::vector<double> J;       // size (3N)*7 row-major blocks')
    hpp.append('};')
    hpp.append('')
    hpp.append('Result compute_B_and_J(const std::vector<double>& sensors_xyz,')  # 3N
    hpp.append('                       const std::array<double,3>& center,')
    hpp.append('                       const std::array<double,3>& u_raw,')
    hpp.append('                       double scale);')
    hpp.append('} // namespace model')
    open(os.path.join(outdir, 'include', 'model.hpp'), 'w', encoding='utf-8').write('\n'.join(hpp))

    # ---------------- arrays ----------------
    def fmt_row(v): return '{' + ','.join(f'{x:.17g}' for x in v) + '}'

    plus_lines = ',\n'.join(fmt_row(p) for p in p_plus)
    minus_lines = ',\n'.join(fmt_row(p) for p in p_minus)
    dS_line = ','.join(f'{x:.17g}' for x in dS)

    arrays = []
    arrays.append('#include "model.hpp"')
    arrays.append('namespace model {')
    arrays.append('const std::array<std::array<double,3>, NPTS> P_PLUS_LOCAL = {std::array')
    arrays.append(plus_lines)
    arrays.append('};')
    arrays.append('const std::array<std::array<double,3>, NPTS> P_MINUS_LOCAL = {std::array')
    arrays.append(minus_lines)
    arrays.append('};')
    arrays.append(f'const std::array<double, NPTS> D_S = {{ {dS_line} }};')
    arrays.append('} // namespace model')
    open(os.path.join(outdir, 'src', 'model_data.cpp'), 'w', encoding='utf-8').write('\n'.join(arrays))

    # ---------------- math ----------------
    cpp = r'''#include "model.hpp"
#include <algorithm>
#include <numbers>
namespace model {
    static inline void sph_basis_from_u(const double uhat[3],
                                        double e1[3], double e2[3], double e3[3],
                                        double& alpha, double& beta)
    {
        const double ux=uhat[0], uy=uhat[1], uz=uhat[2];
        alpha = std::acos(std::clamp(uz, -1.0, 1.0));
        beta  = std::atan2(uy, ux);
        const double sa = std::sin(alpha), ca = std::cos(alpha);
        const double cb = std::cos(beta),  sb = std::sin(beta);
        const double e_r[3]    = { sa*cb, sa*sb, ca };
        const double e_theta[3]= { ca*cb, ca*sb,-sa };
        const double e_phi[3]  = { -sb, cb, 0.0 };
        for(int i=0;i<3;++i){ e1[i]=e_phi[i]; e2[i]=-e_theta[i]; e3[i]=e_r[i]; }
    }
    
    static inline void d_basis_dalpha(double alpha, double beta,
                                      double de1[3], double de2[3], double de3[3])
    {
        const double sa = std::sin(alpha), ca = std::cos(alpha);
        const double cb = std::cos(beta),  sb = std::sin(beta);
        const double e_r[3]    = { sa*cb, sa*sb, ca };
        const double e_theta[3]= { ca*cb, ca*sb,-sa };
        const double de_r_da[3]    = { e_theta[0], e_theta[1], e_theta[2] };
        const double de_theta_da[3]= { -e_r[0], -e_r[1], -e_r[2] };
        const double de_phi_da[3]  = { 0.0,0.0,0.0 };
        for(int i=0;i<3;++i){ de1[i]=de_phi_da[i]; de2[i]=-de_theta_da[i]; de3[i]=de_r_da[i]; }
    }
    
    static inline void d_basis_dbeta(double alpha, double beta,
                                     double de1[3], double de2[3], double de3[3])
    {
        const double sa = std::sin(alpha), ca = std::cos(alpha);
        const double cb = std::cos(beta),  sb = std::sin(beta);
        const double e_r[3]    = { sa*cb, sa*sb, ca };
        const double e_theta[3]= { ca*cb, ca*sb,-sa };
        const double e_phi[3]  = { -sb, cb, 0.0 };
        const double de_r_db[3]    = { sa*e_phi[0], sa*e_phi[1], sa*e_phi[2] };
        const double de_theta_db[3]= { ca*e_phi[0], ca*e_phi[1], ca*e_phi[2] };
        const double de_phi_db[3]  = { -sa*e_r[0]-ca*e_theta[0],
                                       -sa*e_r[1]-ca*e_theta[1],
                                       -sa*e_r[2]-ca*e_theta[2] };
        for(int i=0;i<3;++i){ de1[i]=de_phi_db[i]; de2[i]=-de_theta_db[i]; de3[i]=de_r_db[i]; }
    }
    
    static inline void normalize(const double u[3], double uhat[3], double& nr)
    {
        nr = std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        const double inv = 1.0/std::max(nr,1e-30);
        uhat[0]=u[0]*inv; uhat[1]=u[1]*inv; uhat[2]=u[2]*inv;
    }
    
    static inline void kernel_and_J(const double r[3], double f[3], double J[9])
    {
        const double rx=r[0], ry=r[1], rz=r[2];
        const double r2 = rx*rx + ry*ry + rz*rz + 1e-30;
        const double inv_r = 1.0/std::sqrt(r2);
        const double inv_r3 = inv_r*inv_r*inv_r;
        f[0]=rx*inv_r3; f[1]=ry*inv_r3; f[2]=rz*inv_r3;
        const double inv_r5 = inv_r3*inv_r*inv_r;
        J[0]=(r2 - 3*rx*rx)*inv_r5; J[1]=(-3*rx*ry)*inv_r5; J[2]=(-3*rx*rz)*inv_r5;
        J[3]=(-3*ry*rx)*inv_r5; J[4]=(r2 - 3*ry*ry)*inv_r5; J[5]=(-3*ry*rz)*inv_r5;
        J[6]=(-3*rz*rx)*inv_r5; J[7]=(-3*rz*ry)*inv_r5; J[8]=(r2 - 3*rz*rz)*inv_r5;
    }
    
    model::Result compute_B_and_J(const std::vector<double>& sensors_xyz,
                           const std::array<double,3>& center,
                           const std::array<double,3>& u_raw,
                           double scale)
    {
        using std::numbers::pi;
        const int Ns = (int)(sensors_xyz.size()/3);
        Result out;
        out.B.assign(3*Ns, 0.0);
        out.J.assign(3*Ns*7, 0.0);
    
        double uhat[3], nr;
        normalize(u_raw.data(), uhat, nr);
        double e1[3], e2[3], e3[3], alpha, beta;
        sph_basis_from_u(uhat, e1,e2,e3, alpha,beta);
        double de1_da[3],de2_da[3],de3_da[3];
        double de1_db[3],de2_db[3],de3_db[3];
        d_basis_dalpha(alpha,beta, de1_da,de2_da,de3_da);
        d_basis_dbeta(alpha,beta, de1_db,de2_db,de3_db);
    
        const double sa = std::max(std::sin(alpha), 1e-12);
        const double s2 = sa*sa;
        const double dalpha_duhat[3] = {0.0, 0.0, -1.0/sa};
        const double dbeta_duhat[3]  = {-uhat[1]/s2, uhat[0]/s2, 0.0};
        const double inv_nr = 1.0/std::max(nr,1e-30);
        double d_uhat_duraw[9];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            d_uhat_duraw[3*i+j] = (i==j?1.0:0.0) - uhat[i]*uhat[j];
            d_uhat_duraw[3*i+j] *= inv_nr;
        }
        double dalpha_duraw[3]={0}, dbeta_duraw[3]={0};
        for(int k=0;k<3;++k){
            for(int j=0;j<3;++j){
                dalpha_duraw[k] += dalpha_duhat[j]*d_uhat_duraw[3*j+k];
                dbeta_duraw[k]  += dbeta_duhat[j] *d_uhat_duraw[3*j+k];
            }
        }
    
        const double M0 = Br/MU0;
        const double wconst = MU0/(4*pi) * M0 * scale;
    
        auto acc = [&](const std::array<std::array<double,3>, NPTS>& PLOCAL, const int sign){
            for(int i=0;i<Ns;++i){
                double JBc[9] = {0};
                double JB_a[3] = {0};
                double JB_b[3] = {0};
                double Bi[3]  = {0};
                const double sx = sensors_xyz[3*i+0];
                const double sy = sensors_xyz[3*i+1];
                const double sz = sensors_xyz[3*i+2];
                for(int k=0;k<NPTS;++k){
                    const double xL = PLOCAL[k][0];
                    const double yL = PLOCAL[k][1];
                    const double zL = PLOCAL[k][2];
                    const double px = center[0] + e1[0]*xL + e2[0]*yL + e3[0]*zL;
                    const double py = center[1] + e1[1]*xL + e2[1]*yL + e3[1]*zL;
                    const double pz = center[2] + e1[2]*xL + e2[2]*yL + e3[2]*zL;
                    const double r[3] = { sx - px, sy - py, sz - pz };
                    double f[3], J[9];
                    kernel_and_J(r, f, J);
                    const double w = wconst * (double)sign * D_S[k];
                    Bi[0] += w*f[0]; Bi[1] += w*f[1]; Bi[2] += w*f[2];
                    for(int a=0;a<3;++a)for(int b=0;b<3;++b) JBc[3*a+b] += -w * J[3*a+b];
                    const double dP_da[3] = { de1_da[0]*xL + de2_da[0]*yL + de3_da[0]*zL,
                                              de1_da[1]*xL + de2_da[1]*yL + de3_da[1]*zL,
                                              de1_da[2]*xL + de2_da[2]*yL + de3_da[2]*zL };
                    const double dP_db[3] = { de1_db[0]*xL + de2_db[0]*yL + de3_db[0]*zL,
                                              de1_db[1]*xL + de2_db[1]*yL + de3_db[1]*zL,
                                              de1_db[2]*xL + de2_db[2]*yL + de3_db[2]*zL };
                    for(int a=0;a<3;++a){
                        JB_a[a] += w * ( J[3*a+0]*(-dP_da[0]) + J[3*a+1]*(-dP_da[1]) + J[3*a+2]*(-dP_da[2]) );
                        JB_b[a] += w * ( J[3*a+0]*(-dP_db[0]) + J[3*a+1]*(-dP_db[1]) + J[3*a+2]*(-dP_db[2]) );
                    }
                }
                out.B[3*i+0] += Bi[0];
                out.B[3*i+1] += Bi[1];
                out.B[3*i+2] += Bi[2];
                for(int a=0;a<3;++a)for(int b=0;b<3;++b){
                    out.J[(3*i+a)*7 + b] += JBc[3*a+b];
                }
                for(int a=0;a<3;++a){
                    for(int k=0;k<3;++k){
                        out.J[(3*i+a)*7 + (3+k)] += JB_a[a]*dalpha_duraw[k] + JB_b[a]*dbeta_duraw[k];
                    }
                }
                for(int k=0;k<NPTS;++k){
                    const double xL = PLOCAL[k][0];
                    const double yL = PLOCAL[k][1];
                    const double zL = PLOCAL[k][2];
                    const double px = center[0] + e1[0]*xL + e2[0]*yL + e3[0]*zL;
                    const double py = center[1] + e1[1]*xL + e2[1]*yL + e3[1]*zL;
                    const double pz = center[2] + e1[2]*xL + e2[2]*yL + e3[2]*zL;
                    const double r[3] = { sx - px, sy - py, sz - pz };
                    double f[3], Jtmp[9];
                    kernel_and_J(r, f, Jtmp);
                    const double wscale = (double)sign * MU0/(4*pi) * (Br/MU0) * D_S[k];
                    out.J[(3*i+0)*7 + 6] += wscale * f[0];
                    out.J[(3*i+1)*7 + 6] += wscale * f[1];
                    out.J[(3*i+2)*7 + 6] += wscale * f[2];
                }
            }
        };
        acc(P_PLUS_LOCAL, +1);
        acc(P_MINUS_LOCAL, -1);
        return out;
    }
}
'''
    open(os.path.join(outdir, 'src', 'model.cpp'), 'w', encoding='utf-8').write(cpp)

    # demo main

    main = r"""#include "model.hpp"
#include <iostream>
int main(){
  std::vector<double> sensors = {0.03,0.01,0.02, 0.04,0.02,0.03};
  std::array<double,3> c = {0.01,0.01,0.012};
  std::array<double,3> u = {0.0,0.0,1.0};
  auto res = model::compute_B_and_J(sensors, c, u, 1.0);
  std::cout << "B: " << res.B[0] << "," << res.B[1] << "," << res.B[2] << std::endl;
}
"""
    open(os.path.join(outdir, 'src', 'main.cpp'), 'w', encoding='utf-8').write(main)