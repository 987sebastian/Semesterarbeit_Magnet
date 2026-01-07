# Magnet Pose Tracking Package / 永磁体姿态追踪包

## Overview 概述
- Models: Biot–Savart cylinder (analytic, direction vector **u**) and Dipole.
- Calibration: rotate sensor **B** to a common frame using per-sensor 4×4 transforms (R,t). Position uses **t**.
- Optimization: SciPy least-squares over pose **(p,u,k)** with analytic Jacobians preferred.
- 可视化：生成 JPG 曲线与误差图。

## Layout 目录
```
data/    原始与校准后数据
py/      Python 源码
cpp/     C++17 头与示例
docs/    说明文档
figs/    图片
results/ 结果
```

## Quick start 快速开始
```bash
# Python
cd py
python run_opt.py

# C++
mkdir -p build && cd build
cmake ../cpp && cmake --build . -j
./mag_example
```
