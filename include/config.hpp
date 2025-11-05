// config.hpp
// Configuration parameters / 配置参数
#pragma once
#include <string>

namespace biot {
    namespace config {
        // File paths / 文件路径
        const std::string CALIBRATION_FILE = "data/current_calibration_Biot.json";
        const std::string OBSERVATION_FILE = "data/Observational_data.csv";
        const std::string OUTPUT_DIR = "results";
        const std::string OUTPUT_SUMMARY = "results/opt_summary.csv";
        const std::string OUTPUT_ERRORS = "results/pose_error_summary.csv";

        // Magnet geometry / 磁体几何参数
        namespace magnet {
            const double BR = 1.2;       // Remanence [T] / 剩磁
            const double RADIUS = 0.004; // Radius [m] / 半径 (4mm)
            const double LENGTH = 0.005; // Length [m] / 长度 (5mm)
        }

        // Discretization / 离散化参数
        namespace mesh {
            const int NR = 12;   // Radial divisions / 径向剖分数
            const int NTH = 36;  // Angular divisions / 角向剖分数
        }

        // Optimization / 优化参数
        namespace optimizer {
            const int MAX_ITERATIONS = 15;       // Maximum iterations / 最大迭代次数
            const double TOLERANCE = 5e-5;       // Relaxed tolerance [T] / 放宽容差 (50 uT)
            const double MIN_SCALE = 0.01;        // Minimum scale factor / 最小尺度因子
            const double MAX_SCALE = 10.0;       // Maximum scale factor / 最大尺度因子

            // Levenberg-Marquardt damping / LM阻尼参数
            const double LAMBDA_INIT = 1e-3;     // Initial damping / 初始阻尼
            const double LAMBDA_UP = 10.0;       // Damping increase / 阻尼增大倍数
            const double LAMBDA_DOWN = 0.1;      // Damping decrease / 阻尼减小倍数
            const double LAMBDA_MAX = 1e6;       // Maximum damping / 最大阻尼
            const double LAMBDA_MIN = 1e-7;      // Minimum damping / 最小阻尼

            // Trust region and step control / 信赖域和步长控制
            const double TRUST_RADIUS = 0.01;    // Trust region radius [m] / 信赖域半径10mm
            const double MIN_STEP_QUALITY = 0.25; // Minimum gain ratio / 最小增益比
            const int MAX_REJECTIONS = 5;        // Max consecutive rejections / 最大连续拒绝
            const double DIVERGENCE_FACTOR = 2.0; // Cost increase threshold / 发散阈值
        }

        // Numerical differentiation / 数值微分
        namespace numerical {
            const double FINITE_DIFF_EPS = 1e-6;  // Finite difference epsilon / 有限差分步长
        }

        // Output / 输出设置
        namespace output {
            const int PRECISION = 10;            // CSV output precision / CSV输出精度
            const int PROGRESS_INTERVAL = 5;     // Print progress interval / 打印进度间隔
            const bool VERBOSE = false;          // Verbose output (disable for parallel) / 详细输出
        }

        // OpenMP / 并行计算
        namespace parallel {
            const int NUM_THREADS = 10;          // Number of threads / 线程数
            const bool ENABLE_ROW_PARALLEL = true;       // Parallelize rows / 并行行处理
            const bool ENABLE_SENSOR_PARALLEL = false;   // Disable nested / 禁用嵌套
            const bool ENABLE_JACOBIAN_PARALLEL = false; // Disable nested / 禁用嵌套
            const int CHUNK_SIZE = 8;            // Schedule chunk size / 调度块大小
        }
    } // namespace config
} // namespace biot