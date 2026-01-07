# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(magnetic_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Platform-specific compiler settings
if(MSVC)
    add_compile_options(
        /W4 /O2 /Oi /Ot /fp:fast /utf-8 /MP
        /wd4100 /wd4127 /wd4244 /wd4267 /wd4459 /wd4819 /wd4828
    )
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(/openmp)
        add_compile_definitions(USE_OPENMP)
        message(STATUS "OpenMP enabled (MSVC)")
    endif()
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
    add_compile_definitions(_USE_MATH_DEFINES)
    
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(-fopenmp)
        link_libraries(OpenMP::OpenMP_CXX)
        add_compile_definitions(USE_OPENMP)
        message(STATUS "OpenMP enabled (GCC/Clang)")
    endif()
endif()

# Find Eigen
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS
            ${CMAKE_SOURCE_DIR}/Eigen
            ${CMAKE_SOURCE_DIR}/Eigen-master
            ${CMAKE_SOURCE_DIR}/eigen
            /usr/include/eigen3
            /usr/local/include/eigen3
    )
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Eigen3 not found!")
    endif()
else()
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)

# ============================================================================
# Executable 1: Original Biot-Savart Tracking
# ============================================================================

set(BIOT_TRACKING_SOURCES
    src/main.cpp
    src/biot_field.cpp
    src/pose_optimizer.cpp
    src/io_utils.cpp
)

add_executable(biot_tracking ${BIOT_TRACKING_SOURCES})

target_include_directories(biot_tracking PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(biot_tracking PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Executable 2: Model Comparison (Biot-Savart vs Dipole)
# ============================================================================

set(MODEL_COMPARISON_SOURCES
    src/main_comparison.cpp
    src/biot_field.cpp
    src/dipole_field.cpp
    src/pose_optimizer_generic.cpp
    src/model_adapters.cpp
    src/io_utils.cpp
)

add_executable(model_comparison ${MODEL_COMPARISON_SOURCES})

target_include_directories(model_comparison PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(model_comparison PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Executable 3: Multi-Core Performance Benchmark
# ============================================================================

set(BENCHMARK_PARALLEL_SOURCES
    benchmark_parallel.cpp
    src/biot_field.cpp
    src/pose_optimizer.cpp
    src/io_utils.cpp
)

add_executable(benchmark_parallel ${BENCHMARK_PARALLEL_SOURCES})

target_include_directories(benchmark_parallel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_parallel PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Configuration Summary")
message(STATUS "========================================")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")

if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP: Enabled")
else()
    message(STATUS "  OpenMP: Disabled")
endif()

if(EIGEN3_INCLUDE_DIR)
    message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

message(STATUS "")
message(STATUS "  Executables to build:")
message(STATUS "    - biot_tracking")
message(STATUS "    - model_comparison")
message(STATUS "    - benchmark_parallel (NEW)")
message(STATUS "========================================")
message(STATUS "")