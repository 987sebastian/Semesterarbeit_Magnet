# CMakeLists.txt for Model Comparison
# Biot-Savart vs Dipole模型对比

cmake_minimum_required(VERSION 3.15)
project(magnetic_tracking_comparison LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific settings
if(MSVC)
    # Windows with Visual Studio
    add_compile_options(
        /W4                  # Warning level 4
        /O2                  # Optimize for speed
        /Oi                  # Enable intrinsic functions
        /Ot                  # Favor fast code
        /fp:fast             # Fast floating point
        /utf-8               # UTF-8 encoding
        # Disable specific warnings
        /wd4100              # Unreferenced formal parameter
        /wd4127              # Conditional expression is constant
        /wd4244              # Conversion possible loss of data
        /wd4267              # Conversion size_t to int
        /wd4459              # Declaration hides global declaration
        /wd4819              # Character encoding
        /wd4828              # Invalid characters
    )
    add_compile_definitions(_USE_MATH_DEFINES)
    
    # OpenMP for MSVC
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(-openmp)
        add_compile_definitions(USE_OPENMP)
        message(STATUS "OpenMP found and enabled (MSVC)")
    endif()
else()
    # Unix-like systems (Linux, macOS, MinGW)
    add_compile_options(
        -Wall
        -Wextra
        -O3
        -march=native
        -ffast-math
    )
    add_compile_definitions(_USE_MATH_DEFINES)
    
    # OpenMP for GCC/Clang
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(-fopenmp)
        link_libraries(OpenMP::OpenMP_CXX)
        add_compile_definitions(USE_OPENMP)
        message(STATUS "OpenMP found and enabled (GCC/Clang)")
    endif()
endif()

# Eigen library (header-only)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Try to find Eigen in common locations
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS
            ${CMAKE_SOURCE_DIR}/Eigen-master
            ${CMAKE_SOURCE_DIR}/eigen
            ${CMAKE_SOURCE_DIR}/../Eigen-master
            /usr/include/eigen3
            /usr/local/include/eigen3
    )
    if(EIGEN3_INCLUDE_DIR)
        include_directories(${EIGEN3_INCLUDE_DIR})
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Eigen3 not found! Please install Eigen or set EIGEN3_INCLUDE_DIR")
    endif()
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# JSON library (header-only, bundled in project)
include_directories(${CMAKE_SOURCE_DIR})

# Source files for original Biot-Savart tracking
set(BIOT_TRACKING_SOURCES
    src/main.cpp
    src/biot_field.cpp
    src/pose_optimizer.cpp
    src/io_utils.cpp
)

# Source files for model comparison
set(COMPARISON_SOURCES
    main_comparison.cpp
    dipole_field.cpp
    model_adapters.cpp
    pose_optimizer_generic.cpp
    src/biot_field.cpp
    src/io_utils.cpp
)

# Original Biot-Savart executable
add_executable(biot_tracking ${BIOT_TRACKING_SOURCES})
target_link_libraries(biot_tracking PRIVATE nlohmann_json::nlohmann_json)

# Model comparison executable
add_executable(model_comparison ${COMPARISON_SOURCES})

# Link nlohmann_json if available as target
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(model_comparison PRIVATE nlohmann_json::nlohmann_json)
endif()

# Output directory for results
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)

# Installation
install(TARGETS biot_tracking model_comparison
        RUNTIME DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP: Enabled")
else()
    message(STATUS "  OpenMP: Disabled")
endif()
message(STATUS "")