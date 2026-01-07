# Teil_4_C++ Project

C++ Parallel Computing and Performance Benchmarking Project

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dependencies Installation](#dependencies-installation)
- [Building the Project](#building-the-project)
- [Running the Program](#running-the-program)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [FAQ](#faq)
- [Collaboration Guide](#collaboration-guide)

---

## Project Overview

This is a high-performance C++ computing project, primarily used for [please fill in according to your actual project content].

The project uses CMake for build management, adopts modern C++ standards (C++17/20), and integrates multiple third-party libraries for efficient numerical computation.

---

## Features

- ✅ Parallel computing implementation
- ✅ Performance benchmarking
- ✅ Data visualization support
- ✅ Statistical analysis tools
- ✅ Cross-platform support (Windows/Linux/macOS)

---

## Requirements

### Compilers
- **Windows**: Visual Studio 2022 (MSVC) or higher
- **Linux**: GCC 9+ or Clang 10+
- **macOS**: Xcode 12+ or Clang 10+

### Build Tools
- **CMake**: 3.15 or higher
- **Make** or **Ninja** (Linux/macOS)

### Python (Optional, for data processing)
- **Python**: 3.8 or higher
- **Recommended**: Python 3.10+

### Other Tools
- **Git**: For version control
- **Git LFS** (if handling large files)

---

## Dependencies Installation

This project depends on the following third-party libraries, which need to be installed before building:

### 1. Eigen Library (Linear Algebra Library)

Eigen is a C++ template library for linear algebra, matrix and vector operations.

#### Method A: Manual Download (Recommended)

**Download latest version**:
```bash
# Method 1: Using wget
cd /path/to/Teil_4_C++
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
unzip eigen-master.zip

# Method 2: Using Git
git clone https://gitlab.com/libeigen/eigen.git Eigen-master
```

**Or download stable version**:
```bash
# Download Eigen 3.4.0 (stable)
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
mv eigen-3.4.0 Eigen-master
```

#### Method B: Using CMake FetchContent (Automatic)

If FetchContent is configured in CMakeLists.txt, CMake will automatically download Eigen during the first build:

```cmake
include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen)
```

For first build, run:
```bash
cmake -B build
```

CMake will automatically download and configure Eigen.

#### Method C: Using Package Managers

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install libeigen3-dev
```

**macOS (Homebrew)**:
```bash
brew install eigen
```

**Windows (vcpkg)**:
```bash
vcpkg install eigen3:x64-windows
```

### 2. Other Dependencies

Depending on project requirements, you may also need:

```bash
# OpenMP (parallel computing support)
# Windows: Built-in with Visual Studio
# Linux:
sudo apt-get install libomp-dev

# Python dependencies (if using Python scripts)
python -m pip install --upgrade pip
pip install numpy matplotlib pandas
```

---

## Building the Project

### Windows (Visual Studio 2022)

#### Using CMake GUI:

1. Open CMake GUI
2. Set source directory: `/path/to/Teil_4_C++`
3. Set build directory: `/path/to/Teil_4_C++/build`
4. Click "Configure", select "Visual Studio 17 2022"
5. Click "Generate"
6. Click "Open Project" to open Visual Studio
7. In Visual Studio, select "Build Solution"

#### Using Command Line (Recommended):

```cmd
# In project root directory
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Linux / macOS

```bash
# Create build directory
mkdir -p build
cd build

# Configure project
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
cmake --build . -j$(nproc)

# Or use make
make -j$(nproc)
```

### Using CMake Presets (Recommended)

If the project includes `CMakePresets.json`:

```bash
# View available presets
cmake --list-presets

# Use preset configuration
cmake --preset=release

# Build
cmake --build --preset=release
```

---

## Running the Program

### Run Main Program

```bash
# Windows
build\Release\main.exe

# Linux/macOS
./build/main
```

### Run Benchmarks

```bash
# Using provided script
./run_benchmark

# Or run directly
./build/benchmark_parallel
```

### Data Visualization

```bash
# Run visualization script
python visualize_comparison.py

# Or
python visualize_benchmark.py
```

---

## Project Structure

```
Teil_4_C++/
├── src/                       # Source code
├── include/                   # Header files
├── data/                      # Data files
├── CMakeLists.txt             # CMake configuration
├── CMakePresets.json          # CMake presets
├── benchmark_parallel.cpp     # Benchmark tests
├── README.md                  # This document
├── PROJECT_STRUCTURE.md       # Detailed structure guide
└── .gitignore                 # Git ignore configuration

# The following directories are NOT under version control
├── build/                     # Build output (locally generated)
├── Eigen-master/              # Eigen library (download separately)
├── external/                  # Other external libraries
├── results/                   # Program output
└── .vs/                       # Visual Studio configuration
```

For detailed directory structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

---

## Development Guide

### Code Style

This project follows these coding standards:
- Use C++17/20 standard
- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use 4-space indentation
- Class names use PascalCase, function names use camelCase

### Adding New Source Files

1. Create `.cpp` file in `src/` directory
2. Create corresponding `.h` header file in `include/` directory
3. Add new file in `CMakeLists.txt`:
   ```cmake
   add_executable(your_program
       src/main.cpp
       src/new_file.cpp  # Add this line
   )
   ```

### Performance Optimization Tips

- Use Release mode for performance testing
- Enable compiler optimization flags: `-O3`
- Use `-march=native` for local CPU optimization
- Consider using OpenMP or TBB for parallelization

---

## FAQ

### Q1: CMake cannot find Eigen library

**Solution**:
```bash
# Method 1: Specify Eigen path
cmake -DEigen3_DIR=/path/to/Eigen-master ..

# Method 2: Set environment variable
export CMAKE_PREFIX_PATH=/path/to/Eigen-master:$CMAKE_PREFIX_PATH
```

### Q2: Visual Studio compilation errors

**Solution**:
- Ensure using Visual Studio 2022 or higher
- Specify correct generator in CMake:
  ```cmd
  cmake -G "Visual Studio 17 2022" -A x64 ..
  ```
- Check if C++ Desktop Development tools are installed

### Q3: Linker error: library file not found

**Solution**:
```bash
# Clean build directory
rm -rf build
mkdir build
cd build

# Reconfigure
cmake ..
```

### Q4: Python script execution fails

**Solution**:
```bash
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # If this file exists
# Or
pip install numpy matplotlib pandas
```

---

## Collaboration Guide

### Using Git Version Control (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd Teil_4_C++

# Install dependencies
# Refer to "Dependencies Installation" section

# Build project
mkdir build && cd build
cmake ..
make
```

### Using Sync+Share File Synchronization

If your team uses LRZ Sync+Share or similar services for collaboration:

#### ⚠️ Important: DO NOT sync these folders
- `build/` - Build output (locally generated, large)
- `Eigen-master/` - Third-party library (download separately)
- `.git/` - Git data (if using Git simultaneously)
- `.vs/`, `.vscode/`, `.idea/` - IDE configurations
- `.venv/` - Python virtual environment
- `results/` - Program output

#### ✅ SHOULD sync these
- `src/` - Source code
- `include/` - Header files
- `CMakeLists.txt` - Build configuration
- `README.md` - Documentation
- `data/` - Data files (if not too large)

#### Setup steps after first checkout:

1. **Download project files from Sync+Share**

2. **Install Eigen library**:
   ```bash
   cd /path/to/Teil_4_C++
   wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
   unzip eigen-master.zip
   ```

3. **Create build directory**:
   ```bash
   mkdir build
   cd build
   ```

4. **Configure and compile**:
   ```bash
   cmake ..
   cmake --build . --config Release
   ```

### Git Workflow

```bash
# Create new branch
git checkout -b feature/your-feature-name

# Commit changes
git add .
git commit -m "Describe your changes"

# Push to remote
git push origin feature/your-feature-name

# Create Pull Request (if using GitHub/GitLab)
```

---

## Performance Benchmarking

Run benchmarks and generate reports:

```bash
# Run benchmarks
./run_benchmark

# View results
cat results/benchmark_results.txt

# Visualize results
python visualize_benchmark.py
```

---

## Documentation

- [Detailed Project Structure](PROJECT_STRUCTURE.md)
- [API Documentation](docs/API.md) (if available)
- [Developer Guide](docs/CONTRIBUTING.md) (if available)

---

## License

[Please choose appropriate license for your project]

---

## Contact

- **Project Lead**: [Your Name]
- **Email**: [Your Email]
- **GitHub/GitLab**: [Repository Link]

---

## Changelog

### 2025-01-07
- Initial version
- Added benchmark functionality
- Improved documentation

---

## Acknowledgments

- Eigen Library: https://eigen.tuxfamily.org/
- CMake: https://cmake.org/
- [Other Contributors]

---

**Last Updated**: 2025-01-07

