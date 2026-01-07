# Quick Reference Card

## 🚀 Quick Start

### First-time Setup (5 minutes)
```bash
# 1. Download Eigen library
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
unzip eigen-master.zip

# 2. Build project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. Run
./main
```

---

## 🚫 Folders NOT to Sync

```
❌ build/          (100-500 MB)  - Build output
❌ Eigen-master/   (~40 MB)      - Third-party library
❌ .vs/            (10-100 MB)   - VS configuration
❌ .venv/          (50-500 MB)   - Python environment
❌ results/        (Varies)      - Execution results
❌ .git/           (Varies)      - Git data
```

---

## ✅ Files to Sync

```
✅ src/           - Source code
✅ include/       - Header files
✅ CMakeLists.txt - Build configuration
✅ README.md      - Documentation
✅ *.cpp, *.h     - Code files
```

---

## 🛠️ Common Commands

### Build
```bash
# Full build
mkdir build && cd build
cmake .. && make

# Clean and rebuild
rm -rf build && mkdir build && cd build
cmake .. && make
```

### Windows
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

---

## 📦 Dependency Download

### Eigen (Required)
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
unzip eigen-master.zip
```

### Python Dependencies (Optional)
```bash
pip install numpy matplotlib pandas
```

---

## 🔍 Troubleshooting

### CMake cannot find Eigen?
```bash
cmake -DEigen3_DIR=/path/to/Eigen-master ..
```

### Compilation errors?
```bash
# Clean and reconfigure
rm -rf build && mkdir build && cd build
cmake .. && make
```

### Sync+Share insufficient space?
Check if you accidentally synced `build/` directory, delete it!

---

## 📞 Quick Links

- **Project Documentation**: README.md
- **Structure Guide**: PROJECT_STRUCTURE.md
- **Sync Guide**: SYNC_SHARE_GUIDE.md
- **LRZ Sync+Share**: https://syncandshare.lrz.de/
- **Eigen Documentation**: https://eigen.tuxfamily.org/

---

## ⚡ One-click Scripts

### Linux/macOS
Save as `setup.sh`:
```bash
#!/bin/bash
echo "Starting project setup..."

# Download Eigen
if [ ! -d "Eigen-master" ]; then
    echo "Downloading Eigen library..."
    wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
    unzip eigen-master.zip
    rm eigen-master.zip
fi

# Build project
echo "Building project..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "Setup complete!"
echo "Run: cd build && ./main"
```

Run: `chmod +x setup.sh && ./setup.sh`

### Windows
Save as `setup.bat`:
```batch
@echo off
echo Starting project setup...

REM Check Eigen
if not exist "Eigen-master" (
    echo Downloading Eigen library...
    powershell -Command "Invoke-WebRequest -Uri https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip -OutFile eigen-master.zip"
    powershell -Command "Expand-Archive -Path eigen-master.zip -DestinationPath ."
    del eigen-master.zip
)

REM Build project
echo Building project...
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

echo Setup complete!
echo Run: cd build\Release then run main.exe
pause
```

Run: Double-click `setup.bat`

---

## 🔧 CMake Preset Usage

### View Available Presets
```bash
cmake --list-presets
```

### Use Preset
```bash
# Configure with preset
cmake --preset=release

# Build with preset
cmake --build --preset=release
```

---

## 📊 Benchmark Commands

### Run Benchmarks
```bash
# Using script
./run_benchmark

# Direct execution
./build/benchmark_parallel
```

### Visualize Results
```bash
python visualize_comparison.py
python visualize_benchmark.py
```

---

## 🐛 Debug vs Release

### Debug Build (for debugging)
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
gdb ./main
```

### Release Build (for performance)
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./main
```

---

## 📝 Git Quick Commands

### Daily Workflow
```bash
# Update code
git pull

# Check status
git status

# Commit changes
git add src/ include/
git commit -m "Your message"
git push
```

### Branch Management
```bash
# Create new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# Merge branch
git merge feature/new-feature
```

---

## 🔢 File Size Quick Reference

| Item | Size | Sync? |
|------|------|-------|
| Source code | < 10 MB | ✅ |
| build/ | 100-500 MB | ❌ |
| Eigen-master/ | ~40 MB | ❌ |
| .vs/ | 10-100 MB | ❌ |
| .venv/ | 50-500 MB | ❌ |

---

## ⚙️ Environment Setup

### Check Tools
```bash
# Check CMake version
cmake --version

# Check compiler
g++ --version  # Linux
clang --version  # macOS

# Check Python
python --version
```

### Install Missing Tools

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install cmake g++ make
```

**macOS**:
```bash
brew install cmake
```

**Windows**:
- Install Visual Studio 2022
- Install CMake from https://cmake.org/download/

---

## 🎯 Performance Tips

### Compilation Optimization
```bash
# Use all CPU cores
make -j$(nproc)

# Optimize for local CPU
cmake .. -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

### Runtime Optimization
- Always use Release build for benchmarks
- Enable OpenMP for parallel execution
- Profile before optimizing: `perf record ./main`

---

## 📋 Checklist for New Team Members

- [ ] Clone/download project from Sync+Share
- [ ] Download Eigen library
- [ ] Install CMake and compiler
- [ ] Read README.md
- [ ] Build project successfully
- [ ] Run test program
- [ ] Set up exclusion rules in Sync+Share
- [ ] (Optional) Set up Git

---

## 🚨 Emergency Commands

### Project won't compile?
```bash
# Nuclear option: start fresh
rm -rf build Eigen-master
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
unzip eigen-master.zip
mkdir build && cd build
cmake .. && make
```

### Sync issues?
1. Stop Sync+Share client
2. Check exclusion rules
3. Delete large directories locally
4. Restart sync

---

**Last Updated**: 2025-01-07
**For detailed information, see README.md**
