# Sync+Share Collaboration Guide

This document specifically explains how to use LRZ Sync+Share (or similar file synchronization services) for project collaboration.

---

## ⚠️ Important Reminder

**DO NOT sync the following folders!** They take up significant storage space, and each developer should regenerate them locally:

### 🚫 Directories to EXCLUDE from Sync

| Directory | Reason | Size Estimate |
|-----------|--------|---------------|
| `build/` | CMake build output, locally generated | 100-500 MB |
| `out/` | Compilation output directory | 50-200 MB |
| `Eigen-master/` | Third-party library, download separately | ~40 MB |
| `external/` | Other external dependencies | Varies |
| `.git/` | Git version control data (manage with Git separately) | 10-100 MB |
| `.vs/` | Visual Studio configuration | 10-100 MB |
| `.vscode/` | VS Code configuration | < 5 MB |
| `.idea/` | JetBrains IDE configuration | 5-20 MB |
| `.venv/` | Python virtual environment | 50-500 MB |
| `results/` | Program execution results | Varies |

**Total savings**: Approximately 300 MB - 1.5 GB of storage space!

---

## ✅ Content that SHOULD be Synced

### Source Code and Configuration
- ✅ `src/` - Source code files
- ✅ `include/` - Header files
- ✅ `CMakeLists.txt` - CMake configuration
- ✅ `CMakePresets.json` - CMake presets
- ✅ `.gitignore` - Git ignore rules
- ✅ `README.md` - Project documentation
- ✅ `PROJECT_STRUCTURE.md` - Structure documentation
- ✅ `*.cpp`, `*.h`, `*.hpp` - All code files

### Scripts and Documentation
- ✅ `run`, `run_benchmark` - Run scripts
- ✅ `*.py` - Python scripts (if not too large)
- ✅ `*.md` - Markdown documentation

### Data Files (depending on situation)
- ✅ Small data files (< 10 MB)
- ⚠️ Large datasets (> 50 MB) - Consider using alternative sharing methods

---

## 📋 Sync+Share Setup Steps

### Method 1: Set Exclusion Rules in Sync+Share Client

If you use the PowerFolder client (underlying software of LRZ Sync+Share):

1. Open Sync+Share client
2. Right-click on the synced folder
3. Select "Settings" → "Exclusion Patterns"
4. Add the following exclusion rules:

```
build/
build/*
out/
out/*
Eigen-master/
Eigen-master/*
external/
external/*
.git/
.git/*
.vs/
.vs/*
.vscode/
.vscode/*
.idea/
.idea/*
.venv/
.venv/*
results/
results/*
*.o
*.obj
*.exe
*.dll
*.so
*.dylib
*.a
*.lib
```

### Method 2: Use .syncignore File

Create a `.syncignore` file in the project root directory (if Sync+Share client supports it):

```
# Build output
build/
out/
cmake/

# External libraries
Eigen-master/
external/
third_party/

# IDE configurations
.vs/
.vscode/
.idea/

# Python environment
.venv/
venv/
__pycache__/

# Version control
.git/

# Compiled artifacts
*.o
*.obj
*.exe
*.dll
*.so
*.dylib
*.a
*.lib

# Result files
results/
*.log
```

---

## 🚀 First-time Project Setup Process

When you first obtain the project from Sync+Share, follow these steps:

### Step 1: Sync Project Files

1. Log in to LRZ Sync+Share: https://syncandshare.lrz.de/
2. Find the shared project folder
3. Click "Sync" or download to local

### Step 2: Install Dependencies - Eigen Library

The project requires the Eigen math library, but this library is **NOT in sync scope** and needs to be downloaded separately:

**Windows**:
```cmd
cd C:\path\to\Teil_4_C++
powershell -Command "Invoke-WebRequest -Uri https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip -OutFile eigen-master.zip"
powershell -Command "Expand-Archive -Path eigen-master.zip -DestinationPath ."
```

**Linux/macOS**:
```bash
cd /path/to/Teil_4_C++
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
unzip eigen-master.zip
```

**Or using Git**:
```bash
git clone https://gitlab.com/libeigen/eigen.git Eigen-master
```

### Step 3: Create Build Directory

```bash
mkdir build
cd build
```

### Step 4: Configure and Compile

**Windows (Visual Studio 2022)**:
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**Linux/macOS**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 🔄 Daily Workflow

### Updating Code

1. **Sync latest code from Sync+Share**
   - Let Sync+Share client automatically sync
   - Or manually click "Sync" button

2. **Check for dependency updates**
   - Check if README.md has updates
   - Check if CMakeLists.txt has new dependencies

3. **Rebuild** (if there are changes)
   ```bash
   cd build
   cmake ..
   cmake --build .
   ```

### Committing Your Changes

1. **Ensure not modifying excluded directories**
   - Don't manually create files in `build/`
   - Don't modify contents in `Eigen-master/`

2. **Save your code**
   - Edit code in `src/` or `include/`
   - Sync+Share will automatically sync

3. **(Optional) Use Git**
   - If the team also uses Git, remember to commit and push
   ```bash
   git add src/ include/
   git commit -m "Describe your changes"
   git push
   ```

---

## 🆘 Common Issues

### Q1: Sync+Share has been syncing for a long time, insufficient storage space

**Cause**: May have accidentally synced `build/` or other large directories

**Solution**:
1. Stop syncing
2. Delete `build/`, `Eigen-master/` and other directories locally
3. Set exclusion rules in Sync+Share client (see above)
4. Resync

### Q2: Team member's code cannot compile

**Cause**: May be missing dependency libraries or build configuration

**Solution**:
1. Confirm Eigen library has been downloaded
2. Delete `build/` directory, rebuild
3. Check CMake output to confirm all dependencies are found

### Q3: Sync conflict

**Cause**: Multiple people editing the same file simultaneously

**Solution**:
1. Sync+Share usually creates conflict copies (with timestamp)
2. Manually merge changes
3. Delete conflict copies
4. Or use Git for better version control

### Q4: I accidentally deleted the local build directory, will it affect the team?

**Answer**: No! The `build/` directory should NOT be synced anyway. Deleting your local `build/` will not affect others. You can regenerate it anytime:
```bash
mkdir build && cd build
cmake .. && make
```

---

## 💡 Best Practices

### DO ✅
- ✅ Regularly sync code
- ✅ Keep `.gitignore` updated
- ✅ Test code compiles before committing
- ✅ Write clear code comments
- ✅ Update README.md to document new features

### DON'T ❌
- ❌ Don't sync build output (`build/`, `out/`)
- ❌ Don't sync external libraries (`Eigen-master/`, `external/`)
- ❌ Don't sync IDE configurations (`.vs/`, `.vscode/`)
- ❌ Don't directly edit files in `build/` directory
- ❌ Don't commit untested code

---

## 📞 Getting Help

If you encounter problems:

1. **Check documentation**
   - README.md
   - PROJECT_STRUCTURE.md
   - LRZ Sync+Share documentation: https://doku.lrz.de/

2. **Contact team members**
   - [Project lead email]
   - [Team collaboration platform]

3. **LRZ Support**
   - LRZ Servicedesk: https://doku.lrz.de/sync+share-servicedesks-11476018.html

---

## 📚 Related Resources

- **LRZ Sync+Share Documentation**: https://doku.lrz.de/bayernshare-sync+share-10745682.html
- **LRZ Sync+Share Quick Start**: https://doku.lrz.de/sync+share-schnelleinstieg-11476027.html
- **PowerFolder Open Source Project**: https://github.com/powerfolder/PF-CORE
- **CMake Documentation**: https://cmake.org/documentation/
- **Eigen Documentation**: https://eigen.tuxfamily.org/

---

**Last Updated**: 2025-01-07
