# Project Structure Documentation

## Directory Structure

```
Teil_4_C++/
├── src/                          # Source code directory
│   └── *.cpp                     # C++ source files
├── include/                      # Header files directory
│   └── *.h, *.hpp                # C++ header files
├── data/                         # Data files directory
│   └── *.csv, *.dat              # Input data files
├── CMakeLists.txt                # CMake main configuration
├── CMakePresets.json             # CMake preset configuration
├── benchmark_parallel.cpp        # Parallel benchmark test
├── run                           # Run script
├── run_benchmark                 # Benchmark run script
├── .gitignore                    # Git ignore configuration
├── .gitmodules                   # Git submodule configuration
├── README.md                     # Project documentation
└── PROJECT_STRUCTURE.md          # This document

# The following directories are NOT under version control (excluded via .gitignore)
├── build/                        # CMake build output (locally generated)
├── out/                          # Compilation output directory (locally generated)
├── results/                      # Program execution results (locally generated)
├── Eigen-master/                 # Eigen math library (external dependency, obtain separately)
├── external/                     # Other external libraries (external dependencies)
├── .git/                         # Git version control data
├── .idea/                        # JetBrains IDE configuration
├── .venv/                        # Python virtual environment (if using Python helper tools)
├── .vs/                          # Visual Studio configuration
└── .vscode/                      # VS Code configuration
```

## Directory Descriptions

### Source Code Directories (SHOULD sync)
- **`src/`**: Contains all C++ source files (.cpp)
- **`include/`**: Contains all header files (.h, .hpp)
- **`data/`**: Contains input data files required by the project
- **`CMakeLists.txt`**: CMake build configuration file
- **`CMakePresets.json`**: CMake preset configuration, defining different build configurations

### Generated/Build Directories (should NOT sync)
- **`build/`**: CMake build output directory, contains all intermediate files during compilation
  - ⚠️ **Locally generated, DO NOT upload!** Each developer should rebuild locally
- **`out/`**: Compilation output executable directory
  - ⚠️ **Locally generated, DO NOT upload!**
- **`results/`**: Result files generated after program execution
  - ⚠️ **Generated at runtime, DO NOT upload!** Unless results need to be shared for analysis

### External Dependencies (should NOT sync)
- **`Eigen-master/`**: Eigen linear algebra library (third-party library)
  - ⚠️ **External dependency, DO NOT upload!** Large size (~40MB), should be downloaded via package manager or script
  - See "Dependencies Installation" section in README.md
- **`external/`**: Other external libraries and dependencies
  - ⚠️ **External dependency, DO NOT upload!**

### IDE/Editor Configurations (should NOT sync)
- **`.vs/`**: Visual Studio 2022 project configuration and cache files
  - ⚠️ **IDE configuration, DO NOT upload!** Each user's IDE settings may differ
- **`.vscode/`**: Visual Studio Code workspace configuration
  - ⚠️ **IDE configuration, DO NOT upload!**
- **`.idea/`**: JetBrains CLion IDE configuration
  - ⚠️ **IDE configuration, DO NOT upload!**

### Python Environment (if used)
- **`.venv/`**: Python virtual environment (if project uses Python helper scripts)
  - ⚠️ **Python environment, DO NOT upload!** Each user should create virtual environment separately

### Version Control
- **`.git/`**: Git version control data
  - ⚠️ **Git internal files!** Git already manages versions, no need for additional sync
- **`.gitignore`**: Defines files and directories Git should ignore
  - ✅ **SHOULD sync!** Ensure team members use the same ignore rules
- **`.gitmodules`**: Git submodule configuration
  - ✅ **SHOULD sync!** If project uses Git submodules

## File Size Estimates

| Directory/File Type | Estimated Size | Should Sync |
|---------------------|---------------|-------------|
| `src/` + `include/` | < 10 MB | ✅ Yes |
| `data/` | Depends on data size | ✅ Yes (if not too large) |
| `build/` | 100-500 MB | ❌ No |
| `Eigen-master/` | ~40 MB | ❌ No |
| `.vs/` | 10-100 MB | ❌ No |
| `.git/` | Depends on history | ❌ No (managed by Git separately) |

## Sync Recommendations

### ✅ Files that SHOULD be synced
- All source code (`src/`, `include/`)
- CMake configuration files (`CMakeLists.txt`, `CMakePresets.json`)
- Documentation files (`README.md`, `*.md`)
- Script files (`run`, `run_benchmark`)
- `.gitignore` file
- Small data files (< 50 MB)

### ❌ Files/directories that should NOT be synced
- Build output (`build/`, `out/`)
- IDE configurations (`.vs/`, `.vscode/`, `.idea/`)
- External libraries (`Eigen-master/`, `external/`)
- Python environment (`.venv/`)
- Git internal files (`.git/`)
- Compiled artifacts (`*.o`, `*.exe`, `*.dll`, etc.)
- Temporary files and logs

## First-time Setup Process

When team members first obtain the project, follow these steps:

1. **Sync/Clone the project** (from Sync+Share or Git)
2. **Install dependencies** (refer to README.md)
3. **Configure build environment**
4. **Execute build**
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## Important Notes

⚠️ **Important Reminders**:
- Build directory (`build/`) may contain hundreds of MB of compilation files, DO NOT sync!
- Eigen library is ~40 MB, should be obtained via script or CMake automatic fetch, DO NOT upload manually!
- If using Sync+Share, ensure these large directories are excluded from sync scope
- If using Git, `.gitignore` is already configured and will automatically ignore these directories

## Data File Management

For the `data/` directory:
- **Small data files** (< 10 MB): Can be synced
- **Large datasets** (> 50 MB): Consider using independent data sharing solution
  - Consider using Git LFS (Large File Storage)
  - Or use independent data storage service
  - Provide data acquisition method in README

## Last Updated
2025-01-07
