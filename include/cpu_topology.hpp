// cpu_topology.hpp
// CPU topology detection for hybrid architecture (P-cores + E-cores)
#pragma once
#include <vector>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>

struct CPUTopology {
    int total_logical_processors;
    int performance_cores_threads;
    int efficiency_cores_threads;
    int physical_cores;
    bool has_hybrid_architecture;
};

inline CPUTopology detect_cpu_topology() {
    CPUTopology topology = { 0, 0, 0, 0, false };

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    topology.total_logical_processors = sysInfo.dwNumberOfProcessors;

    // Get detailed CPU information
    DWORD bufferSize = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferSize);

    if (bufferSize == 0) {
        // Fallback: assume all cores are performance cores
        topology.performance_cores_threads = topology.total_logical_processors;
        topology.physical_cores = topology.total_logical_processors;
        return topology;
    }

    std::vector<BYTE> buffer(bufferSize);
    auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bufferSize)) {
        topology.performance_cores_threads = topology.total_logical_processors;
        topology.physical_cores = topology.total_logical_processors;
        return topology;
    }

    int p_cores = 0, e_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX current = info;
    size_t offset = 0;

    while (offset < bufferSize) {
        if (current->Relationship == RelationProcessorCore) {
            DWORD thread_count = 0;
            DWORD64 mask = current->Processor.GroupMask[0].Mask;

            // Count threads for this core
            while (mask) {
                thread_count += (mask & 1);
                mask >>= 1;
            }

            // Intel 12th gen+: P-cores support HT (2 threads), E-cores don't (1 thread)
            if (thread_count == 2) {
                p_cores++;
            }
            else if (thread_count == 1) {
                e_cores++;
            }

            topology.physical_cores++;
        }

        offset += current->Size;
        current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            reinterpret_cast<BYTE*>(current) + current->Size);
    }

    topology.performance_cores_threads = p_cores * 2;
    topology.efficiency_cores_threads = e_cores;
    topology.has_hybrid_architecture = (e_cores > 0);

    // Verify total matches
    int calculated_total = topology.performance_cores_threads + topology.efficiency_cores_threads;
    if (calculated_total != topology.total_logical_processors) {
        // Detection failed, use conservative estimate
        std::cerr << "[Warning] CPU topology detection mismatch. Using fallback.\n";
        std::cerr << "  Calculated: " << calculated_total
            << ", Expected: " << topology.total_logical_processors << "\n";
        topology.performance_cores_threads = topology.total_logical_processors;
        topology.efficiency_cores_threads = 0;
        topology.has_hybrid_architecture = false;
    }

    return topology;
}

inline void print_cpu_topology(const CPUTopology& topo) {
    std::cout << "CPU Topology Detection:\n";
    std::cout << "  Total logical processors: " << topo.total_logical_processors << "\n";
    std::cout << "  Physical cores: " << topo.physical_cores << "\n";

    if (topo.has_hybrid_architecture) {
        std::cout << "  Hybrid architecture detected:\n";
        std::cout << "    - Performance cores (P-cores): "
            << topo.performance_cores_threads / 2 << " cores * 2 threads = "
            << topo.performance_cores_threads << " threads\n";
        std::cout << "    - Efficiency cores (E-cores): "
            << topo.efficiency_cores_threads << " cores * 1 thread = "
            << topo.efficiency_cores_threads << " threads\n";
    }
    else {
        std::cout << "  Uniform architecture (all cores support HT/SMT)\n";
        std::cout << "    - Using " << topo.performance_cores_threads << " threads\n";
    }
}

#else
// Linux/macOS fallback
#ifdef USE_OPENMP
#include <omp.h>
#endif

struct CPUTopology {
    int total_logical_processors;
    int performance_cores_threads;
    int efficiency_cores_threads;
    int physical_cores;
    bool has_hybrid_architecture;
};

inline CPUTopology detect_cpu_topology() {
    CPUTopology topology;

#ifdef USE_OPENMP
    topology.total_logical_processors = omp_get_max_threads();
#else
    topology.total_logical_processors = 1;
#endif

    topology.performance_cores_threads = topology.total_logical_processors;
    topology.efficiency_cores_threads = 0;
    topology.physical_cores = topology.total_logical_processors;
    topology.has_hybrid_architecture = false;

    return topology;
}

inline void print_cpu_topology(const CPUTopology& topo) {
    std::cout << "CPU Topology Detection (Linux/macOS):\n";
    std::cout << "  Total logical processors: " << topo.total_logical_processors << "\n";
    std::cout << "  Note: Detailed topology detection not available on this platform\n";
}
#endif