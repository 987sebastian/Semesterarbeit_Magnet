#pragma once
#include <vector>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#include <sysinfoapi.h>

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

    // 获取详细CPU信息
    DWORD bufferSize = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferSize);

    if (bufferSize == 0) {
        // Fallback: 假设全是性能核
        topology.performance_cores_threads = topology.total_logical_processors;
        return topology;
    }

    std::vector<BYTE> buffer(bufferSize);
    auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bufferSize)) {
        topology.performance_cores_threads = topology.total_logical_processors;
        return topology;
    }

    int p_cores = 0, e_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX current = info;
    size_t offset = 0;

    while (offset < bufferSize) {
        if (current->Relationship == RelationProcessorCore) {
            DWORD thread_count = 0;
            DWORD64 mask = current->Processor.GroupMask[0].Mask;

            // 计算该核心的线程数
            while (mask) {
                thread_count += (mask & 1);
                mask >>= 1;
            }

            // Intel 12代+: 性能核支持超线程(2线程), 能效核不支持(1线程)
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

    // 验证总数
    int calculated_total = topology.performance_cores_threads + topology.efficiency_cores_threads;
    if (calculated_total != topology.total_logical_processors) {
        // 检测失败, 使用保守估计
        std::cerr << "[Warning] CPU topology detection mismatch. Using fallback.\n";
        topology.performance_cores_threads = topology.total_logical_processors;
        topology.efficiency_cores_threads = 0;
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
            << topo.performance_cores_threads / 2 << " cores × 2 threads = "
            << topo.performance_cores_threads << " threads\n";
        std::cout << "    - Efficiency cores (E-cores): "
            << topo.efficiency_cores_threads << " cores × 1 thread = "
            << topo.efficiency_cores_threads << " threads\n";
    }
    else {
        std::cout << "  Uniform architecture (all cores support HT/SMT)\n";
        std::cout << "    - Using " << topo.performance_cores_threads << " threads\n";
    }
}

#else
// Linux/macOS fallback
inline CPUTopology detect_cpu_topology() {
    CPUTopology topology;
    topology.total_logical_processors = omp_get_max_threads();
    topology.performance_cores_threads = topology.total_logical_processors;
    topology.efficiency_cores_threads = 0;
    topology.physical_cores = topology.total_logical_processors;
    topology.has_hybrid_architecture = false;
    return topology;
}

inline void print_cpu_topology(const CPUTopology& topo) {
    std::cout << "CPU Topology Detection (Linux/macOS):\n";
    std::cout << "  Total logical processors: " << topo.total_logical_processors << "\n";
}
#endif