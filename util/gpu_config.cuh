// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_GPU_CONFIG_CUH__
#define __SPPARK_UTIL_GPU_CONFIG_CUH__

#include "gpu_t.cuh"

// GPU-aware configuration system for adaptive kernel launch parameters
struct gpu_config_t {
  // GPU properties
  int compute_capability_major;
  int compute_capability_minor;
  int sm_count;
  int max_threads_per_block;
  size_t shared_mem_per_block;
  size_t shared_mem_per_sm;
  int warp_size;
  int max_blocks_per_sm;
  int max_threads_per_sm;

  // Architecture-specific limits
  size_t max_dynamic_shared_mem;
  int max_occupancy_target;

  // Computed optimal values
  int optimal_block_size;
  int optimal_occupancy;

  gpu_config_t(const stream_t &stream) {
    const cudaDeviceProp &props = gpu_props(stream);

    compute_capability_major = props.major;
    compute_capability_minor = props.minor;
    sm_count = props.multiProcessorCount;
    max_threads_per_block = props.maxThreadsPerBlock;
    shared_mem_per_block = props.sharedMemPerBlock;
    shared_mem_per_sm = props.sharedMemPerMultiprocessor;
    warp_size = props.warpSize;
    max_blocks_per_sm = props.maxBlocksPerMultiProcessor;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;

    // Architecture-specific configurations
    if (compute_capability_major >= 10) {
      // Blackwell (B100, B200, etc.)
      max_dynamic_shared_mem = 256 * 1024; // 256KB - Enhanced shared memory
      max_occupancy_target = 24; // Improved scheduler supports more blocks
      optimal_block_size = 512; // Larger blocks for better parallelism
    } else if (compute_capability_major >= 9) {
      // Hopper (H100, etc.)
      max_dynamic_shared_mem = 228 * 1024; // 228KB
      max_occupancy_target = 16;
      optimal_block_size = 256;
    } else if (compute_capability_major >= 8) {
      // Ampere (A100, RTX 30xx, etc.)
      max_dynamic_shared_mem = 164 * 1024; // 164KB
      max_occupancy_target = 16;
      optimal_block_size = 256;
    } else if (compute_capability_major >= 7) {
      // Volta/Turing (V100, RTX 20xx, etc.)
      max_dynamic_shared_mem = 96 * 1024; // 96KB
      max_occupancy_target = 8;
      optimal_block_size = 256;
    } else {
      // Older architectures (Pascal, etc.)
      max_dynamic_shared_mem = 48 * 1024; // 48KB
      max_occupancy_target = 4;
      optimal_block_size = 128;
    }

    // Compute optimal occupancy based on available resources
    optimal_occupancy = max_occupancy_target;
  }

  // Get safe block size for a given kernel with register pressure
  int get_safe_block_size(int max_threads,
                          int estimated_registers_per_thread = 32) const {
    // Estimate registers per SM
    int estimated_registers_per_sm =
        estimated_registers_per_thread * max_threads;
    int max_registers_per_sm = 65536; // Typical limit

    // Calculate occupancy based on register limits
    int occupancy_by_registers =
        max_registers_per_sm / estimated_registers_per_sm;
    int occupancy_by_threads = max_threads_per_sm / max_threads;
    int occupancy_by_blocks = max_blocks_per_sm;

    // Use the most restrictive limit
    int max_occupancy = std::min(
        {occupancy_by_registers, occupancy_by_threads, occupancy_by_blocks});

    // If we can't achieve minimum occupancy, reduce block size
    if (max_occupancy < 2 && max_threads > 64) {
      return get_safe_block_size(max_threads / 2,
                                 estimated_registers_per_thread);
    }

    return max_threads;
  }

  // Get maximum safe shared memory per block
  size_t get_safe_shared_mem(size_t requested,
                             int num_blocks_per_sm = 1) const {
    size_t per_block_limit = shared_mem_per_block;
    size_t per_sm_limit = shared_mem_per_sm / num_blocks_per_sm;
    return std::min(std::min(requested, per_block_limit), per_sm_limit);
  }

  // Get optimal number of blocks for a given workload
  uint32_t get_optimal_blocks(uint32_t desired_blocks,
                              uint32_t threads_per_block) const {
    // Calculate based on SM count and occupancy
    uint32_t max_concurrent_blocks = sm_count * max_blocks_per_sm;
    uint32_t max_by_threads =
        (max_threads_per_sm * sm_count) / threads_per_block;

    uint32_t max_blocks = (max_concurrent_blocks < max_by_threads)
                              ? max_concurrent_blocks
                              : max_by_threads;

    // For resource-constrained kernels, be more conservative
    if (threads_per_block <= 64) {
      max_blocks = std::min(max_blocks, (uint32_t)(sm_count * 4));
    }

    return (desired_blocks < max_blocks) ? desired_blocks : max_blocks;
  }

  // Check if GPU supports large shared memory allocations
  bool supports_large_shared_mem() const {
    return compute_capability_major >= 7;
  }

  // Get architecture-specific launch bounds recommendation
  int get_recommended_launch_bounds_max_threads() const {
    if (compute_capability_major >= 10) {
      return 1024; // Blackwell: enhanced scheduler handles very large blocks
    } else if (compute_capability_major >= 9) {
      return 512; // Hopper: can handle larger blocks
    } else if (compute_capability_major >= 8) {
      return 512; // Ampere: can handle larger blocks
    } else if (compute_capability_major >= 7) {
      return 256; // Volta/Turing
    } else {
      return 128; // Older architectures
    }
  }

  int get_recommended_launch_bounds_occupancy() const {
    return max_occupancy_target;
  }

  // Get safe configuration for LDE_distribute_powers kernel
  struct lde_config_t {
    uint32_t threads_per_block;
    uint32_t max_blocks;
    int launch_bounds_max_threads;
    int launch_bounds_occupancy;
  };

  lde_config_t get_lde_config(uint32_t domain_size,
                              uint32_t lg_domain_size) const {
    lde_config_t config;

    // Start with conservative values
    config.launch_bounds_max_threads = 32;
    config.launch_bounds_occupancy = 16;

    // Adjust based on GPU capabilities
    if (compute_capability_major >= 10) {
      // Blackwell: massive parallelism with enhanced register file
      config.launch_bounds_max_threads = 256;
      config.launch_bounds_occupancy = 8;
    } else if (compute_capability_major >= 9) {
      // Hopper: high thread count
      config.launch_bounds_max_threads = 128;
      config.launch_bounds_occupancy = 8;
    } else if (compute_capability_major >= 8) {
      // Ampere: can use more threads
      config.launch_bounds_max_threads = 64;
      config.launch_bounds_occupancy = 8;
    } else if (compute_capability_major >= 7) {
      // Volta/Turing: moderate
      config.launch_bounds_max_threads = 32;
      config.launch_bounds_occupancy = 8;
    }

    // Calculate optimal thread count per block
    if (lg_domain_size >= 14) {
      // For large domains, use smaller blocks to avoid resource exhaustion
      config.threads_per_block = std::min(
          (uint32_t)config.launch_bounds_max_threads, (uint32_t)warp_size * 2);
    } else {
      config.threads_per_block = warp_size * 4; // 128 threads
    }

    // Calculate optimal number of blocks
    uint32_t desired_blocks =
        (domain_size + config.threads_per_block - 1) / config.threads_per_block;
    config.max_blocks =
        get_optimal_blocks(desired_blocks, config.threads_per_block);

    // Additional safety for very large domains
    if (lg_domain_size >= 14) {
      if (config.max_blocks > 256)
        config.max_blocks = 256;
    }

    return config;
  }

  // Get safe configuration for CT_NTT kernel
  struct ctt_config_t {
    uint32_t max_block_size;
    size_t max_shared_mem;
    bool use_coalesced_access;
  };

  ctt_config_t get_ctt_config(uint32_t lg_domain_size) const {
    ctt_config_t config;

    if (compute_capability_major >= 10) {
      // Blackwell: leverage massive shared memory and bandwidth
      if (lg_domain_size >= 14) {
        config.max_block_size = 512;
        config.max_shared_mem = get_safe_shared_mem(64 * 1024, 2);
      } else {
        config.max_block_size = 1024;
        config.max_shared_mem = get_safe_shared_mem(96 * 1024, 4);
      }
    } else if (lg_domain_size >= 14) {
      // Conservative settings for large domains
      config.max_block_size = 256;
      config.max_shared_mem = get_safe_shared_mem(32 * 1024, 2);
    } else {
      config.max_block_size = get_recommended_launch_bounds_max_threads();
      config.max_shared_mem = get_safe_shared_mem(48 * 1024, 4);
    }

    // Use coalesced access for modern architectures with good memory bandwidth
    // Blackwell: use coalesced access even for smaller domains (>= 8)
    // Volta+: use coalesced access for medium+ domains (>= 10)
    if (compute_capability_major >= 10) {
      config.use_coalesced_access = (lg_domain_size >= 8);
    } else {
      config.use_coalesced_access =
          (compute_capability_major >= 7 && lg_domain_size >= 10);
    }

    return config;
  }
};

// Helper function to get GPU config from stream
inline gpu_config_t get_gpu_config(const stream_t &stream) {
  return gpu_config_t(stream);
}

#endif // __SPPARK_UTIL_GPU_CONFIG_CUH__
