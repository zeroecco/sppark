// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_BATCH_ADDITION_CUH__
#define __SPPARK_MSM_BATCH_ADDITION_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <vector>

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#define BATCH_ADD_BLOCK_SIZE 256
#ifndef BATCH_ADD_NSTREAMS
# define BATCH_ADD_NSTREAMS 8
#elif BATCH_ADD_NSTREAMS == 0
# error "invalid BATCH_ADD_NSTREAMS"
#endif

template<class bucket_t, class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__device__ __forceinline__
static void add(bucket_h ret[], const affine_h points[], uint32_t npoints,
                const uint32_t bitmap[], const uint32_t refmap[],
                bool accumulate, uint32_t sid)
{
    // Use shared memory to reduce contention on atomics
    __shared__ uint32_t shared_base;
    static __device__ uint32_t streams[BATCH_ADD_NSTREAMS];
    uint32_t& current = streams[sid % BATCH_ADD_NSTREAMS];

    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
    const uint32_t xid = tid % warp_sz;

    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));

    // Prefetch accumulator to reduce memory latency
    bucket_t acc;
    acc.inf();

    if (accumulate && tid < gridDim.x*blockDim.x/WARP_SZ)
        acc = ret[tid];

    // Use thread with laneid 0 to update shared value first
    if (laneid == 0) {
        if (threadIdx.x == 0) {
            shared_base = atomicAdd(&current, 32*WARP_SZ);
        } else {
            shared_base = atomicAdd(&current, 32*WARP_SZ);
        }
    }

    // Broadcast base to all threads in warp
    uint32_t base = __shfl_sync(0xffffffff, shared_base, 0);

    // Precompute chunk size to avoid recomputation in loop
    uint32_t chunk = min(32*WARP_SZ, npoints-base);
    uint32_t bits, refs, word, sign = 0;

    // Use register to track offset with sentinel value
    uint32_t off = 0xffffffff;

    // Main processing loop with optimized control flow
    for (uint32_t i = 0, j = 0; base < npoints;) {
        // Load bitmap/refmap data at the start of each new chunk
        if (i == 0) {
            // Coalesced loads from global memory
            bits = bitmap[base/WARP_SZ + laneid];
            refs = refmap ? refmap[base/WARP_SZ + laneid] : 0;

            // Compute masks once to avoid redundant operations
            bits ^= refs;
            refs &= bits;
        }

        // Process multiple elements in each iteration for better ILP
        for (; i < chunk && j < warp_sz; i++) {
            // Fetch word and sign values from other threads in warp
            if (i%32 == 0) {
                word = __shfl_sync(0xffffffff, bits, i/32);
                if (refmap) {
                    sign = __shfl_sync(0xffffffff, refs, i/32);
                }
            }

            // Check if point should be processed
            bool process_point = (word & 1);
            if (process_point) {
                if (j++ == xid) {
                    // Store offset with sign bit for later processing
                    off = (base + i) | (sign << 31);
                }
            }

            // Shift word and sign for next iteration
            word >>= 1;
            sign >>= 1;
        }

        // Get next chunk when current chunk is fully processed
        if (i == chunk) {
            if (laneid == 0) {
                shared_base = atomicAdd(&current, 32*WARP_SZ);
            }

            // Broadcast new base to all threads in warp
            base = __shfl_sync(0xffffffff, shared_base, 0);

            // Update chunk size for new base
            chunk = min(32*WARP_SZ, npoints-base);
            i = 0;
        }

        // Process accumulated point when warp is full or no more points
        if (base >= npoints || j == warp_sz) {
            if (off != 0xffffffff) {
                // Load point data with coalesced access pattern
                affine_t p = points[off & 0x7fffffff];

                // Use degree-specific addition to avoid branching
                if (degree == 2)
                    acc.uadd(p, off >> 31);
                else
                    acc.add(p, off >> 31);

                // Reset offset for next batch
                off = 0xffffffff;
            }
            j = 0;
        }
    }

#ifdef __CUDA_ARCH__
    // Parallel reduction within warp using butterfly pattern
    for (uint32_t off = 1; off < warp_sz;) {
        // Use register to hold intermediate value
        bucket_t down = acc.shfl_down(off*degree);

        // Double offset for next iteration (butterfly pattern)
        off <<= 1;

        // Add only if thread should participate in this step
        if ((xid & (off-1)) == 0)
            acc.uadd(down); // .add() triggers spills in .shfl_down()
    }
#endif

    // Ensure all threads finish computation before writing result
    cooperative_groups::this_grid().sync();

    // Only first thread in each warp group writes final result
    if (xid == 0)
        ret[tid/warp_sz] = acc;

    // Reset counter for next kernel invocation
    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template<class bucket_t, class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__launch_bounds__(BATCH_ADD_BLOCK_SIZE, 2) __global__  // Increased occupancy to 2
void batch_addition(bucket_h ret[], const affine_h points[], uint32_t npoints,
                    const uint32_t bitmap[], bool accumulate = false,
                    uint32_t sid = 0)
{   add<bucket_t>(ret, points, npoints, bitmap, nullptr, accumulate, sid);   }

template<class bucket_t, class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__launch_bounds__(BATCH_ADD_BLOCK_SIZE, 2) __global__  // Increased occupancy to 2
void batch_diff(bucket_h ret[], const affine_h points[], uint32_t npoints,
                const uint32_t bitmap[], const uint32_t refmap[],
                bool accumulate = false, uint32_t sid = 0)
{   add<bucket_t>(ret, points, npoints, bitmap, refmap, accumulate, sid);   }

template<class bucket_t, class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__launch_bounds__(BATCH_ADD_BLOCK_SIZE, 2) __global__  // Increased occupancy to 2
void batch_addition(bucket_h ret[], const affine_h points[], size_t npoints,
                    const uint32_t digits[], const uint32_t& ndigits)
{
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
    const uint32_t xid = tid % warp_sz;

    // Initialize accumulator once
    bucket_t acc;
    acc.inf();

    // Process batches of digits for better memory access pattern
    for (size_t i = tid; i < ndigits; i += gridDim.x*blockDim.x/degree) {
        uint32_t digit = digits[i];
        // Coalesced load from global memory
        affine_t p = points[digit & 0x7fffffff];

        // Use degree-specific addition
        if (degree == 2)
            acc.uadd(p, digit >> 31);
        else
            acc.add(p, digit >> 31);
    }

#ifdef __CUDA_ARCH__
    // Parallel reduction with butterfly pattern
    for (uint32_t off = 1; off < warp_sz;) {
        bucket_t down = acc.shfl_down(off*degree);

        off <<= 1;
        if ((xid & (off-1)) == 0)
            acc.uadd(down); // .add() triggers spills ... in .shfl_down()
    }
#endif

    // Only first thread in each warp group writes result
    if (xid == 0)
        ret[tid/warp_sz] = acc;
}

template<class bucket_t>
bucket_t sum_up(const bucket_t inp[], size_t n)
{
    bucket_t sum = inp[0];
    // Use larger step size for initial passes with more elements
    const size_t BLOCK_SIZE = 16;

    if (n > BLOCK_SIZE*2) {
        // Two-stage reduction for better parallelism
        for (size_t i = 1; i < n/BLOCK_SIZE; i++) {
            bucket_t block_sum = inp[i*BLOCK_SIZE];
            for (size_t j = 1; j < BLOCK_SIZE && (i*BLOCK_SIZE+j < n); j++) {
                block_sum.add(inp[i*BLOCK_SIZE+j]);
            }
            sum.add(block_sum);
        }

        // Handle remaining elements
        for (size_t i = (n/BLOCK_SIZE)*BLOCK_SIZE; i < n; i++) {
            sum.add(inp[i]);
        }
    } else {
        // Simple linear reduction for small arrays
        for (size_t i = 1; i < n; i++)
            sum.add(inp[i]);
    }
    return sum;
}

template<class bucket_t>
bucket_t sum_up(const std::vector<bucket_t>& inp)
{   return sum_up(&inp[0], inp.size());   }
#endif
