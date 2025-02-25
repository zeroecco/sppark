// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_SORT_CUH__
#define __SPPARK_MSM_SORT_CUH__

/*
 * Custom sorting, we take in digits and return their indices.
 */

#define SORT_BLOCKDIM 1024
#ifndef DIGIT_BITS
# define DIGIT_BITS 13
#endif
#if DIGIT_BITS < 10 || DIGIT_BITS > 14
# error "impossible DIGIT_BITS"
#endif

__launch_bounds__(SORT_BLOCKDIM, 2)
__global__ void sort(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1);

#ifndef __MSM_SORT_DONT_IMPLEMENT__

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

static const uint32_t N_COUNTERS = 1<<DIGIT_BITS;
static const uint32_t N_SUMS = N_COUNTERS / SORT_BLOCKDIM;
extern __shared__ uint32_t counters[/*N_COUNTERS*/];

__device__ __forceinline__
uint32_t pack(uint32_t a, uint32_t mask, uint32_t b)
{
    uint32_t ret;

    asm("lop3.b32 %0, %1, %2, %3, 0xb8;" // a & ~mask | mask & b
        : "=r"(ret)
        : "r"(a), "r"(mask), "r"(b));

    return ret;
}

__device__ __forceinline__
uint32_t sum_up(uint32_t sum, const uint32_t limit = WARP_SZ)
{
    #pragma unroll
    for (uint32_t off = 1; off < limit; off <<= 1)
        asm("{ .reg.b32 %v; .reg.pred %did;"
            "  shfl.sync.up.b32 %v|%did, %0, %1, 0, 0xffffffff;"
            "  @%did add.u32 %0, %0, %v;"
            "}" : "+r"(sum) : "r"(off));

    return sum;
}

__device__ __forceinline__
void zero_counters()
{
#if DIGIT_BITS >= 12
    // Use vectorized stores for better memory throughput
    uint4 zeros = {0, 0, 0, 0};
    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS/4; i++)
        ((uint4*)counters)[threadIdx.x + i*SORT_BLOCKDIM] = zeros;
#else
    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++)
        counters[threadIdx.x + i*SORT_BLOCKDIM] = 0;
#endif
    __syncthreads();
}

__device__ __forceinline__
void count_digits(const uint32_t src[], uint32_t base, uint32_t len,
                  uint32_t lshift, uint32_t rshift, uint32_t mask)
{
    zero_counters();

    const uint32_t pack_mask = 0xffffffffU << lshift;

    src += base;
    // Calculate stride for better coalescing
    const uint32_t stride = SORT_BLOCKDIM;

    // count occurrences of each non-zero digit
    for (uint32_t i = threadIdx.x; i < len; i += stride) {
        auto val = src[(size_t)i];
        auto pck = pack(base+i, pack_mask, (val-1) << lshift);
        if (val) {
            // Use faster atomic add
            atomicAdd(&counters[(pck >> rshift) & mask], 1);
        }
    }

    __syncthreads();
}

__device__ __forceinline__
void scatter(uint2 dst[], const uint32_t src[], uint32_t base, uint32_t len,
             uint32_t lshift, uint32_t rshift, uint32_t mask,
             uint32_t pidx[] = nullptr)
{
    const uint32_t pack_mask = 0xffffffffU << lshift;
    const uint32_t stride = SORT_BLOCKDIM;

    src += base;
    #pragma unroll 1
    for (uint32_t i = threadIdx.x; i < len; i += stride) {
        auto val = src[(size_t)i];
        if (val) {
            auto pck = pack(base+i, pack_mask, (val-1) << lshift);
            uint32_t digit_idx = (pck >> rshift) & mask;
            uint32_t idx = atomicSub(&counters[digit_idx], 1) - 1;
            uint32_t pid = pidx ? pidx[base+i] : base+i;
            dst[idx] = uint2{pck, pack(pid, 0x80000000, val)};
        }
    }
}

__device__
static void upper_sort(uint2 dst[], const uint32_t src[], uint32_t len,
                       uint32_t lsbits, uint32_t bits, uint32_t digit,
                       uint32_t histogram[])
{
    uint32_t grid_div = 31 - __clz(gridDim.x);
    uint32_t grid_rem = (1<<grid_div) - 1;

    uint32_t slice = len >> grid_div;   // / gridDim.x;
    uint32_t rem   = len & grid_rem;    // % gridDim.x;
    uint32_t base;

    // Compute base with better arithmetic (avoids branch divergence)
    base = blockIdx.x < rem ? (slice + 1) * blockIdx.x : slice * blockIdx.x + rem;

    const uint32_t mask = (1<<bits) - 1;
    const uint32_t lshift = digit + bits - lsbits;

    count_digits(src, base, slice, lshift, digit, mask);

    // collect counters from SMs in the histogram
    // Use vector loads/stores for better memory throughput when possible
    #pragma unroll 1
    for (uint32_t i = threadIdx.x; i < 1<<bits; i += SORT_BLOCKDIM)
        histogram[2 + (i<<digit) + blockIdx.x] = counters[i];

    cooperative_groups::this_grid().sync();
    __syncthreads();

    const uint32_t warpid = threadIdx.x / WARP_SZ;
    const uint32_t laneid = threadIdx.x % WARP_SZ;
    const uint32_t sub_warpid = laneid >> grid_div; // / gridDim.x;
    const uint32_t sub_laneid = laneid & grid_rem;  // % gridDim.x;
    const uint32_t stride = WARP_SZ >> grid_div;    // / gridDim.x;

    uint2 h = uint2{0, 0};
    uint32_t sum, warp_off = warpid*WARP_SZ*N_SUMS + sub_warpid;

    #pragma unroll 1
    for (uint32_t i = 0; i < WARP_SZ*N_SUMS; i += stride, warp_off += stride) {
        auto* hptr = &histogram[warp_off << digit];

        // Pre-check bounds to reduce divergence
        sum = (warp_off < 1<<bits) ? hptr[2 + sub_laneid] : 0;
        sum = sum_up(sum) + h.x;

        if (sub_laneid == blockIdx.x)
            counters[warp_off] = sum;

        asm("{ .reg.b32 %v; .reg.pred %did;");
        asm("shfl.sync.up.b32 %v|%did, %0, 1, 0, 0xffffffff;" :: "r"(sum));
        asm("@%did mov.b32 %0, %v;" : "+r"(h.x));
        asm("}");
        h.y = __shfl_down_sync(0xffffffff, sum, gridDim.x-1) - h.x;

        if (blockIdx.x == 0 && sub_laneid == 0 && warp_off < 1<<bits)
            *(uint2*)hptr = h;

        h.x = __shfl_sync(0xffffffff, sum, WARP_SZ-1, WARP_SZ);
    }

    if (warpid == 0)    // offload some counters to registers
        sum = counters[laneid];

    __syncthreads();

    // carry over most significant prefix sums from each warp
    if (laneid == WARP_SZ-1)
        counters[warpid] = h.x;

    __syncthreads();

    uint32_t carry_sum = laneid ? counters[laneid-1] : 0;

    __syncthreads();

    if (warpid == 0)    // restore offloaded counters
        counters[laneid] = sum;

    __syncthreads();

    carry_sum = sum_up(carry_sum, SORT_BLOCKDIM/WARP_SZ);
    carry_sum = __shfl_sync(0xffffffff, carry_sum, warpid);

    uint32_t lane_off = warpid*WARP_SZ*N_SUMS + laneid;

    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++)
        atomicAdd(&counters[lane_off + i*WARP_SZ], carry_sum);

    __syncthreads();

    scatter(dst, src, base, slice, lshift, digit, mask);

    if (blockIdx.x == 0) {
        #pragma unroll 1
        for (uint32_t i = 0; i < N_SUMS; i++, lane_off += WARP_SZ)
            if (lane_off < 1<<bits)
                atomicAdd(&histogram[lane_off << digit], carry_sum);
    }

    cooperative_groups::this_grid().sync();
    __syncthreads();
}

__device__ __forceinline__
void count_digits(const uint2 src[], uint32_t len, uint32_t mask)
{
    zero_counters();

    // Use stride for better memory coalescing
    const uint32_t stride = SORT_BLOCKDIM;

    // count occurrences of each digit
    for (size_t i = threadIdx.x; i < len; i += stride)
        atomicAdd(&counters[src[i].x & mask], 1);

    __syncthreads();
}

__device__ __forceinline__
void scatter(uint32_t dst[], const uint2 src[], uint32_t len, uint32_t mask)
{
    const uint32_t stride = SORT_BLOCKDIM;

    #pragma unroll 1
    for (uint32_t i = threadIdx.x; i < len; i += stride) {
        auto val = src[(size_t)i];
        uint32_t idx = atomicSub(&counters[val.x & mask], 1) - 1;
        dst[idx] = val.y;
    }

    __syncthreads();
}

__device__
static void middle_sort(uint32_t dst[], const uint2 src[], uint32_t len,
                 uint32_t mask, uint32_t histogram[])
{
    uint32_t grid_div = 31 - __clz(gridDim.x);
    uint32_t grid_rem = (1<<grid_div) - 1;

    uint32_t slice = len >> grid_div;   // / gridDim.x;
    uint32_t rem   = len & grid_rem;    // % gridDim.x;
    uint32_t base;

    // Compute base with better arithmetic
    base = blockIdx.x < rem ? (slice + 1) * blockIdx.x : slice * blockIdx.x + rem;
    slice = blockIdx.x < rem ? slice + 1 : slice;

    count_digits(&src[base], slice, mask);

    // collect counters from SMs in the histogram
    #pragma unroll 1
    for (uint32_t i = threadIdx.x; i < N_COUNTERS; i += SORT_BLOCKDIM)
        histogram[2 + i*gridDim.x + blockIdx.x] = counters[i];

    cooperative_groups::this_grid().sync();
    __syncthreads();

    // Compute prefix sums so that threads within warps process numbers
    // from same digit range, as opposed to same area in |src|. This is
    // done to minimize thread divergence.

    const uint32_t warpid = threadIdx.x / WARP_SZ;
    const uint32_t laneid = threadIdx.x % WARP_SZ;

    if (warpid < N_COUNTERS/WARP_SZ) {
        uint32_t sum = 0;
        uint32_t idx = warpid*WARP_SZ + laneid;
        uint32_t hoff = 2 + idx*gridDim.x;

        // Sum across all blocks for this digit
        #pragma unroll 1
        for (uint32_t i = 0; i < gridDim.x; i++)
            sum += histogram[hoff + i];

        sum = sum_up(sum);

        if (laneid == WARP_SZ-1)
            counters[warpid] = sum;

        histogram[hoff - 2] = sum - histogram[hoff + blockIdx.x];
    }

    __syncthreads();

    if (warpid == 0 && laneid > 0) {
        uint32_t prev_sum = 0;
        for (uint32_t i = 0; i < laneid; i++)
            prev_sum += counters[i];
        uint32_t warp_off = laneid*WARP_SZ*gridDim.x;
        for (uint32_t i = 0; i < gridDim.x; i++)
            histogram[warp_off + i] += prev_sum;
    }

    cooperative_groups::this_grid().sync();
    __syncthreads();

    // compute digit boundary offsets
    uint32_t off = 0;
    #pragma unroll 1
    for (uint32_t i = threadIdx.x; i < N_COUNTERS; i += SORT_BLOCKDIM) {
        uint32_t hoff = 2 + i*gridDim.x + blockIdx.x;
        uint32_t old = histogram[hoff];
        histogram[hoff] = histogram[i*gridDim.x] + off;
        off += old;
    }

    cooperative_groups::this_grid().sync();
    __syncthreads();

    scatter(&dst[0], &src[base], slice, mask);

    cooperative_groups::this_grid().sync();
    __syncthreads();
}

__device__
static void count_histogram(const uint32_t src[], size_t len, uint32_t win,
                            uint32_t wbits, uint32_t histogram[])
{
    uint32_t hlen = 1 << wbits;
    uint32_t grid_div = 31 - __clz(gridDim.x);
    uint32_t grid_rem = (1<<grid_div) - 1;

    uint32_t slice = (len+gridDim.x-1) >> grid_div;   // (len+gridDim.x-1) / gridDim.x;
    uint32_t base  = slice * blockIdx.x;
    uint32_t sz    = blockIdx.x<gridDim.x-1 ? slice : len-base;

    // Count occurrences of each digit value using atomic operations
    for (uint32_t i = threadIdx.x; i < sz; i += blockDim.x) {
        uint32_t val = src[base + i];
        if (val)
            atomicAdd(&histogram[win*hlen + (val & 0x7fffffff) % hlen], 1);
    }

    cooperative_groups::this_grid().sync();
}

__device__
static void radix_sort(uint32_t result[], const uint32_t input[], size_t len,
                       uint32_t win, vec2d_t<uint2> temps,
                       vec2d_t<uint32_t> histograms, uint32_t wbits,
                       uint32_t lsbits0, uint32_t lsbits1)
{
    uint32_t* histogram = &histograms[win][0];

    uint32_t bits0 = lsbits0 < DIGIT_BITS ? lsbits0 : DIGIT_BITS;
    uint32_t bits1 = lsbits1 < DIGIT_BITS ? lsbits1 : DIGIT_BITS;
    uint32_t mask0 = (1 << bits0) - 1;
    uint32_t mask1 = (1 << bits1) - 1;

    const int bits0_shift = lsbits0 - bits0;
    const int bits1_shift = lsbits1 - bits1;

    upper_sort(temps[0], input, len, lsbits0, bits0, bits0_shift, histogram);
    middle_sort(result, temps[0], len, mask1, histogram);
}

__launch_bounds__(SORT_BLOCKDIM, 2) __global__
void sort(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
          vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
          uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1)
{
    const int grid_div = 31 - __clz(gridDim.x);
    const uint32_t total_bits = lsbits0 + lsbits1;

    // Count histogram first
    count_histogram(inouts[win], len, win, wbits, &histograms[0][0]);

    // Then perform radix sort
    radix_sort(inouts[win], inouts[win], len, win, temps, histograms,
               wbits, lsbits0, lsbits1);
}

#endif // __MSM_SORT_DONT_IMPLEMENT__
#endif // __SPPARK_MSM_SORT_CUH__
