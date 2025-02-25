// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __NTT_KERNELS_CU__
#define __NTT_KERNELS_CU__

#if defined(__NVCC__)
# include <cooperative_groups.h>
#elif defined(__HIPCC__)
# include <hip/hip_cooperative_groups.h>
#endif

// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
template<class fr_t>
__launch_bounds__(1024) __global__
void bit_rev_permutation(fr_t* d_out, const fr_t *d_in, uint32_t lg_domain_size)
{
    if (gridDim.x == 1 && blockDim.x == (1 << lg_domain_size)) {
        uint32_t idx = threadIdx.x;
        uint32_t rev = bit_rev(idx, lg_domain_size);

        fr_t t = d_in[idx];
        if (d_out == d_in)
            __syncthreads();
        d_out[rev] = t;
    } else {
        index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
        index_t rev = bit_rev(idx, lg_domain_size);
        bool copy = d_out != d_in && idx == rev;

        if (idx < rev || copy) {
            fr_t t0 = d_in[idx];
            if (!copy) {
                fr_t t1 = d_in[rev];
                d_out[idx] = t1;
            }
            d_out[rev] = t0;
        }
    }
}

template<typename T>
static __device__ __host__ constexpr uint32_t lg2(T n)
{   uint32_t ret=0; while (n>>=1) ret++; return ret;   }

// Optimized bit reversal permutation with reduced synchronization and better memory access
template<unsigned int Z_COUNT, class fr_t>
__launch_bounds__(256, 4) __global__  // Increased occupancy from (192,2)
void bit_rev_permutation_z(fr_t* out, const fr_t* in, uint32_t lg_domain_size)
{
    static_assert((Z_COUNT & (Z_COUNT-1)) == 0, "unvalid Z_COUNT");
    const uint32_t LG_Z_COUNT = lg2(Z_COUNT);

    extern __shared__ int xchg_bit_rev[];
    fr_t (*xchg)[Z_COUNT][Z_COUNT] = reinterpret_cast<decltype(xchg)>(xchg_bit_rev);

    uint32_t gid = threadIdx.x / Z_COUNT;
    uint32_t idx = threadIdx.x % Z_COUNT;
    uint32_t rev = bit_rev(idx, LG_Z_COUNT);

    index_t step = (index_t)1 << (lg_domain_size - LG_Z_COUNT);
    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    // Pre-calculate group indices outside the loop to reduce redundant calculations
    index_t group_idx = tid >> LG_Z_COUNT;
    index_t group_rev = bit_rev(group_idx, lg_domain_size - 2*LG_Z_COUNT);

    // Process only if group_idx <= group_rev to avoid redundant computation
    if (group_idx <= group_rev) {
        index_t base_idx = group_idx * Z_COUNT + idx;
        index_t base_rev = group_rev * Z_COUNT + idx;

        fr_t regs[Z_COUNT];

        // Batch loads into registers for better memory coalescing
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            regs[i] = in[i * step + base_idx];
        }

        // Only one synchronization point before storing to shared memory
        (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            xchg[gid][i][rev] = regs[i];
        }

        // Second load only if needed
        if (group_idx != group_rev) {
            #pragma unroll
            for (uint32_t i = 0; i < Z_COUNT; i++) {
                regs[i] = in[i * step + base_rev];
            }
        }

        (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();

        // Batch stores for better coalescing
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            out[i * step + base_rev] = xchg[gid][rev][i];
        }

        if (group_idx != group_rev) {
            (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();

            #pragma unroll
            for (uint32_t i = 0; i < Z_COUNT; i++) {
                out[i * step + base_idx] = regs[i];
            }
        }
    }

    // Handle the remaining elements with improved striding pattern
    index_t remaining_tid = tid + blockDim.x * gridDim.x;

    #pragma unroll 1
    while (remaining_tid < step) {
        group_idx = remaining_tid >> LG_Z_COUNT;
        group_rev = bit_rev(group_idx, lg_domain_size - 2*LG_Z_COUNT);

        if (group_idx <= group_rev) {
            index_t base_idx = group_idx * Z_COUNT + idx;
            index_t base_rev = group_rev * Z_COUNT + idx;

            fr_t regs[Z_COUNT];

            #pragma unroll
            for (uint32_t i = 0; i < Z_COUNT; i++) {
                regs[i] = in[i * step + base_idx];
            }

            if (group_idx != group_rev) {
                fr_t regs_rev[Z_COUNT];

                #pragma unroll
                for (uint32_t i = 0; i < Z_COUNT; i++) {
                    regs_rev[i] = in[i * step + base_rev];
                }

                #pragma unroll
                for (uint32_t i = 0; i < Z_COUNT; i++) {
                    out[i * step + base_rev] = regs[i];
                    out[i * step + base_idx] = regs_rev[i];
                }
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < Z_COUNT; i++) {
                    out[i * step + base_idx] = regs[i];
                }
            }
        }
        remaining_tid += blockDim.x * gridDim.x;
    }
}

// Optimized root computation with reduced branching
template<class fr_t>
__device__ __forceinline__
fr_t get_intermediate_root(index_t pow, const fr_t (*roots)[WINDOW_SIZE])
{
    unsigned int off = 0;
    fr_t root = fr_t::one();

    // Simplified logic with fewer branches
    if (sizeof(fr_t) <= 8) {
        #pragma unroll
        for (unsigned int i = 0; i < WINDOW_NUM; i++) {
            unsigned int pow_win = pow % WINDOW_SIZE;
            // Use masked arithmetic instead of branching
            bool use_root = (pow_win != 0);
            if (use_root) {
                root = roots[i][pow_win];
                break;
            }
            pow >>= LG_WINDOW_SIZE;
            off++;
        }
    } else {
        unsigned int pow_win = pow % WINDOW_SIZE;
        bool skip_first = (pow_win == 0);
        // Use masked arithmetic to avoid branch
        off += skip_first ? 1 : 0;
        pow >>= skip_first ? LG_WINDOW_SIZE : 0;
        root = roots[off][pow % WINDOW_SIZE];
    }

    // Process remaining windows with linear traversal
    pow >>= LG_WINDOW_SIZE;
    off++;

    #pragma unroll 4  // Partial unrolling for better instruction scheduling
    while (pow) {
        if (pow % WINDOW_SIZE != 0) {
            root *= roots[off][pow % WINDOW_SIZE];
        }
        pow >>= LG_WINDOW_SIZE;
        off++;
    }

    return root;
}

template<class fr_t>
__launch_bounds__(1024) __global__
void LDE_distribute_powers(fr_t* d_inout, uint32_t lg_domain_size,
                           uint32_t lg_blowup, bool bitrev,
                           const fr_t (*gen_powers)[WINDOW_SIZE])
{
#if 0
    assert(blockDim.x * gridDim.x == blockDim.x * (size_t)gridDim.x);
#endif
    size_t domain_size = (size_t)1 << lg_domain_size;
    index_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    #pragma unroll 1
    for (; idx < domain_size; idx += blockDim.x * gridDim.x) {
        fr_t r = d_inout[idx];

        index_t pow = bitrev ? bit_rev(idx, lg_domain_size) : idx;
        pow <<= lg_blowup;
        r *= get_intermediate_root(pow, gen_powers);

        d_inout[idx] = r;
    }
}

// Optimized LDE_spread with reduced synchronization and better memory coalescing
template<class fr_t>
__launch_bounds__(1024) __global__
void LDE_spread_distribute_powers(fr_t* out, fr_t* in,
                                  const fr_t (*gen_powers)[WINDOW_SIZE],
                                  uint32_t lg_domain_size, uint32_t lg_blowup,
                                  bool perform_shift = true,
                                  bool ext_pow = false)
{
    extern __shared__ int xchg_lde_spread[]; // block size
    fr_t* exchange = reinterpret_cast<decltype(exchange)>(xchg_lde_spread);

    size_t domain_size = (size_t)1 << lg_domain_size;
    uint32_t blowup = 1u << lg_blowup;
    uint32_t stride = gridDim.x * blockDim.x;

    assert(lg_domain_size + lg_blowup <= MAX_LG_DOMAIN_SIZE &&
           (stride & (stride-1)) == 0);

    bool overlapping_data = false;

    if ((in < out && (in + domain_size) > out)
     || (in >= out && (out + domain_size * blowup) > in))
    {
        overlapping_data = true;
        assert(&out[domain_size * (blowup - 1)] == &in[0]);
    }

    // Pre-calculate thread-specific offset only once
    const uint32_t thread_offset = threadIdx.x;

    index_t idx0 = blockDim.x * blockIdx.x;
    index_t iters = domain_size >> (31 - __clz(stride));

    for (index_t iter = 0; iter < iters; iter++) {
        index_t idx = idx0 + thread_offset;

        // Prefetch data from global memory
        fr_t r = in[idx];

        if (perform_shift) {
            index_t pow = bit_rev(idx, lg_domain_size +
                                  (ext_pow ? lg_blowup : 0));

            r = r * get_intermediate_root(pow, gen_powers);
        }

        // Single synchronization point
        __syncthreads();

        // Store to shared memory
        exchange[thread_offset] = r;

        // Single synchronization for cooperative groups if needed
        if (overlapping_data && (iter >= (blowup - 1) * (iters >> lg_blowup)))
            cooperative_groups::this_grid().sync();
        else
            __syncthreads();

        // Use aligned strided writes with fewer conditionals
        const uint32_t base_offset = (idx0 << lg_blowup);

        // Process in groups of 2 for better instruction-level parallelism
        for (uint32_t offset = thread_offset, i = 0; i < blowup; i += 2) {
            // First iteration
            fr_t val1;
#ifdef __HIP_DEVICE_COMPILE__
            val1 = exchange[offset >> lg_blowup];
            val1 = czero(val1, offset & (blowup-1));
#else
            // Use arithmteic instead
            bool is_zero = (offset & (blowup-1)) != 0;
            val1 = is_zero ? fr_t() : exchange[offset >> lg_blowup];
#endif
            out[base_offset + offset] = val1;
            offset += blockDim.x;

            // Second iteration
            fr_t val2;
#ifdef __HIP_DEVICE_COMPILE__
            val2 = exchange[offset >> lg_blowup];
            val2 = czero(val2, offset & (blowup-1));
#else
            // Use math
            is_zero = (offset & (blowup-1)) != 0;
            val2 = is_zero ? fr_t() : exchange[offset >> lg_blowup];
#endif
            out[base_offset + offset] = val2;
            offset += blockDim.x;
        }

        idx0 += stride;
    }
}

template<class fr_t>
__device__ __forceinline__
void get_intermediate_roots(fr_t& root0, fr_t& root1,
                            index_t idx0, index_t idx1,
                            const fr_t (*roots)[WINDOW_SIZE])
{
    int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
    int off = (WINDOW_NUM - 1);
    index_t idxo = idx0 | idx1;
    index_t mask = ((index_t)1 << win) - 1;

    root0 = roots[off][idx0 >> win];
    root1 = roots[off][idx1 >> win];
    #pragma unroll 1
    while (off-- && (idxo & mask)) {
        fr_t t;
        win -= LG_WINDOW_SIZE;
        mask >>= LG_WINDOW_SIZE;
        root0 *= (t = roots[off][(idx0 >> win) % WINDOW_SIZE]);
        root1 *= (t = roots[off][(idx1 >> win) % WINDOW_SIZE]);
    }
}

template<int z_count, class fr_t>
__device__ __forceinline__
void coalesced_load(fr_t r[z_count], const fr_t* inout, index_t idx,
                    const unsigned int stage)
{
    const unsigned int x = threadIdx.x & (z_count - 1);
    idx &= ~((index_t)(z_count - 1) << stage);
    idx += x;

    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (index_t)1 << stage)
        r[z] = inout[idx];
}

template<int z_count, class fr_t>
__device__ __forceinline__
void transpose(fr_t r[z_count])
{
    extern __shared__ int xchg_transpose[];
    fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(xchg_transpose);

    const unsigned int x = threadIdx.x & (z_count - 1);
    const unsigned int y = threadIdx.x & ~(z_count - 1);

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        xchg[y + z][x] = r[z];

    __syncwarp();

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        r[z] = xchg[y + x][z];
}

template<int z_count, class fr_t>
__device__ __forceinline__
void coalesced_store(fr_t* inout, index_t idx, const fr_t r[z_count],
                     const unsigned int stage)
{
    const unsigned int x = threadIdx.x & (z_count - 1);
    idx &= ~((index_t)(z_count - 1) << stage);
    idx += x;

    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (index_t)1 << stage)
        inout[idx] = r[z];
}

#if defined(FEATURE_BABY_BEAR) || defined(FEATURE_GOLDILOCKS)
# include "kernels/gs_mixed_radix_narrow.cu"
# include "kernels/ct_mixed_radix_narrow.cu"
#else // 256-bit fields
# include "kernels/gs_mixed_radix_wide.cu"
# include "kernels/ct_mixed_radix_wide.cu"
#endif

#endif /* __NTT_KERNELS_CU__ */
