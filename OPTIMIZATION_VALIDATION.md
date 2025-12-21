# NTT GPU Optimization Validation Plan

This document outlines the steps needed to ensure the NVIDIA GPU optimizations work correctly.

## ✅ Pre-Deployment Checklist

### 1. Code Compilation & Syntax Validation

#### A. Compile for Multiple Architectures
```bash
# Test compilation for all supported architectures
cd poc/ntt-cuda

# Pascal (SM 6.0) - GTX 1080
cargo build --release --features bls12_381
cargo clean

# Volta (SM 7.0) - V100
CUDA_ARCH=70 cargo build --release --features bls12_381
cargo clean

# Turing (SM 7.5) - RTX 2080
CUDA_ARCH=75 cargo build --release --features bls12_381
cargo clean

# Ampere (SM 8.0) - A100
CUDA_ARCH=80 cargo build --release --features bls12_381
cargo clean

# Hopper (SM 9.0) - H100
CUDA_ARCH=90 cargo build --release --features bls12_381
cargo clean

# Blackwell (SM 10.0) - B100/B200
CUDA_ARCH=100 cargo build --release --features bls12_381
cargo clean
```

**Expected Result**: All compilations should succeed without errors or warnings.

#### B. Multi-Target Compilation
```bash
# Build for multiple architectures simultaneously
CUDA_ARCH="70,75,80,86,89,90,100" cargo build --release --features bls12_381
```

**Expected Result**: Single binary works across all GPU generations.

#### C. Check for Compiler Warnings
```bash
# Enable all warnings
RUSTFLAGS="-D warnings" cargo build --release --features bls12_381
```

**Expected Result**: Zero warnings.

---

### 2. Functional Correctness Testing

#### A. Run Existing Test Suite
```bash
cd poc/ntt-cuda

# Test with different field types
cargo test --release --features bls12_381
cargo test --release --features bls12_377
cargo test --release --features pallas
cargo test --release --features vesta
cargo test --release --features bn254
cargo test --release --features gl64  # Goldilocks (32-bit field)
cargo test --release --features bb31  # Baby Bear (31-bit field)
```

**Expected Result**: All tests pass with 100% success rate.

#### B. Test with Debug Mode Enabled
```bash
# Compile with debug output to verify correctness
CUDA_FLAGS="-DDEBUG_NTT" cargo test --release --features bls12_381
```

**Expected Result**:
- Tests pass
- Debug output shows expected kernel configurations
- No CUDA errors in output

#### C. Correctness Validation
Create a test that compares GPU results against CPU reference:
```bash
cd poc/ntt-cuda/tests
# Run the ntt.rs test which validates GPU against CPU
cargo test --release --features bls12_381 -- --nocapture
```

**Expected Result**: GPU output matches CPU reference implementation exactly.

---

### 3. Performance Benchmarking

#### A. Baseline Performance (Before Optimization)
```bash
# Checkout original code
git stash
cd poc/ntt-cuda

# Run benchmarks
cargo bench --features bls12_381

# Save results
cargo bench --features bls12_381 > baseline_results.txt
```

#### B. Optimized Performance (After Optimization)
```bash
# Return to optimized code
git stash pop
cd poc/ntt-cuda

# Run same benchmarks
cargo bench --features bls12_381 > optimized_results.txt

# Compare results
diff baseline_results.txt optimized_results.txt
```

**Expected Results by Architecture**:
- **Blackwell (B100/B200)**: 5-8x improvement
- **Hopper (H100)**: 3-5x improvement
- **Ampere (A100)**: 2-4x improvement
- **Volta (V100)**: 1.5-2.5x improvement

#### C. Profile with NVIDIA Nsight Compute
```bash
# Profile the NTT kernel
ncu --target-processes all \
    --set full \
    --export ntt_profile \
    cargo test --release --features bls12_381

# Generate report
ncu -i ntt_profile.ncu-rep --page details
```

**Key Metrics to Verify**:
1. **Achieved Occupancy**: Should be 50-80%
2. **Memory Throughput**: Should be >80% of peak bandwidth
3. **Warp Execution Efficiency**: Should be >90%
4. **Register Usage**: Should not cause spilling
5. **Shared Memory Usage**: Should be within limits
6. **L2 Cache Hit Rate**: Should improve with optimizations

---

### 4. Architecture-Specific Validation

#### A. Verify Launch Configurations
Add temporary logging to verify launch bounds:
```cpp
#ifdef DEBUG_NTT
printf("GPU Arch: %d, Block Size: %d, Blocks per SM: %d\n",
       __CUDA_ARCH__, blockDim.x, /* blocks per SM */);
#endif
```

**Expected Values**:

| Architecture | _CT_NTT Block Size | LDE Block Size |
|--------------|-------------------|----------------|
| Blackwell    | 1024              | 256            |
| Hopper       | 512               | 128            |
| Ampere       | 512               | 128            |
| Volta        | 256               | 64             |
| Pascal       | 256               | 32             |

#### B. Verify Memory Optimizations
Use Nsight Compute to check:
- Cache line alignment (should be 256B for Blackwell, 128B for others)
- Coalesced memory access percentage (should be >95%)
- Global memory load efficiency (should be >80%)

---

### 5. Stress Testing

#### A. Large Domain Sizes
```bash
# Test with maximum domain sizes
cargo test --release --features bls12_381 -- --nocapture large_domain
```

Test domain sizes: 2^10, 2^15, 2^20, 2^24, 2^28

**Expected Result**: No memory errors, reasonable performance scaling.

#### B. Edge Cases
Test special cases:
- Very small domains (2^4, 2^6)
- Odd-sized iterations
- INTT (inverse NTT)
- Coset NTT vs standard NTT

#### C. Memory Safety
```bash
# Run with CUDA memory checker
cuda-memcheck cargo test --release --features bls12_381

# Or use compute-sanitizer on newer CUDA versions
compute-sanitizer --tool memcheck cargo test --release --features bls12_381
```

**Expected Result**: Zero memory errors or leaks.

---

### 6. Cross-GPU Validation

If you have access to multiple GPU types:

```bash
# On each GPU, run:
nvidia-smi  # Verify GPU detected
cargo test --release --features bls12_381
cargo bench --features bls12_381
```

Test on:
- [x] V100 (Volta)
- [x] RTX 2080 Ti (Turing)
- [x] A100 (Ampere)
- [x] RTX 4090 (Ada Lovelace - SM 8.9)
- [x] H100 (Hopper)
- [ ] B100/B200 (Blackwell) - when available

---

### 7. Regression Testing

#### A. Verify No Breaking Changes
```bash
# Run full test suite multiple times
for i in {1..10}; do
    echo "Test run $i"
    cargo test --release --features bls12_381 || exit 1
done
```

**Expected Result**: 100% pass rate across all runs.

#### B. Compare Output Bit-for-Bit
```bash
# Generate reference output with old code
git stash
cargo test --release --features bls12_381 > old_output.txt

# Generate output with new code
git stash pop
cargo test --release --features bls12_381 > new_output.txt

# Compare (should be identical)
diff old_output.txt new_output.txt
```

---

### 8. Production Readiness

#### A. Remove Debug Code
Ensure production builds DON'T define `DEBUG_NTT`:
```bash
# This should have NO debug output
cargo build --release --features bls12_381
```

#### B. Optimization Flags
Verify optimal compiler flags:
```bash
# Should be in Cargo.toml or build.rs
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

#### C. Documentation
- [x] Document optimization changes
- [x] Update README with new performance numbers
- [ ] Add comments explaining architecture-specific code
- [ ] Document DEBUG_NTT flag usage

---

### 9. Known Issues & Limitations

#### Potential Issues to Watch For:

1. **Compilation Time**: Multi-arch builds may take longer
2. **Binary Size**: Multi-arch binaries will be larger
3. **Blackwell Testing**: May not be testable until hardware is available
4. **Older GPUs**: Ensure Pascal and Maxwell (if supported) still work

#### Fallback Plan:
If issues arise with specific architectures:
```cpp
#if __CUDA_ARCH__ >= 1000
  // Blackwell path
#elif __CUDA_ARCH__ >= 900
  // Hopper path
#else
  // Fallback to original code
  #include "original_kernel.cu"
#endif
```

---

### 10. Deployment Checklist

Before merging to main:
- [ ] All compilation tests pass
- [ ] All functional tests pass (100% success)
- [ ] Performance improvements verified
- [ ] No regressions on any architecture
- [ ] Memory checker shows zero errors
- [ ] Nsight Compute profiles look good
- [ ] Code reviewed by team
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## Quick Validation Script

```bash
#!/bin/bash
# quick_validate.sh - Run essential validation checks

set -e  # Exit on error

echo "=== 1. Compilation Test ==="
cargo clean
cargo build --release --features bls12_381

echo "=== 2. Functional Tests ==="
cargo test --release --features bls12_381

echo "=== 3. Memory Check ==="
compute-sanitizer --tool memcheck \
    cargo test --release --features bls12_381 2>&1 | grep "ERROR SUMMARY"

echo "=== 4. Quick Benchmark ==="
cargo bench --features bls12_381 | tail -20

echo ""
echo "✅ All validation checks passed!"
```

---

## Performance Profiling Commands

### Detailed Kernel Analysis
```bash
# Profile specific kernel
ncu --kernel-name _CT_NTT \
    --launch-skip 0 \
    --launch-count 1 \
    --set full \
    cargo test --release --features bls12_381

# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    cargo test --release --features bls12_381

# Check memory efficiency
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    cargo test --release --features bls12_381
```

### Compare Before/After
```bash
# Profile old version
git stash
ncu --export baseline cargo test --release --features bls12_381

# Profile new version
git stash pop
ncu --export optimized cargo test --release --features bls12_381

# Compare
ncu --import baseline.ncu-rep optimized.ncu-rep --page comparison
```

---

## Expected Nsight Compute Results

### Target Metrics (Optimized Code):

| Metric | Target | Critical If Below |
|--------|--------|-------------------|
| Achieved Occupancy | 50-80% | 30% |
| Memory Throughput | >80% | 50% |
| Compute Throughput | >70% | 40% |
| Warp Execution Efficiency | >90% | 70% |
| L2 Cache Hit Rate | >60% | 30% |
| Global Load Efficiency | >80% | 50% |
| Shared Memory Efficiency | >90% | 70% |

### Red Flags to Watch For:
- Register spilling (check registers/thread)
- Shared memory bank conflicts
- Uncoalesced global memory accesses
- Low occupancy (<30%)
- High tail effect (warp divergence)

---

## Troubleshooting Guide

### Issue: Compilation Fails
**Solution**: Check CUDA version compatibility
```bash
nvcc --version  # Should be CUDA 11.0+
```

### Issue: Tests Fail
**Solution**: Enable debug mode to see where
```bash
CUDA_FLAGS="-DDEBUG_NTT" cargo test --features bls12_381 -- --nocapture
```

### Issue: Performance Worse Than Baseline
**Possible Causes**:
1. Debug output still enabled (remove `-DDEBUG_NTT`)
2. Wrong architecture compiled (check with `ncu`)
3. Thermal throttling (monitor with `nvidia-smi`)
4. Different GPU than expected

### Issue: Memory Errors
**Solution**: Check alignment and bounds
```bash
compute-sanitizer --tool memcheck \
    --leak-check full \
    cargo test --release --features bls12_381
```

---

## Contact & Support

If validation fails or you encounter issues:
1. Check git diff to see what changed
2. Review Nsight Compute profiles
3. Test on different GPU if available
4. Revert to baseline if critical issue found

---

## Summary

**Minimum Required Tests Before Deployment**:
1. ✅ Compiles without errors
2. ✅ All unit tests pass
3. ✅ No memory errors (cuda-memcheck)
4. ✅ Performance improves vs baseline
5. ✅ No regressions on any architecture

**Recommended Additional Tests**:
6. Profile with Nsight Compute
7. Test on multiple GPUs
8. Stress test with large domains
9. Validate bit-for-bit correctness

**Critical Success Criteria**:
- Zero functional regressions
- Measurable performance improvement (>1.5x)
- Clean memory checker output
- All architectures supported
