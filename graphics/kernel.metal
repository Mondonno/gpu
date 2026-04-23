#include <metal_stdlib>

kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* res [[buffer(2)]],
                    uint index[[thread_position_in_grid]])
{
    res[index] = a[index] + b[index];
}

kernel void vec_multiply(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* res [[buffer(2)]], uint index[[thread_position_in_grid]]) {
    res[index] = a[index] * b[index];
}
