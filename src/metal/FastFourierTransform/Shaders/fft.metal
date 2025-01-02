#include <metal_stdlib>
using namespace metal;

kernel void fft(device const float* evenIndexed,
                device const float* oddIndexed,
                device float* result,
                uint index [[thread_position_in_grid]])
{
    result[index] = evenIndexed[index] + oddIndexed[index];
}
