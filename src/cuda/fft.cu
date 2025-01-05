#include <iostream>
#include <cmath>
#include <cuComplex.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__device__
cuFloatComplex exp(cuFloatComplex z) {
    float expReal = exp(cuCrealf(z));
    float cosImag = cos(cuCimagf(z));
    float sinImag = sin(cuCimagf(z));

    return make_cuFloatComplex(expReal * cosImag, expReal * sinImag);
}

__device__
unsigned int reverseBits(unsigned int n, unsigned int numBits)
{
    unsigned int reversed{ 0 };
    for (unsigned int i = 0; i < numBits; ++i)
    {
        unsigned int lsb { n & 1 };
        reversed = (reversed << 1) | lsb;
        n >>= 1;
    }
    return reversed;
}

__global__
void reverse_bits(const cuFloatComplex *fftSequence, cuFloatComplex *fftSequenceBitReversed, const int log2N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  fftSequenceBitReversed[reverseBits(tid, log2N)] = make_cuFloatComplex(cuCrealf(fftSequence[tid]), cuCimagf(fftSequence[tid]));
}

__global__
void fft(cuFloatComplex *fftSequenceBitReversed, const int N, const int s)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  uint k { tid/(s/2) * s };
  uint j { tid%(s/2) };

  auto W{ exp(make_cuFloatComplex(0, -2 * M_PI_F * j / s)) };

  cuFloatComplex c1 { fftSequenceBitReversed[k+j] };
  cuFloatComplex c2 { fftSequenceBitReversed[k+j+s/2] };
  cuFloatComplex c3 { cuCmulf(W, c2) };

  fftSequenceBitReversed[k+j] = cuCaddf(c1, c3);
  fftSequenceBitReversed[k+j+ss/2] = cuCsubf(c1, c3);
}

void printCuComplex(cuFloatComplex c)
{
  std::cout << "(" << cuCrealf(c) << "," << cuCimagf(c) << ") ";
}

int main(void)
{
  int N = 8;
  cuFloatComplex *fftSequence, *fftSequenceBitReversed;

  checkCuda( cudaMallocManaged(&fftSequence, N*sizeof(cuFloatComplex)) );
  checkCuda( cudaMallocManaged(&fftSequenceBitReversed, N*sizeof(cuFloatComplex)) );

  for (size_t i = 1; i <= N; i++)
  {
    fftSequence[i-1] = make_cuFloatComplex(i, 0);
  }

  fftSequenceBitReversed[2] = make_cuFloatComplex(1,1);
  reverse_bits<<<1, 8>>>(fftSequence, fftSequenceBitReversed, log2(N));

  for (int s = 2; s <= N; s *= 2)
  {
    fft<<<1, 8>>>(fftSequenceBitReversed, N, s);
  }

  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++)
  {
    printCuComplex(fftSequenceBitReversed[i]);
  }
  std::cout << std::endl;

  cudaFree(fftSequence);
  cudaFree(fftSequenceBitReversed);
  
  return 0;
}