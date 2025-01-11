#include <iostream>
#include <cmath>
#include <cuComplex.h>
#include <chrono>

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
void fft(cuFloatComplex *fftSequenceBitReversed, const int s)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  int k { tid/(s/2) * s };
  int j { tid%(s/2) };

  auto W{ exp(make_cuFloatComplex(0, -2 * M_PI * j / s)) };

  cuFloatComplex c1 { fftSequenceBitReversed[k+j] };
  cuFloatComplex c2 { fftSequenceBitReversed[k+j+s/2] };
  cuFloatComplex c3 { cuCmulf(W, c2) };

  fftSequenceBitReversed[k+j] = cuCaddf(c1, c3);
  fftSequenceBitReversed[k+j+s/2] = cuCsubf(c1, c3);
}

void printCuComplex(cuFloatComplex c)
{
  std::cout << "(" << cuCrealf(c) << "," << cuCimagf(c) << ") ";
}

int main(int argc, const char * argv[])
{
  int N = 8;

  if (argc == 2)
  {
    N = std::stoi(argv[1]);
    if (N > 100'000)
      {
          std::cerr << "Safety limit 100 000 threads.\n";
          return 1;
      }
  }
  else
  {
      std::cerr << "Usage: " << argv[0] << " <natural_number>\n";
      return 1;
  }

  auto start{ std::chrono::high_resolution_clock::now() };

  for (size_t i = 0; i < 100'00; i++)
  {
    cuFloatComplex *fftSequence, *fftSequenceBitReversed;

    checkCuda( cudaMallocManaged(&fftSequence, N*sizeof(cuFloatComplex)) );
    checkCuda( cudaMallocManaged(&fftSequenceBitReversed, N*sizeof(cuFloatComplex)) );

    for (size_t i = 1; i <= N; i++)
    {
      fftSequence[i-1] = make_cuFloatComplex(i, 0);
    }

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Maximum work group size (threads per block): " << prop.maxThreadsPerBlock << std::endl;

    /// Important to set the work group to maximum size for maximum compute unit unitilization
    int blockSize = prop.maxThreadsPerBlock > N ? N : prop.maxThreadsPerBlock;
    int numBlocks = (N + blockSize - 1) / blockSize;

    reverse_bits<<<numBlocks, blockSize>>>(fftSequence, fftSequenceBitReversed, log2(N));

    blockSize = 256 > N ? N : 256;
    numBlocks = (N/2 + blockSize - 1) / blockSize;

    for (int s = 2; s <= N; s *= 2)
    {
      fft<<<numBlocks, blockSize>>>(fftSequenceBitReversed, s);
    }

    cudaDeviceSynchronize();
  }

  auto end{ std::chrono::high_resolution_clock::now() };

  std::chrono::duration<double> elapsed{ end - start };
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

  cudaFree(fftSequence);
  cudaFree(fftSequenceBitReversed);
  
  return 0;
}