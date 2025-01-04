# Fast Fourier Transform GPU Implementation

This repository contains the source code for a GPU-accelerated implementation of the Fast Fourier Transform (FFT) algorithm. The goal of this project is to evaluate the performance of FFT on different GPU hardware using three major frameworks: **Metal** (for Apple hardware), **CUDA** (for NVIDIA hardware), and **OpenCL** (for AMD hardware).

By comparing the speedup achieved by the algorithm on various hardware, we aim to determine the optimal platform for systems where FFT computations dominate the computational workload.



## Features


**Multi-platform support**:
  - **Metal**: Optimized for macOS and iOS devices with Apple GPUs.
  - **CUDA**: Supports NVIDIA GPUs, leveraging their high-performance CUDA architecture.
  - **OpenCL**: A platform-independent implementation for AMD GPUs and other OpenCL-compatible devices.

**Performance benchmarking**:
  - Compare FFT execution times across different GPU frameworks and hardware.
  - Evaluate scalability with varying input sizes.



## Requirements

### General
- C++14 or later
- A GPU compatible with the targeted frameworks
- Benchmarking tools for execution time measurement

### Framework-specific
- **Metal**: macOS 10.13+ with Xcode and a Metal-compatible GPU.
- **CUDA**: NVIDIA GPU with CUDA Toolkit installed.
- **OpenCL**: OpenCL drivers installed for your platform.


## Repository Structure

```plaintext
├── src/
│   ├── metal/
│   │   └── FastFourierTransformMetal.xcodeproj # Project file for Metal & CPU implementation.
│   ├── cuda/
│   │   └── fft_cuda.cu    # CUDA implementation
│   ├── opencl/
│       └── fft_opencl.cl  # OpenCL implementation
├── benchmarks/
│   ├── run_benchmarks.py  # Script for running benchmarks
├── README.md              # Project documentation
└── LICENSE                # License information
```

## How to build

- Metal implementation:
```bash
xcodebuild -project FastFourierTransformMetal.xcodeproj \
           -scheme FastFourierTransformMetal \
           -configuration Release \
           -derivedDataPath build \ # Override the build directory
           build
```
- CUDA implementation:
```bash
CUDA test
```
- OpenCL implementation:
```bash
OpenCL test
```

## How to run

- Metal implementation:
```bash
cd build/Build/Products/Release/
./FastFourierTransformMetal -cpu 1024 # Executes FFT with 1024 elements on CPU
./FastFourierTransformMetal -gpu 1024 # Executes FFT with 1024 elements on GPU
```
- CUDA implementation:
```bash
CUDA test
```
- OpenCL implementation:
```bash
OpenCL test
```
<br>

> ⚠️ **Note:**  
> Currently only powers of 2 are supported for the number of the elements.

## Benchmarking 

To benchmark the FFT implementation on different hardware, use the provided run_benchmarks.py script in the benchmarks/ directory. This script collects execution times and generates comparison charts.
```bash
cd benchmarks
python run_benchmarks.py
```

_Note: Ensure all implementations are correctly built and the required hardware and frameworks are accessible._