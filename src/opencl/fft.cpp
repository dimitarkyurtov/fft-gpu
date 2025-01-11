#include <CL/cl.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error during " << operation << ": " << err << std::endl;
        exit(1);
    }
}

const char* reverseBitsKernel = R"(
__kernel void reverse_bits(__global float2* input, __global float2* output, int log2N) {
    int tid = get_global_id(0);
    int reversed = 0;
    int numBits = log2N;

    for (int i = 0; i < numBits; ++i) {
        reversed = (reversed << 1) | (tid & 1);
        tid >>= 1;
    }

    output[reversed] = input[get_global_id(0)];
}
)";

const char* fftKernel = R"(
__kernel void fft(__global float2* data, int s) {
    int tid = get_global_id(0);
    int k = tid / (s / 2) * s;
    int j = tid % (s / 2);

    float angle = -2.0f * M_PI_F * j / s;
    float2 W = (float2)(cos(angle), sin(angle));

    float2 c1 = data[k + j];
    float2 c2 = data[k + j + s / 2];
    float2 c3 = (float2)(W.x * c2.x - W.y * c2.y, W.x * c2.y + W.y * c2.x);

    data[k + j] = (float2)(c1.x + c3.x, c1.y + c3.y);
    data[k + j + s / 2] = (float2)(c1.x - c3.x, c1.y - c3.y);
}
)";

void printComplex(const std::complex<float>& c) {
    std::cout << "(" << c.real() << ", " << c.imag() << ") ";
}

int main(int argc, const char* argv[]) {
    int N = 8;

    if (argc == 2) {
        N = std::stoi(argv[1]);
        if (N > 100'000) {
            std::cerr << "Safety limit 100,000 threads.\n";
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " <natural_number>\n";
        return 1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL GPU devices found." << std::endl;
        return 1;
    }

    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    /// Important to set the work group to maximum size for maximum compute unit unitilization
    size_t maxWorkGroupSize;
    checkError(device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize), "getInfo");

    const size_t blockSizeReverseBits = maxWorkGroupSize > N ? N : maxWorkGroupSize;
    const size_t blockSizeFFT = maxWorkGroupSize > N/2 ? N/2 : maxWorkGroupSize;

    cl::Program::Sources sources;
    sources.push_back({reverseBitsKernel, strlen(reverseBitsKernel)});
    sources.push_back({fftKernel, strlen(fftKernel)});
    cl::Program program(context, sources);
    checkError(program.build({device}), "Program Build");

    cl::Kernel reverseBits(program, "reverse_bits");
    cl::Kernel fft(program, "fft");

    std::vector<std::complex<float>> fftSequence(N);
    std::vector<std::complex<float>> fftSequenceBitReversed(N);

    for (size_t i = 0; i < N; ++i) {
        fftSequence[i] = std::complex<float>(i + 1, 0);
    }

    auto start{ std::chrono::high_resolution_clock::now() };

    for (size_t i = 0; i < 100'00) {
        cl::Buffer bufferInput(context, CL_MEM_READ_WRITE, N * sizeof(cl_float2));
        cl::Buffer bufferOutput(context, CL_MEM_READ_WRITE, N * sizeof(cl_float2));

        queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, N * sizeof(cl_float2), fftSequence.data());

        reverseBits.setArg(0, bufferInput);
        reverseBits.setArg(1, bufferOutput);
        reverseBits.setArg(2, (int)std::log2(N));
        queue.enqueueNDRangeKernel(reverseBits, cl::NullRange, cl::NDRange(N), cl::NDRange(blockSizeReverseBits));

        cl::Buffer* currentBuffer = &bufferOutput;
        for (int s = 2; s <= N; s *= 2) {
            fft.setArg(0, *currentBuffer);
            fft.setArg(1, s);
            queue.enqueueNDRangeKernel(fft, cl::NullRange, cl::NDRange(N/2), cl::NDRange(blockSizeFFT));
        }

        queue.enqueueReadBuffer(*currentBuffer, CL_TRUE, 0, N * sizeof(cl_float2), fftSequenceBitReversed.data());
    }

    auto end{ std::chrono::high_resolution_clock::now() };

    std::chrono::duration<double> elapsed{ end - start };
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}