#include <metal_stdlib>
#include <metal_numeric>
using namespace metal;

struct Complex {
    float real;
    float imag;
    
    Complex(float real = 0, float imag = 0) : real{ real }, imag{ imag } {}
    Complex(const thread Complex& c) : real{ c.real }, imag{ c.imag } {}
    Complex(const device Complex& c) : real{ c.real }, imag{ c.imag } {}
};

Complex add(const thread Complex& a, const thread Complex& b) {
    return Complex(a.real + b.real, a.imag + b.imag);
}

Complex subtract(const thread Complex& a, const thread Complex& b) {
    return Complex(a.real - b.real, a.imag - b.imag);
}

Complex multiply(const thread Complex& a, const thread Complex& b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

Complex exp(Complex z) {
    float expReal = exp(z.real);
    float cosImag = cos(z.imag);
    float sinImag = sin(z.imag);

    return Complex(expReal * cosImag, expReal * sinImag);
}

unsigned int reverseBits(unsigned int n, unsigned int numBits) {
    unsigned int reversed = 0;
    for (unsigned int i = 0; i < numBits; ++i) {
        unsigned int lsb = n & 1;
        reversed = (reversed << 1) | lsb;
        n >>= 1;
    }
    return reversed;
}

kernel void fft(device const Complex* fftSequence,
                device Complex* fftSequenceBitReversed,
                device const uint* N,
                uint index [[thread_position_in_grid]])
{
    fftSequenceBitReversed[reverseBits(index*2, log2(static_cast<half>(*N)))] = Complex(fftSequence[index*2].real, fftSequence[index*2].imag);
    fftSequenceBitReversed[reverseBits(index*2+1, log2(static_cast<half>(*N)))] = Complex(fftSequence[index*2+1].real, fftSequence[index*2+1].imag);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 2; s <= *N; s *= 2)
    {
        uint k { index/(s/2) * s };
        uint j { index%(s/2) };
        
        auto W{ exp(Complex(0, -2 * M_PI_F * j / s)) };
        
        Complex c1 = fftSequenceBitReversed[k+j];
        Complex c2 = fftSequenceBitReversed[k+j+s/2];
        Complex c3 = multiply(W, c2);
        
        fftSequenceBitReversed[k+j] = add(c1, c3);
        fftSequenceBitReversed[k+j+s/2] = subtract(c1, c3);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
