#include <metal_stdlib>
#include <metal_numeric>
using namespace metal;

/// Struct representing complex numbers.
struct Complex {
    float real;
    float imag;
    
    Complex(float real = 0, float imag = 0) : real{ real }, imag{ imag } {}
    Complex(const thread Complex& c) : real{ c.real }, imag{ c.imag } {}
    Complex(const device Complex& c) : real{ c.real }, imag{ c.imag } {}
};

/// Adds 2 complex numbers and returns the addition.
Complex add(const thread Complex& a, const thread Complex& b) {
    return Complex(a.real + b.real, a.imag + b.imag);
}

/// Subtracts 2 complex numbers and returns the subtraction.
Complex subtract(const thread Complex& a, const thread Complex& b) {
    return Complex(a.real - b.real, a.imag - b.imag);
}

/// Multiplies 2 complex numbers and returns the multiplication.
Complex multiply(const thread Complex& a, const thread Complex& b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

/// Returns e^z.
Complex exp(Complex z) {
    float expReal = exp(z.real);
    float cosImag = cos(z.imag);
    float sinImag = sin(z.imag);

    return Complex(expReal * cosImag, expReal * sinImag);
}

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

kernel void reverse_bits(device const Complex* fftSequence,
                         device Complex* fftSequenceBitReversed,
                         device const uint* N,
                         uint index [[thread_position_in_grid]])
{
    fftSequenceBitReversed[reverseBits(index, log2(static_cast<half>(*N)))] = Complex(fftSequence[index].real, fftSequence[index].imag);
}

kernel void fft(device Complex* fftSequenceResult,
                device uint* s,
                uint index [[thread_position_in_grid]])
{
    auto ss { *s };
    uint k { index/(ss/2) * (ss) };
    uint j { index%(ss/2) };
    
    auto W{ exp(Complex(0, -2 * M_PI_F * j / (ss))) };
    
    Complex c1 { fftSequenceResult[k+j] };
    Complex c2 { fftSequenceResult[k+j+ss/2] };
    Complex c3 { multiply(W, c2) };

    fftSequenceResult[k+j] = add(c1, c3);
    fftSequenceResult[k+j+ss/2] = subtract(c1, c3);
    
    if (index == 0)
    {
        *s *= 2;
    }
}
