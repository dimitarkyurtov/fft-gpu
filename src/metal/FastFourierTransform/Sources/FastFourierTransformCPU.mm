#import "FastFourierTransformCPU.h"
#import <vector>
#import <complex>
#import <iostream>
#import <numbers>

@implementation FastFourierTransformCPU
{
    std::vector<std::complex<double>> fftSequence;
    std::vector<std::complex<double>> ifftSequence;
}

- (instancetype)initWithFFTSequence: (std::vector<float>)fftSequence
{
    self = [super init];
    if (self) {
        self->fftSequence = std::vector<std::complex<double>>(fftSequence.size());
        std::transform(fftSequence.begin(), fftSequence.end(), self->fftSequence.begin(),
                           [](float value) {
                                return std::complex<double>(value);
                           });
    }
    return self;
}

- (std::vector<std::complex<double>>)fft
{
    return ifftSequence = [self fftBase:fftSequence withRootOfUnityFactor:1];
}

- (std::vector<std::complex<double>>)ifft
{
    fftSequence = [self fftBase:ifftSequence withRootOfUnityFactor:-1];
    std::for_each(fftSequence.begin(), fftSequence.end(), [&self](std::complex<double>& el){
        el = el / std::complex<double>(self->ifftSequence.size());
    });

    return fftSequence;
}

- (std::vector<std::complex<double>>)fftBase: (std::vector<std::complex<double>>)fftSequence withRootOfUnityFactor:(double)factor
{
    auto n { fftSequence.size() };
    if (n == 1) {
        return fftSequence;
    }
    // Vectors to store even-indexed and odd-indexed elements
    std::vector<std::complex<double>> evenIndexed;
    std::vector<std::complex<double>> oddIndexed;

    std::copy_if(fftSequence.begin(), fftSequence.end(), std::back_inserter(evenIndexed),
                 [&, index = 0](const std::complex<double>&) mutable {
                     return (index++ % 2) == 0;
                 });

    std::copy_if(fftSequence.begin(), fftSequence.end(), std::back_inserter(oddIndexed),
                 [&, index = 0](const std::complex<double>&) mutable {
                     return (index++ % 2) != 0;
                 });
    
    std::vector<std::complex<double>> evenFFT {
        [self fftBase:evenIndexed withRootOfUnityFactor:factor]
    };
    
    std::vector<std::complex<double>> oddFFT {
        [self fftBase:oddIndexed withRootOfUnityFactor:factor]
    };
    
    std::vector<std::complex<double>> result { std::vector<std::complex<double>>(n) };
    
    for (int j = 0; j < n/2; j ++) {
        constexpr double pi = std::numbers::pi;
        double angle = factor * 2 * j * pi / fftSequence.size();
        std::complex<double> rootOfUnity = std::exp(std::complex<double>(0, angle)); // e^(2*pi*i*j/n)

        result[j] = evenFFT[j] + rootOfUnity * oddFFT[j];
        result[j+n/2] = evenFFT[j] - rootOfUnity * oddFFT[j];
    }
    
    return result;
}

- (void)printFFTSequence
{
    for (const auto& el : fftSequence) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

- (void)printIFFTSequence
{
    for (const auto& el : ifftSequence) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

- (std::complex<double>)roundToNthDigitWithNumber: (std::complex<double>)number digits:(unsigned)digits
{
    double imagRounded = std::round(number.imag()*std::pow(10, digits))/static_cast<int>(std::pow(10, digits));
    double realRounded = std::round(number.real()*std::pow(10, digits))/static_cast<int>(std::pow(10, digits));
    return std::complex<double>(realRounded, imagRounded);
}

@end

