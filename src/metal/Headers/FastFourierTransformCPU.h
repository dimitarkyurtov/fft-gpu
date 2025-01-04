#import <Foundation/Foundation.h>
#import <vector>
#import <complex>

NS_ASSUME_NONNULL_BEGIN

/// Perform recursive FFT algorithm based on the Cooley-Tukey FFT algorithm.
/// Single threaded CPU implementation.
@interface FastFourierTransformCPU : NSObject

/// Initializes the internal structures.
/// - Parameter fftSequence: The input vector.
- (instancetype) initWithFFTSequence: (std::vector<float>)fftSequence;

/// Performs the FFT algorithm.
- (std::vector<std::complex<double>>)fft;

/// Performs the Inverse FFT algorithm on the output from the FFT algorithm.
- (std::vector<std::complex<double>>)ifft;

/// Prints the result vector of solving the DFT equation using the FFT algorithm.
- (void)printFFTSequence;

/// Prints the result vector of solving the DFT equation using the FFT algorithm on the result vector.
/// Should be equal to the input sequence.
- (void)printIFFTSequence;
@end

NS_ASSUME_NONNULL_END
