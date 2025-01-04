#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// Perform iterative FFT algorithm based on the Cooley-Tukey FFT algorithm.
/// GPU implementation using the Metal framework.
@interface FastFourierTranformMetal : NSObject

/// Initializes the internal structures.
/// - Parameter fftSequence: The input vector.
/// - Parameter device: The GPU to run the computations on.
- (instancetype) initWithFFTSequence: (std::vector<float>)fftSequence withDevice: (id<MTLDevice>) device;

/// Performs the FFT algorithm.
- (void) fft;

/// Prints the result vector of solving the DFT equation using the FFT algorithm.
- (void) printResult;

@end

NS_ASSUME_NONNULL_END
