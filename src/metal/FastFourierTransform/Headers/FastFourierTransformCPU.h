#import <Foundation/Foundation.h>
#import <vector>
#import <complex>

NS_ASSUME_NONNULL_BEGIN

@interface FastFourierTransformCPU : NSObject
- (instancetype) initWithFFTSequence: (std::vector<float>)fftSequence;
- (std::vector<std::complex<double>>)fft;
- (std::vector<std::complex<double>>)ifft;
- (void)printFFTSequence;
- (void)printIFFTSequence;
@end

NS_ASSUME_NONNULL_END
