#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "FastFourierTransformCPU.h"
#import "FastFourierTranformMetal.h"

//const unsigned int arrayLength = 1 << 24;
//const unsigned int bufferSize = arrayLength * sizeof(float);

int main(int argc, const char * argv[]) {
    std::vector<float> base {1,2,3,4,5,6,7,8};
    
    FastFourierTransformCPU *fftCPU = [[FastFourierTransformCPU alloc] initWithFFTSequence:base];
    [fftCPU fft];
    [fftCPU ifft];
    [fftCPU printFFTSequence];
    [fftCPU printIFFTSequence];
    
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        FastFourierTranformMetal* fftMetal = [[FastFourierTranformMetal alloc] initWithDevice:device];
        
        [fftMetal prepareData];
        [fftMetal sendComputeCommand];
        [fftMetal printResult];
    }
    return 0;
}
