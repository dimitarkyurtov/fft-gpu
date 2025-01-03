#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <iostream>

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
        FastFourierTranformMetal* fftMetal = [[FastFourierTranformMetal alloc] initWithFFTSequence:base withDevice:device];
        
        [fftMetal prepareData];
        [fftMetal sendComputeCommand];
        std::cout << "Metal: ";
        [fftMetal printResult];
    }
    return 0;
}
