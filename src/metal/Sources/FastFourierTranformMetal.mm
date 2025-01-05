#import <iostream>
#import <complex>

#import "FastFourierTranformMetal.h"

@implementation FastFourierTranformMetal
{
    id<MTLDevice> _mDevice;

    id<MTLComputePipelineState> _mFFTFunctionCP;
    id<MTLComputePipelineState> _mReverseBitsFunctionCP;

    id<MTLCommandQueue> _mCommandQueue;

    id<MTLBuffer> _mBufferFFTSequence;
    id<MTLBuffer> _mBufferFFTSequenceBitsReversed;
    id<MTLBuffer> _mBufferSize;
    id<MTLBuffer> _mBufferElementsPerSequence;

    std::vector<std::complex<float>> fftSequence;
    std::vector<std::complex<float>> ifftSequence;
}

- (instancetype) initWithFFTSequence: (std::vector<float>)fftSequence withDevice: (id<MTLDevice>) device;
{
    self = [super init];
    if (self) {
        self->fftSequence = std::vector<std::complex<float>>(fftSequence.size());
        std::transform(fftSequence.begin(), fftSequence.end(), self->fftSequence.begin(),
                           [](float value) {
                                return std::complex<float>(value);
                           });
        
        _mDevice = device;
        NSError* error = nil;

        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        if (defaultLibrary == nil) {
            NSLog(@"Failed to find the default library.");
            return nil;
        }

        id<MTLFunction> fftFunction = [defaultLibrary newFunctionWithName:@"fft"];
        if (fftFunction == nil) {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }

        _mFFTFunctionCP = [_mDevice newComputePipelineStateWithFunction: fftFunction error:&error];
        if (_mFFTFunctionCP == nil) {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return nil;
        }
        
        id<MTLFunction> reverseBitsFunction = [defaultLibrary newFunctionWithName:@"reverse_bits"];
        if (reverseBitsFunction == nil) {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }

        _mReverseBitsFunctionCP = [_mDevice newComputePipelineStateWithFunction: reverseBitsFunction error:&error];
        if (_mReverseBitsFunctionCP == nil) {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return nil;
        }

        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil) {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }

    return self;
}

/// See the header
- (void) fft
{
    [self prepareData];

    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodeReverseBitsCommand:computeEncoder];
    
    for (uint s = 2; s <= fftSequence.size(); s *= 2)
    {
        [self encodeFFTCommand:computeEncoder];
    }

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

/// See the header.
- (void) printResult
{
    std::cout << "Metal: ";
    auto dataPtr{ static_cast<std::complex<float>*>(_mBufferFFTSequenceBitsReversed.contents) };

    for (unsigned long index = 0; index < fftSequence.size(); index++) {
        std::cout << dataPtr[index] << " ";
    }
    std::cout << std::endl;
}

/// Helper function which sets up the necessary buffers used by the GPU.
- (void) prepareData
{
    _mBufferFFTSequence = [_mDevice newBufferWithLength:fftSequence.size()*sizeof(std::complex<float>) options:MTLResourceStorageModeShared];
    _mBufferSize = [_mDevice newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    _mBufferElementsPerSequence = [_mDevice newBufferWithLength:sizeof(uint) options:MTLResourceStorageModeShared];
    _mBufferFFTSequenceBitsReversed = [_mDevice newBufferWithLength:fftSequence.size()*sizeof(std::complex<float>) options:MTLResourceStorageModeShared];

    memcpy(_mBufferFFTSequence.contents, fftSequence.data(), fftSequence.size() * sizeof(std::complex<float>));
    static_cast<uint*>(_mBufferSize.contents)[0] = static_cast<uint>(fftSequence.size());
    static_cast<uint*>(_mBufferElementsPerSequence.contents)[0] = 2;
}

/// Encodes a reverse bit Metal function to an encoder.
/// - Parameter computeEncoder: The compute encoder.
- (void)encodeReverseBitsCommand:(id<MTLComputeCommandEncoder>)computeEncoder
{
    [computeEncoder setComputePipelineState:_mReverseBitsFunctionCP];
    [computeEncoder setBuffer:_mBufferFFTSequence offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferFFTSequenceBitsReversed offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferSize offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(fftSequence.size(), 1, 1);

    NSUInteger threadGroupSize = _mReverseBitsFunctionCP.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > fftSequence.size()) {
        threadGroupSize = fftSequence.size();
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

/// Encodes an iteration of the fft Metal function to an encoder.
/// - Parameter computeEncoder: The compute encoder.
- (void)encodeFFTCommand:(id<MTLComputeCommandEncoder>)computeEncoder
{
    [computeEncoder setComputePipelineState:_mFFTFunctionCP];
    
    [computeEncoder setBuffer:_mBufferFFTSequenceBitsReversed offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferElementsPerSequence offset:0 atIndex:1];

    auto numberOfThreads { fftSequence.size()/2 };
    MTLSize gridSize = MTLSizeMake(numberOfThreads, 1, 1);

    NSUInteger threadGroupSize = _mFFTFunctionCP.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > numberOfThreads) {
        threadGroupSize = numberOfThreads;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}


@end
