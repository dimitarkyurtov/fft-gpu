#import <iostream>
#import <complex>

#import "FastFourierTranformMetal.h"

@implementation FastFourierTranformMetal
{
    id<MTLDevice> _mDevice;

    id<MTLComputePipelineState> _mAddFunctionPSO;

    id<MTLCommandQueue> _mCommandQueue;

    id<MTLBuffer> _mBufferA;
    id<MTLBuffer> _mBufferSize;
    id<MTLBuffer> _mBufferResult;

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

        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"fft"];
        if (addFunction == nil) {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }

        _mAddFunctionPSO = [_mDevice newComputePipelineStateWithFunction: addFunction error:&error];
        if (_mAddFunctionPSO == nil) {
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

- (void) prepareData
{
    _mBufferA = [_mDevice newBufferWithLength:fftSequence.size()*sizeof(float) options:MTLResourceStorageModeShared];
    _mBufferSize = [_mDevice newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:fftSequence.size()*sizeof(float) options:MTLResourceStorageModeShared];

    [self populateBuffer:_mBufferA];
    static_cast<uint*>(_mBufferSize.contents)[0] = static_cast<uint>(fftSequence.size());
}

- (void) sendComputeCommand
{
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodeAddCommand:computeEncoder];

    [computeEncoder endEncoding];

    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];
}

- (void)encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder
{
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferSize offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(fftSequence.size(), 1, 1);

    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > fftSequence.size()) {
        threadGroupSize = fftSequence.size();
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) populateBuffer: (id<MTLBuffer>) buffer
{
    memcpy(buffer.contents, fftSequence.data(), fftSequence.size() * sizeof(std::complex<float>));
}

- (void) printResult
{
    auto dataPtr{ static_cast<std::complex<float>*>(_mBufferResult.contents) };

    for (unsigned long index = 0; index < fftSequence.size(); index++) {
        std::cout << dataPtr[index] << " ";
    }
    std::cout << std::endl;
}


@end
