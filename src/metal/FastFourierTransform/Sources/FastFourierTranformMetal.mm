#import <iostream>
#import "FastFourierTranformMetal.h"

const unsigned int arrayLength = 1 << 5;
const unsigned int bufferSize = arrayLength * sizeof(float);

@implementation FastFourierTranformMetal
{
    id<MTLDevice> _mDevice;

    id<MTLComputePipelineState> _mAddFunctionPSO;

    id<MTLCommandQueue> _mCommandQueue;

    id<MTLBuffer> _mBufferA;
    id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferResult;

}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self) {
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
    _mBufferA = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferB = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    [self generateRandomFloatData:_mBufferA];
    [self generateRandomFloatData:_mBufferB];
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
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);

    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength) {
        threadGroupSize = arrayLength;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) generateRandomFloatData: (id<MTLBuffer>) buffer
{
    float* dataPtr = static_cast<float*>(buffer.contents);

    for (unsigned long index = 0; index < arrayLength; index++) {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}

- (void) printResult
{
    float* dataPtr = static_cast<float*>(_mBufferResult.contents);

    for (unsigned long index = 0; index < arrayLength; index++) {
        std::cout << dataPtr[index] << " ";
    }
    std::cout << std::endl;
}


@end
