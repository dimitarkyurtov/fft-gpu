#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <iostream>

#import "FastFourierTransformCPU.h"
#import "FastFourierTranformMetal.h"

enum ExecutionMode
{
    CPU,
    GPU
};

int main(int argc, const char * argv[]) {
    ExecutionMode mode { ExecutionMode::GPU };
    int fftSequenceLength = -1;

    if (argc == 2)
    {
        fftSequenceLength = std::stoi(argv[1]);
    }
    else if (argc == 3)
    {
        fftSequenceLength = std::stoi(argv[2]);
        std::string firstArg{ argv[1] };

        if (firstArg == "-cpu")
        {
            mode = ExecutionMode::CPU;
        }
        else if (firstArg == "-gpu")
        {
            mode = ExecutionMode::GPU;
        }
        else
        {
            std::cerr << "Error: First argument must be '-cpu' or '-gpu'.\n";
            return 1;
        }
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " [-cpu|-gpu] <natural_number>\n";
        return 1;
    }

    auto fftSequence { std::vector<float>() };
    fftSequence.reserve(fftSequenceLength);

    for (uint i = 1; i <= fftSequenceLength; i ++)
    {
        fftSequence.emplace_back(i);
    }
    
    switch (mode)
    {
        case ExecutionMode::CPU:
        {
            FastFourierTransformCPU *fftCPU = [[FastFourierTransformCPU alloc] initWithFFTSequence:fftSequence];
            [fftCPU fft];
            [fftCPU printIFFTSequence];
            
            // [fftCPU ifft];
            // [fftCPU printFFTSequence];
            break;
        }
        case ExecutionMode::GPU:
        {
            @autoreleasepool
            {
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                FastFourierTranformMetal* fftMetal = [[FastFourierTranformMetal alloc] initWithFFTSequence:fftSequence withDevice:device];
                
                [fftMetal fft];
                [fftMetal printResult];
            }
        }
        default:
            break;
    }

    return 0;
}