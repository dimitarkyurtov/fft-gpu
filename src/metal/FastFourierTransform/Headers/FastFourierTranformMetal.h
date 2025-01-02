#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface FastFourierTranformMetal : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) prepareData;
- (void) sendComputeCommand;
- (void) printResult;
@end

NS_ASSUME_NONNULL_END
