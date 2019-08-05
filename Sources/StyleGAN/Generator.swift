import Foundation
import TensorFlow

public struct GeneratorFirstBlock: Layer {
    var dense: EqualizedDense
    var conv: EqualizedConv2D
    
    public init() {
        dense = EqualizedDense(inputSize: Config.latentSize,
                               outputSize: 256*4*4,
                               activation: lrelu)
        conv = EqualizedConv2D(inputChannels: 256,
                               outputChannels: 256,
                               kernelSize: (3, 3),
                               activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        var x = dense(input)
        x = x.reshaped(to: [batchSize, 4, 4, 256])
        x = pixelNormalization(x)
        x = pixelNormalization(conv(x)) // [batchSize, 256, 4, 4]
        return x
    }
}

public struct GeneratorBlock: Layer {
    var conv1: EqualizedTransposedConv2D
    var conv2: EqualizedConv2D
    
    var upsample = UpSampling2D<Float>(size: 2)
    
    public init(inputChannels: Int, outputChannels: Int) {
        let stride = Config.useFusedScale ? 2 : 1
        
        conv1 = EqualizedTransposedConv2D(inputChannels: inputChannels,
                                          outputChannels: outputChannels,
                                          kernelSize: (3, 3),
                                          strides: (stride, stride),
                                          activation: lrelu)
        conv2 = EqualizedConv2D(inputChannels: outputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        if !Config.useFusedScale {
            x = upsample(x)
        }
        x = pixelNormalization(conv1(x))
        x = pixelNormalization(conv2(x))
        return x
    }
}

public struct Generator: Layer {
    public var firstBlock = GeneratorFirstBlock()
    
    public var blocks: [GeneratorBlock] = []
    
    public var upsample = UpSampling2D<Float>(size: 2)
    
    public var toRGB1 = EqualizedConv2D(inputChannels: 1, outputChannels: 1, kernelSize: (1, 1), activation: identity, gain: 1) // dummy at first
    public var toRGB2 = EqualizedConv2D(inputChannels: 256, outputChannels: 3, kernelSize: (1, 1), activation: identity, gain: 1)
    
    @noDerivative
    public private(set) var level = 1
    @noDerivative
    public var alpha: Float = 1
    
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = firstBlock(input)
        
        guard level > 1 else {
            // 常にalpha = 1
            return toRGB2(x)
        }
        
        for lv in 0..<level-2 {
            x = blocks[lv](x)
        }
        
        var x1 = x
        x1 = toRGB1(x1)
        x1 = upsample(x1)
        
        var x2 = blocks[level-2](x)
        x2 = toRGB2(x2)
        return lerp(x1, x2, rate: alpha)
    }
    
    static let ioChannels = [
        (256, 256),
        (256, 256),
        (256, 128),
        (128, 64),
        (64, 32),
        (32, 16)
    ]
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Generator.level exceeds Config.maxLevel")
        }
        
        let blockCount = blocks.count
        let io = Generator.ioChannels[blockCount]
        
        blocks.append(GeneratorBlock(inputChannels: io.0, outputChannels: io.1))
        toRGB1 = toRGB2
        toRGB2 = EqualizedConv2D(inputChannels: io.1, outputChannels: 3, kernelSize: (1, 1), activation: identity, gain: 1)
    }
}
