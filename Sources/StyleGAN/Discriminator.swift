import Foundation
import TensorFlow

// Add per channel noise
// https://github.com/tkarras/progressive_growing_of_gans/blob/original-theano-version/network.py#L360-L400
@differentiable(wrt: x)
func addNoise(_ x: Tensor<Float>, noiseScale: Float) -> Tensor<Float> {
    let noiseShape: TensorShape = [1, 1, 1, x.shape[3]]
    let scale = noiseScale * sqrt(Float(x.shape[3]))
    let noise = Tensor<Float>(randomNormal: noiseShape) * scale + 1
    return x * noise
}

@differentiable
func avgPool(_ x: Tensor<Float>) -> Tensor<Float> {
    avgPool2D(x, filterSize: (1, 2, 2, 1), strides: (1, 2, 2, 1), padding: .valid)
}

struct DBlockInput: Differentiable {
    var x: Tensor<Float>
    @noDerivative
    var noiseScale: Float
}

struct DBlock: Layer {
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    @noDerivative
    let blur: Blur3x3
    
    init(inputChannels: Int, outputChannels: Int) {
        let stride = Config.useFusedScale ? 2 : 1
        
        conv1 = EqualizedConv2D(inputChannels: inputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                activation: lrelu)
        conv2 = EqualizedConv2D(inputChannels: outputChannels,
                                outputChannels: outputChannels,
                                kernelSize: (3, 3),
                                strides: (stride, stride),
                                activation: lrelu)
        blur = Blur3x3(channels: outputChannels)
    }
    
    @differentiable
    func callAsFunction(_ input: DBlockInput) -> Tensor<Float> {
        var x = input.x
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv1(x)
        if Config.useBlur {
            x = blur(x)
        }
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv2(x)
        if !Config.useFusedScale {
            x = avgPool(x)
        }
        return x
    }
}

struct DLastBlock: Layer {
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    var dense: EqualizedDense
    
    public init() {
        conv1 = EqualizedConv2D(inputChannels: 256,
                                outputChannels: 256,
                                kernelSize: (3, 3),
                                padding: .same,
                                activation: lrelu)
        conv2 = EqualizedConv2D(inputChannels: 256,
                                outputChannels: 256,
                                kernelSize: (4, 4),
                                padding: .valid,
                                activation: lrelu)
        dense = EqualizedDense(inputSize: 256,
                               outputSize: 1,
                               activation: identity,
                               gain: 1)
    }
    
    @differentiable
    public func callAsFunction(_ input: DBlockInput) -> Tensor<Float> {
        var x = input.x
        let batchSize = x.shape[0]
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv1(x)
        x = addNoise(x, noiseScale: input.noiseScale)
        x = conv2(x)
        x = x.reshaped(to: [batchSize, -1])
        x = dense(x)
        return x
    }
}

public struct Discriminator: Layer {
    
    var lastBlock = DLastBlock()
    
    var blocks: [DBlock] = []
    
    var fromRGB1 = EqualizedConv2D(inputChannels: 1, outputChannels: 1, kernelSize: (1, 1), activation: lrelu) // dummy at first
    var fromRGB2 = EqualizedConv2D(inputChannels: 3, outputChannels: 256, kernelSize: (1, 1), activation: lrelu)
    
    @noDerivative
    public private(set) var level = 1
    @noDerivative
    public var alpha: Float = 1
    
    // Mean of output for fake images
    @noDerivative
    public let outputMean: Parameter<Float> = Parameter(Tensor(0))
    
    public init() {}
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
        // Described in Appendix B of ProGAN paper
        let noiseScale: Float
        if Config.loss == .lsgan {
            noiseScale = 0.2 * pow(max(outputMean.value.scalar! - 0.5, 0), 2)
        } else {
            noiseScale = 0
        }
        
        guard level > 1 else {
            // alpha = 1
            return lastBlock(DBlockInput(x: fromRGB2(input), noiseScale: noiseScale))
        }
        
        let x1 = fromRGB1(avgPool(input))
        var x2 = fromRGB2(input)
        
        let lastIndex = level-2
        x2 = blocks[lastIndex](DBlockInput(x: x2, noiseScale: noiseScale))
        
        var x = lerp(x1, x2, rate: alpha)
        
        for l in (0..<lastIndex).reversed() {
            x = blocks[l](DBlockInput(x: x, noiseScale: noiseScale))
        }
        
        return lastBlock(DBlockInput(x: x, noiseScale: noiseScale))
    }
    
    static let ioChannels = [
        (256, 256),
        (256, 256),
        (128, 256),
        (64, 128),
        (32, 64),
        (16, 32),
    ]
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Generator.level exceeds Config.maxLevel")
        }
        
        let blockCount = blocks.count
        let io = Discriminator.ioChannels[blockCount]
        
        blocks.append(DBlock(inputChannels: io.0,outputChannels: io.1))
        
        fromRGB1 = fromRGB2
        fromRGB2 = EqualizedConv2D(inputChannels: 3, outputChannels: io.0, kernelSize: (1, 1), activation: lrelu)
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict = [
            "disc\(level)/last.conv1": lastBlock.conv1.filter,
            "disc\(level)/last.conv2": lastBlock.conv2.filter,
            "disc\(level)/last.dense": lastBlock.dense.weight,
            "disc\(level)/fromRGB1": fromRGB1.filter,
            "disc\(level)/fromRGB2": fromRGB2.filter,
        ]
        
        for i in 0..<blocks.count {
            dict["disc\(level)/block\(i).conv1"] = blocks[i].conv1.filter
            dict["disc\(level)/block\(i).conv2"] = blocks[i].conv2.filter
        }
        
        return dict
    }
}
