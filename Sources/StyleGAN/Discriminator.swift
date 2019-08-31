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

struct DBlock: Layer {
    struct Input: Differentiable {
        var x: Tensor<Float>
        @noDerivative
        var noiseScale: Float
    }
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
    func callAsFunction(_ input: Input) -> Tensor<Float> {
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
    struct Input: Differentiable {
        var x: Tensor<Float>
        @noDerivative
        var noiseScale: Float
    }
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
    public func callAsFunction(_ input: Input) -> Tensor<Float> {
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
    
    static let channels = [1, 256, 256, 256, 128, 64, 32, 16]
    
    var lastBlock = DLastBlock()
    var lastFromRGB = EqualizedConv2D(inputChannels: 3, outputChannels: channels[1],
                                      kernelSize: (1, 1), padding: .valid, activation: lrelu)
    
    var blocks: [DBlock] = []
    var fromRGBs: [EqualizedConv2D] = []
    
    @noDerivative
    public private(set) var level = 1
    @noDerivative
    public var alpha: Float = 1
    
    // Mean of output for fake images
    @noDerivative
    public let outputMean: Parameter<Float> = Parameter(Tensor(0))
    
    public init() {
        let channels = Self.channels
        for lv in 2...Config.maxLevel {
            fromRGBs.append(.init(inputChannels: 3, outputChannels: channels[lv], kernelSize: (1, 1), padding: .valid, activation: lrelu))
            blocks.append(.init(inputChannels: channels[lv], outputChannels: channels[lv-1]))
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
        // Described in Appendix B of ProGAN paper
        let noiseScale: Float
        if Config.loss == .lsgan {
            noiseScale = 0.2 * pow(max(outputMean.value.scalarized() - 0.5, 0), 2)
        } else {
            noiseScale = 0
        }
        
        guard level > 1 else {
            // alpha = 1
            let x = lastFromRGB(input)
            return lastBlock(.init(x: x, noiseScale: noiseScale))
        }
        guard level > 2 else {
            let x1 = lastFromRGB(avgPool(input))
            var x2 = fromRGBs[0](input)
            x2 = blocks[0](.init(x: x2, noiseScale: noiseScale))
            let x = lerp(x1, x2, rate: alpha)
            return lastBlock(.init(x: x, noiseScale: noiseScale))
        }
        
        // Level >= 3
        
        let x1 = fromRGBs[level-3](avgPool(input))
        var x2 = fromRGBs[level-2](input)
        x2 = blocks[level-2](.init(x: x2, noiseScale: noiseScale))
        
        var x = lerp(x1, x2, rate: alpha)
        
        // FIXME: Loop has problem. Unroll it.
        // https://bugs.swift.org/projects/TF/issues/TF-681
//        for i in (0..<level-2).reversed() {
//            x = blocks[i](.init(x: x, noiseScale: noiseScale))
//        }
        var i = level-3
        if i >= 0 {
            x = blocks[i](.init(x: x, noiseScale: noiseScale))
            i -= 1
        }
        if i >= 0 {
            x = blocks[i](.init(x: x, noiseScale: noiseScale))
            i -= 1
        }
        if i >= 0 {
            x = blocks[i](.init(x: x, noiseScale: noiseScale))
            i -= 1
        }
        if i >= 0 {
            x = blocks[i](.init(x: x, noiseScale: noiseScale))
            i -= 1
        }
        if i >= 0 {
            x = blocks[i](.init(x: x, noiseScale: noiseScale))
            i -= 1
        }
        
        return lastBlock(.init(x: x, noiseScale: noiseScale))
    }
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Discriminator.level exceeds Config.maxLevel")
        }
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict = [
            "disc/last.conv1": lastBlock.conv1.filter,
            "disc/last.conv2": lastBlock.conv2.filter,
            "disc/last.dense": lastBlock.dense.weight,
            "disc/last.fromRGB": lastFromRGB.filter,
        ]
        
        for i in 0..<level-1 {
            dict["disc/block\(i).conv1"] = blocks[i].conv1.filter
            dict["disc/block\(i).conv2"] = blocks[i].conv2.filter
            dict["disc/block\(i).fromRGB"] = fromRGBs[i].filter
        }
        
        return dict
    }
}
