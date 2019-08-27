import Foundation
import TensorFlow

struct SynthesisFirstBlock: Layer {
    var baseImage: Tensor<Float>
    var conv: EqualizedConv2D
    
    var noise1: NoiseLayer
    var noise2: NoiseLayer
    
    var adaIN1: AdaIN
    var adaIN2: AdaIN
    
    init(startSize: Int, outputSize: Int) {
        baseImage = Tensor(ones: [1, 4, 4, startSize])
        
        conv = EqualizedConv2D(inputChannels: startSize,
                               outputChannels: outputSize,
                               kernelSize: (3, 3))
        
        noise1 = NoiseLayer(channels: startSize)
        noise2 = NoiseLayer(channels: outputSize)
        
        adaIN1 = AdaIN(size: startSize, wsize: Config.wsize)
        adaIN2 = AdaIN(size: outputSize, wsize: Config.wsize)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let batchSize = input.shape[0]
        var x = baseImage.tiled(multiples: Tensor<Int32>([Int32(batchSize), 1, 1, 1]))
        if Config.useNoise {
            x = noise1(x)
        }
        x = lrelu(x)
        x = adaIN1(AdaIN.makeInput(x: x, w: input))
        
        x = conv(x)
        if Config.useNoise {
            x = noise2(x)
        }
        x = lrelu(x)
        x = adaIN2(AdaIN.makeInput(x: x, w: input))
        
        return x
    }
}

struct SynthesisBlock: Layer {
    struct Input: Differentiable {
        var x: Tensor<Float>
        var w: Tensor<Float>
    }
    
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    var noise1: NoiseLayer
    var noise2: NoiseLayer
    
    var adaIN1: AdaIN
    var adaIN2: AdaIN
    
    @noDerivative let blur: Blur3x3
    
    init(inputSize: Int, outputSize: Int) {
        conv1 = EqualizedConv2D(inputChannels: inputSize,
                                outputChannels: outputSize,
                                kernelSize: (3, 3))
        conv2 = EqualizedConv2D(inputChannels: outputSize,
                                outputChannels: outputSize,
                                kernelSize: (3, 3))
        
        noise1 = NoiseLayer(channels: outputSize)
        noise2 = NoiseLayer(channels: outputSize)
        
        adaIN1 = AdaIN(size: outputSize, wsize: Config.wsize)
        adaIN2 = AdaIN(size: outputSize, wsize: Config.wsize)
        
        blur = Blur3x3(channels: outputSize)
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        var x = input.x
        
        x = resize2xBilinear(images: x)
        
        x = conv1(x)
        if Config.useBlur {
            x = blur(x)
        }
        if Config.useNoise {
            x = noise1(x)
        }
        x = lrelu(x)
        x = adaIN1(AdaIN.makeInput(x: x, w: input.w))
        
        x = conv2(x)
        if Config.useNoise {
            x = noise2(x)
        }
        x = lrelu(x)
        x = adaIN2(AdaIN.makeInput(x: x, w: input.w))
        
        return x
    }
}

public struct SynthesisModule: Layer {
    var firstBlock: SynthesisFirstBlock
    
    var blocks: [SynthesisBlock] = []
    
    var toRGB1 = EqualizedConv2D(inputChannels: 1,
                                 outputChannels: 1,
                                 kernelSize: (1, 1),
                                 activation: identity,
                                 gain: 1) // dummy at first
    var toRGB2 = EqualizedConv2D(inputChannels: 256,
                                 outputChannels: 3,
                                 kernelSize: (1, 1),
                                 activation: identity,
                                 gain: 1)
    
    @noDerivative public internal(set) var level: Int = 1
    @noDerivative public var alpha: Float = 1
    
    public init() {
        firstBlock = .init(startSize: 256, outputSize: 256)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let w = input
        
        var x = firstBlock(w)
        
        guard level > 1 else {
            // 常にalpha = 1
            return toRGB2(x)
        }
        
        for lv in 0..<level-2 {
            x = blocks[lv](.init(x: x, w: w))
        }
        
        var x1 = x
        x1 = toRGB1(x1)
        x1 = resize2xBilinear(images: x1)
        
        var x2 = blocks[level-2](.init(x: x, w: w))
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
        let io = SynthesisModule.ioChannels[blockCount]
        
        blocks.append(SynthesisBlock(inputSize: io.0, outputSize: io.1))
        toRGB1 = toRGB2
        toRGB2 = EqualizedConv2D(inputChannels: io.1,
                                 outputChannels: 3,
                                 kernelSize: (1, 1),
                                 activation: identity,
                                 gain: 1)
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict = [
            "gen\(level)/first.baseImage": firstBlock.baseImage,
            "gen\(level)/first.conv": firstBlock.conv.filter,
            "gen\(level)/first.adain1.scale": firstBlock.adaIN1.scaleTransform.weight,
            "gen\(level)/first.adain1.bias": firstBlock.adaIN1.biasTransform.weight,
            "gen\(level)/first.adain2.scale": firstBlock.adaIN2.scaleTransform.weight,
            "gen\(level)/first.adain2.bias": firstBlock.adaIN2.biasTransform.weight,
            "gen\(level)/toRGB1": toRGB1.filter,
            "gen\(level)/toRGB2": toRGB2.filter,
        ]
        
        for i in 0..<blocks.count {
            dict["gen\(level)/block\(i).conv1"] = blocks[i].conv1.filter
            dict["gen\(level)/block\(i).conv2"] = blocks[i].conv2.filter
            dict["gen\(level)/block\(i).adain1.scale"] = blocks[i].adaIN1.scaleTransform.weight
            dict["gen\(level)/block\(i).adain1.bias"] = blocks[i].adaIN1.biasTransform.bias
            dict["gen\(level)/block\(i).adain2.scale"] = blocks[i].adaIN2.scaleTransform.weight
            dict["gen\(level)/block\(i).adain2.bias"] = blocks[i].adaIN2.biasTransform.bias
        }
        
        return dict
    }
}
