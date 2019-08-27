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
    var firstToRGB: EqualizedConv2D
    
    var blocks: [SynthesisBlock] = []
    var toRGBs: [EqualizedConv2D] = []
    
    @noDerivative public internal(set) var level: Int = 1
    @noDerivative public var alpha: Float = 1
    
    static let channels = [256, 256, 256, 256, 128, 64, 32, 16]
    
    public init() {
        let channels = Self.channels
        firstBlock = .init(startSize: channels[0], outputSize: channels[1])
        firstToRGB = EqualizedConv2D(inputChannels: channels[1], outputChannels: 3,
                                     kernelSize: (1, 1), padding: .valid)
        
        for lv in 2...Config.maxLevel {
            blocks.append(.init(inputSize: channels[lv-1], outputSize: channels[lv]))
            toRGBs.append(.init(inputChannels: channels[lv], outputChannels: 3,
                                kernelSize: (1, 1), padding: .valid, gain: 1))
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let w = input
        
        var x = firstBlock(w)
        
        guard level > 1 else {
            // 常にalpha = 1
            return firstToRGB(x)
        }
        guard level > 2 else {
            let rgb1 = resize2xBilinear(images: firstToRGB(x))
            let rgb2 = toRGBs[0](blocks[0](.init(x: x, w: w)))
            return lerp(rgb1, rgb2, rate: alpha)
        }
        
        // Level >= 3
        
        // FIXME: Loop causes crash. Unroll it.
        // https://bugs.swift.org/projects/TF/issues/TF-681
//        for i in 0..<level-2 {
//            x = blocks[i](.init(x: x, w: w))
//        }
        var i = 0
        if i < level-2 {
            x = blocks[i](.init(x: x, w: w))
            i += 1
        }
        if i < level-2 {
            x = blocks[i](.init(x: x, w: w))
            i += 1
        }
        if i < level-2 {
            x = blocks[i](.init(x: x, w: w))
            i += 1
        }
        if i < level-2 {
            x = blocks[i](.init(x: x, w: w))
            i += 1
        }
        if i < level-2 {
            x = blocks[i](.init(x: x, w: w))
            i += 1
        }
        
        let rgb1 = resize2xBilinear(images: toRGBs[level-3](x))
        
        x = blocks[level-2](.init(x: x, w: w))
        let rgb2 = toRGBs[level-2](x)
        
        return lerp(rgb1, rgb2, rate: alpha)
    }
    
    public mutating func grow() {
        level += 1
        guard level <= Config.maxLevel else {
            fatalError("Generator.level exceeds Config.maxLevel")
        }
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        var dict = [
            "gen/first.baseImage": firstBlock.baseImage,
            "gen/first.conv": firstBlock.conv.filter,
            "gen/first.adain1.scale": firstBlock.adaIN1.scaleTransform.weight,
            "gen/first.adain1.bias": firstBlock.adaIN1.biasTransform.weight,
            "gen/first.adain2.scale": firstBlock.adaIN2.scaleTransform.weight,
            "gen/first.adain2.bias": firstBlock.adaIN2.biasTransform.weight,
            "gen/first.toRGB": firstToRGB.filter,
        ]
        
        for i in 0..<level-1 {
            dict["gen/block\(i).conv1"] = blocks[i].conv1.filter
            dict["gen/block\(i).conv2"] = blocks[i].conv2.filter
            dict["gen/block\(i).adain1.scale"] = blocks[i].adaIN1.scaleTransform.weight
            dict["gen/block\(i).adain1.bias"] = blocks[i].adaIN1.biasTransform.bias
            dict["gen/block\(i).adain2.scale"] = blocks[i].adaIN2.scaleTransform.weight
            dict["gen/block\(i).adain2.bias"] = blocks[i].adaIN2.biasTransform.bias
            dict["gen/block\(i).toRGB"] = toRGBs[i].filter
        }
        
        return dict
    }
}
