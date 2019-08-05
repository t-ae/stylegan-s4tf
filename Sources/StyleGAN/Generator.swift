import Foundation
import TensorFlow

struct GFirstBlock: Layer {
    struct Input: Differentiable {
        var w1: Tensor<Float>
        var w2: Tensor<Float>
    }
    
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
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        let batchSize = input.w1.shape[0]
        var x = baseImage.tiled(multiples: Tensor<Int32>([Int32(batchSize), 1, 1, 1]))
        x = noise1(x)
        x = lrelu(x)
        x = adaIN1(AdaIN.makeInput(x: x, w: input.w1))
        
        x = conv(x)
        x = noise2(x)
        x = lrelu(x)
        x = adaIN2(AdaIN.makeInput(x: x, w: input.w2))
        
        return x
    }
}

struct GBlock: Layer {
    struct Input: Differentiable {
        var x: Tensor<Float>
        var w1: Tensor<Float>
        var w2: Tensor<Float>
    }
    
    var conv1: EqualizedConv2D
    var conv2: EqualizedConv2D
    
    var noise1: NoiseLayer
    var noise2: NoiseLayer
    
    var adaIN1: AdaIN
    var adaIN2: AdaIN
    
    @noDerivative let blur: Blur3x3
    
    var upsample = UpSampling2D<Float>(size: 2)
    
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
        
        x = upsample(x)
        
        x = conv1(x)
        x = blur(x)
        x = noise1(x)
        x = lrelu(x)
        x = adaIN1(AdaIN.makeInput(x: x, w: input.w1))
        
        x = conv2(x)
        x = noise2(x)
        x = lrelu(x)
        x = adaIN2(AdaIN.makeInput(x: x, w: input.w2))
        
        return x
    }
}

public struct Generator: Layer {
    var mapping = MappingModule()
    
    var firstBlock: GFirstBlock
    
    var blocks: [GBlock] = []
    
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
    
    public var upsample = UpSampling2D<Float>(size: 2)
    
    @noDerivative public private(set) var level: Int = 1
    @noDerivative public var alpha: Float = 1
    
    @noDerivative let wAverageBeta: Float = 0.995
    @noDerivative var wAverage: Parameter<Float>
    
    public init() {
        firstBlock = .init(startSize: 256, outputSize: 256)
        wAverage = Parameter(Tensor(zeros: [1, Config.wsize]))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let z1 = input
        let w1 = mapping(z1)
        
        let training = Context.local.learningPhase == .training
        
        if training {
            wAverage.value = lerp(w1.mean(alongAxes: 0), wAverage.value, rate: wAverageBeta)
        }
        
        var x = firstBlock(.init(w1: w1, w2: w1))
        
        guard level > 1 else {
            // 常にalpha = 1
            return toRGB2(x)
        }
        
        for lv in 0..<level-2 {
            x = blocks[lv](.init(x: x, w1: w1, w2: w1))
        }
        
        var x1 = x
        x1 = toRGB1(x1)
        x1 = upsample(x1)
        
        var x2 = blocks[level-2](.init(x: x, w1: w1, w2: w1))
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
        
        blocks.append(GBlock(inputSize: io.0, outputSize: io.1))
        toRGB1 = toRGB2
        toRGB2 = EqualizedConv2D(inputChannels: io.1,
                                 outputChannels: 3,
                                 kernelSize: (1, 1),
                                 activation: identity,
                                 gain: 1)
    }
}
