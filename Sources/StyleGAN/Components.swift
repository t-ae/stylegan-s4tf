import Foundation
import TensorFlow

@differentiable
func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable(wrt: x)
func pixelNormalization(_ x: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    // 2D or 4D
    let mean = x.squared().mean(alongAxes: x.shape.count-1)
    return x * rsqrt(mean + epsilon)
}

@differentiable(wrt: (a, b))
func lerp(_ a: Tensor<Float>, _ b: Tensor<Float>, rate: Float) -> Tensor<Float> {
    let rate = min(max(rate, 0), 1)
    return a + rate * (b - a)
}

struct Blur3x3: ParameterlessLayer {
    @noDerivative
    let filter: Tensor<Float>
    
    public init(channels: Int) {
        var f = Tensor<Float>([1, 2, 1])
        f = f.reshaped(to: [1, -1]) * f.reshaped(to: [-1, 1])
        f /= f.sum()
        f = f.reshaped(to: [3, 3, 1, 1])
        f = f.tiled(multiples: Tensor([1, 1, Int32(channels), 1]))
        self.filter = f
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        depthwiseConv2D(input, filter: filter, strides: (1, 1, 1, 1), padding: .same)
    }
}

@differentiable
func instanceNorm2D(_ x: Tensor<Float>) -> Tensor<Float> {
    let mean = x.mean(alongAxes: 1, 2)
    let variance = x.variance(alongAxes: 1, 2)
    return (x - mean) * rsqrt(variance + 1e-8)
}

struct AdaIN: Layer {
    struct Input: Differentiable {
        var x: Tensor<Float>
        var w: Tensor<Float>
    }
    
    var scaleTransform: EqualizedDense
    var biasTransform: EqualizedDense
    
    init(size: Int, wsize: Int) {
        scaleTransform = EqualizedDense(inputSize: wsize, outputSize: size, gain: 1)
        biasTransform = EqualizedDense(inputSize: wsize, outputSize: size, gain: 1)
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        let batchSize = input.x.shape[0]
        let x = instanceNorm2D(input.x)
        let scale = scaleTransform(input.w).reshaped(to: [batchSize, 1, 1, -1])
        let bias = biasTransform(input.w).reshaped(to: [batchSize, 1, 1, -1])
        return x * scale + bias
    }
    
    // Directly calling `Input.init` has problem?
    @differentiable
    static func makeInput(x: Tensor<Float>, w: Tensor<Float>) -> Input {
        Input(x: x, w: w)
    }
}

struct NoiseLayer: Layer {
    var noiseScale: Tensor<Float>
    
    init(channels: Int) {
        noiseScale = Tensor(zeros: [1, 1, 1, channels])
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let height = input.shape[1]
        let width = input.shape[2]
        let noise = Tensor<Float>(randomNormal: [1, height, width, 1])
        
        return input + noise * noiseScale
    }
}

struct EqualizedDense: Layer {
    var dense: Dense<Float>
    @noDerivative public let scale: Tensor<Float>
    
    init(inputSize: Int,
         outputSize: Int,
         activation: @escaping Dense<Float>.Activation = identity,
         gain: Float = sqrt(2)) {
        let weight = Tensor<Float>(randomNormal: [inputSize, outputSize])
        let bias = Tensor<Float>(zeros: [outputSize])
        self.dense = Dense(weight: weight, bias: bias, activation: activation)
        
        self.scale = Tensor(gain) / sqrt(Float(inputSize))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        // Scale input instead of dense.weight
        return dense(input * scale)
    }
}

struct EqualizedConv2D: Layer {
    var conv: Conv2D<Float>
    @noDerivative public let scale: Tensor<Float>
    
    init(inputChannels: Int,
         outputChannels: Int,
         kernelSize: (Int, Int),
         strides: (Int, Int) = (1, 1),
         padding: Padding = .same,
         activation: @escaping Conv2D<Float>.Activation = identity,
         gain: Float = sqrt(2)) {
        let filter = Tensor<Float>(randomNormal: [kernelSize.0,
                                                  kernelSize.1,
                                                  inputChannels,
                                                  outputChannels])
        let bias = Tensor<Float>(zeros: [outputChannels])
        
        self.conv = Conv2D(filter: filter,
                           bias: bias,
                           activation: activation,
                           strides: strides,
                           padding: padding)
        
        self.scale = Tensor(gain) / sqrt(Float(inputChannels*kernelSize.0*kernelSize.1))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        // Scale input instead of conv.filter
        return conv(input * scale)
    }
}
