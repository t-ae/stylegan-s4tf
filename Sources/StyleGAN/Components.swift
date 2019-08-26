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
    let moments = x.moments(alongAxes: 1, 2)
    return (x - moments.mean) * rsqrt(moments.variance + 1e-8)
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

public struct EqualizedDense: Layer {
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    
    public var weight: Tensor<Float>
    public var bias: Tensor<Float>
    
    @noDerivative public let scale: Float
    
    @noDerivative public let activation: Activation
    
    public init(inputSize: Int,
                outputSize: Int,
                activation: @escaping Activation = identity,
                gain: Float = sqrt(2)) {
        self.weight = Tensor<Float>(randomNormal: [inputSize, outputSize])
        self.bias = Tensor<Float>(zeros: [outputSize])
        
        self.scale = gain / sqrt(Float(inputSize))
        
        self.activation = activation
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        activation(matmul(input, weight * scale) + bias)
    }
}

public struct EqualizedConv2D: Layer {
    public typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    
    public var filter: Tensor<Float>
    public var bias: Tensor<Float>
    @noDerivative public let scale: Float
    
    @noDerivative public let strides: (Int, Int)
    @noDerivative public let padding: Padding
    
    @noDerivative public let activation: Activation
    
    public init(inputChannels: Int,
                outputChannels: Int,
                kernelSize: (Int, Int),
                strides: (Int, Int) = (1, 1),
                padding: Padding = .same,
                activation: @escaping Activation = identity,
                gain: Float = sqrt(2)) {
        self.filter = Tensor<Float>(randomNormal: [kernelSize.0,
                                                   kernelSize.1,
                                                   inputChannels,
                                                   outputChannels])
        self.bias = Tensor<Float>(zeros: [outputChannels])
        
        self.scale = gain / sqrt(Float(inputChannels*kernelSize.0*kernelSize.1))
        
        self.strides = strides
        self.padding = padding
        
        self.activation = activation
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        activation(conv2D(input,
                          filter: filter * scale,
                          strides: (1, strides.0, strides.1, 1),
                          padding: padding,
                          dilations: (1, 1, 1, 1)) + bias)
    }
}
