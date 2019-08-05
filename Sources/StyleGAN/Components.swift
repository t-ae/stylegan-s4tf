import Foundation
import TensorFlow

@differentiable
public func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable(wrt: x)
public func pixelNormalization(_ x: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    let x2 = x * x
    let mean = x2.mean(alongAxes: 3)
    return x * rsqrt(mean + epsilon)
}

@differentiable(wrt: (a, b))
public func lerp(_ a: Tensor<Float>, _ b: Tensor<Float>, rate: Float) -> Tensor<Float> {
    let rate = min(max(rate, 0), 1)
    return a + rate * (b - a)
}

public struct Blur3x3: ParameterlessLayer {
    @noDerivative
    let filter: Tensor<Float>
    
    public init() {
        var f = Tensor<Float>([1, 2, 1])
        f = f.reshaped(to: [1, -1]) * f.reshaped(to: [-1, 1])
        f /= f.sum()
        f = f.reshaped(to: [3, 3, 1, 1])
        self.filter = f
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        depthwiseConv2D(input, filter: filter, strides: (1, 1, 1, 1), padding: .same)
    }
}

@differentiable
public func instanceNorm2D(_ x: Tensor<Float>) -> Tensor<Float> {
    let mean = x.mean(alongAxes: 1, 2)
    let std = x.standardDeviation(alongAxes: 1, 2)
    return (x - mean) / std
}

public struct AdaIN: Layer {
    public struct Input: Differentiable {
        var x: Tensor<Float>
        var w: Tensor<Float> //
    }
    public var scaleTransform: EqualizedDense
    public var biasTransform: EqualizedDense
    
    public init(wsize: Int, size: Int) {
        scaleTransform = EqualizedDense(inputSize: wsize, outputSize: size, gain: 1)
        biasTransform = EqualizedDense(inputSize: wsize, outputSize: size, gain: 1)
    }
    
    @differentiable
    public func callAsFunction(_ input: AdaIN.Input) -> Tensor<Float> {
        let batchSize = input.x.shape[0]
        let x = instanceNorm2D(input.x)
        let scale = scaleTransform(input.w).reshaped(to: [batchSize, 1, 1, -1])
        let bias = biasTransform(input.w).reshaped(to: [batchSize, 1, 1, -1])
        return x * scale + bias
    }
}

public struct NoiseLayer: Layer {
    public var noiseScale: Tensor<Float>
    
    public init(channels: Int) {
        noiseScale = Tensor(zeros: [1, 1, 1, channels])
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let height = input.shape[1]
        let width = input.shape[2]
        let noise = Tensor<Float>(randomNormal: [1, height, width, 1])
        
        return input + noise * noiseScale
    }
}

public struct EqualizedDense: Layer {
    public var dense: Dense<Float>
    @noDerivative public let scale: Tensor<Float>
    
    public init(inputSize: Int,
                outputSize: Int,
                activation: @escaping Dense<Float>.Activation = identity,
                gain: Float = sqrt(2)) {
        let weight = Tensor<Float>(randomNormal: [inputSize, outputSize])
        let bias = Tensor<Float>(zeros: [outputSize])
        self.dense = Dense(weight: weight, bias: bias, activation: activation)
        
        self.scale = Tensor(gain) / sqrt(Float(inputSize))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        // Scale input instead of dense.weight
        return dense(input * scale)
    }
}

public struct EqualizedConv2D: Layer {
    public var conv: Conv2D<Float>
    @noDerivative public let scale: Tensor<Float>
    
    public init(inputChannels: Int,
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
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        // Scale input instead of conv.filter
        return conv(input * scale)
    }
}

public struct EqualizedTransposedConv2D: Layer {
    public var conv: TransposedConv2D<Float>
    @noDerivative public let scale: Tensor<Float>
    
    public init(inputChannels: Int,
                outputChannels: Int,
                kernelSize: (Int, Int),
                strides: (Int, Int) = (1, 1),
                padding: Padding = .same,
                activation: @escaping TransposedConv2D<Float>.Activation = identity,
                gain: Float = sqrt(2)) {
        let filter = Tensor<Float>(randomNormal: [kernelSize.0,
                                                  kernelSize.1,
                                                  outputChannels,
                                                  inputChannels])
        let bias = Tensor<Float>(zeros: [outputChannels])
        
        self.conv = TransposedConv2D(filter: filter,
                                     bias: bias,
                                     activation: activation,
                                     strides: strides,
                                     padding: padding)
        
        self.scale = Tensor(gain) / sqrt(Float(inputChannels*kernelSize.0*kernelSize.1))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        // Scale input instead of conv.filter
        return conv(input * scale)
    }
}
