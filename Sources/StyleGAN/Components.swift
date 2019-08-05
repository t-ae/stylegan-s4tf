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

@differentiable
public func minibatchStdConcat(_ x: Tensor<Float>) -> Tensor<Float> {
    let groupSize = 4
    let batchSize = x.shape[0]
    let height = x.shape[1]
    let width = x.shape[2]
    let channels = x.shape[3]
    let M = batchSize / groupSize
    
    // Compute stddev of each pixel in group
    var y = x.reshaped(to: [groupSize, M, height, width, channels])
    let mean = y.mean(alongAxes: 0) // [1, M, height, width, channels]
    y = (y - mean).squared().mean(squeezingAxes: 0) // [M, height, width, channels]
    // y = sqrt(y + 1e-8) // stddev
    
    y = y.mean(alongAxes: 1, 2, 3) // [M, 1, 1, 1]
    y = y.tiled(multiples: Tensor([Int32(groupSize), Int32(height), Int32(width), 1]))
    y = y.reshaped(to: [batchSize, height, width, 1])
    
    // https://bugs.swift.org/browse/TF-705
    //return x.concatenated(with: y, alongAxis: 3)
    
    // https://bugs.swift.org/browse/TF-706
    // return Tensor(concatenating: [x, y], alongAxis: 3)
    
    // Dirty hack to avoid the bugs above
    let xs = x.unstacked(alongAxis: 3)
    y = y.squeezingShape(at: 3)
    return Tensor(stacking: xs + [y], alongAxis: 3)
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
