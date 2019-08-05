import Foundation
import TensorFlow

public struct MappingModule: Layer {
    var dense1: EqualizedDense
    var dense2: EqualizedDense
    var dense3: EqualizedDense
    var dense4: EqualizedDense
    var dense5: EqualizedDense
    var dense6: EqualizedDense
    
    public init() {
        let zsize = Config.latentSize
        let wsize = Config.wsize
        dense1 = EqualizedDense(inputSize: zsize, outputSize: zsize, activation: lrelu)
        dense2 = EqualizedDense(inputSize: zsize, outputSize: zsize, activation: lrelu)
        dense3 = EqualizedDense(inputSize: zsize, outputSize: zsize, activation: lrelu)
        dense4 = EqualizedDense(inputSize: zsize, outputSize: zsize, activation: lrelu)
        dense5 = EqualizedDense(inputSize: zsize, outputSize: zsize, activation: lrelu)
        dense6 = EqualizedDense(inputSize: zsize, outputSize: wsize, activation: lrelu)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        if Config.normalizeLatent {
            x = pixelNormalization(x)
        }
        return x.sequenced(through: dense1, dense2, dense3, dense4, dense5, dense6)
    }
}
