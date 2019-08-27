import Foundation
import TensorFlow

public struct Generator: Layer {
    public var mapping = MappingModule()
    public var synthesis = SynthesisModule()
    
    public var alpha: Float {
        get {
            synthesis.alpha
        }
        set {
            synthesis.alpha = newValue
        }
    }
    
    public init() {
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let w = mapping(input)
        return synthesis(w)
    }
    
    public mutating func grow() {
        synthesis.grow()
    }
    
    public func getHistogramWeights() -> [String: Tensor<Float>] {
        let map = mapping.getHistogramWeights()
        let syn = synthesis.getHistogramWeights()
        
        return map.merging(syn) { a, b in a }
    }
}
