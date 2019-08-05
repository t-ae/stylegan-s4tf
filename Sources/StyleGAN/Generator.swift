import Foundation
import TensorFlow

public struct Generator: Layer {
    public var mapping = MappingModule()
    public var synthesis = SynthesisModule()
    
    @noDerivative let wAverageBeta: Float = 0.995
    @noDerivative var wAverage: Parameter<Float>
    
    public var level: Int {
        synthesis.level
    }
    
    public var alpha: Float {
        get {
            synthesis.alpha
        }
        set {
            synthesis.alpha = newValue
        }
    }
    
    public init() {
        wAverage = Parameter(Tensor(zeros: [1, Config.wsize]))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let w = mapping(input)
        
        let ws = createWs(w: w)
        
        return synthesis(ws)
    }
    
    @differentiable
    func createWs(w: Tensor<Float>) -> Tensor<Float> {
        // FIXME: Accessing computed property is not differentiable?
        let level = synthesis.level
        
        let batchSize = w.shape[0]
        let training = Context.local.learningPhase == .training
        let cutoffRange = training.and(Float.random(in: 0..<1) < 0.9)
            ? 1...level*2 : level*2...level*2
        
        // Update wAverage
        if training {
            wAverage.value = lerp(w.mean(alongAxes: 0), wAverage.value, rate: wAverageBeta)
        }
        
        // Style mixing
        let z2 = sampleNoise(size: batchSize)
        let w2 = mapping(z2)
        
        var mask = Tensor<Int32>(zeros: [level*2, batchSize, 1])
        for batch in 0..<batchSize {
            let cutoff = Int.random(in: cutoffRange)
            let size = level*2 - cutoff
            mask[cutoff..., batch] = Tensor(ones: [size, 1])
        }
        mask = mask.reshaped(to: [level, 2, batchSize, 1])
        // [level, 2, batchSize, wsize]
        let mixed = Tensor<Float>(1 - mask) * w + Tensor<Float>(mask) * w2
        
        // truncation trick
        if !training {
            // TODO: Not implemented yet
            // let psi: Float = 0.7
        }
        
        return mixed
    }
    
    public mutating func grow() {
        synthesis.grow()
    }
}
