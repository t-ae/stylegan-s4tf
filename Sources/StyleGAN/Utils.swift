import Foundation
import TensorFlow

public func sampleNoise(size: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, Config.latentSize])
}

// FIXME: && and || can't be used in @differentiable functions directly.
extension Bool {
    func and(_ other: Bool) -> Bool {
        self && other
    }
    
    func or(_ other: Bool) -> Bool {
        self || other
    }
}
