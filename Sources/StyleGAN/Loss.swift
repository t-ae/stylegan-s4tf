import Foundation
import TensorFlow

public protocol Loss {
    @differentiable
    func generatorLoss(fake: Tensor<Float>) -> Tensor<Float>
    
    @differentiable
    func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float>
}

public struct NonSaturatingLoss: Loss {
    public init() {}
    
    @differentiable
    public func generatorLoss(fake: Tensor<Float>) -> Tensor<Float> {
        softplus(-fake).mean()
    }
    
    @differentiable
    public func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        let realLoss = softplus(-real).mean()
        let fakeLoss = softplus(fake).mean()
        
        return realLoss + fakeLoss
    }
}

public struct LSGANLoss: Loss {
    public init() {}
    
    @differentiable
    public func generatorLoss(fake: Tensor<Float>) -> Tensor<Float> {
        meanSquaredError(predicted: fake, expected: Tensor<Float>(ones: fake.shape)) / 2
    }
    
    @differentiable
    public func discriminatorLoss(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        let realLoss = meanSquaredError(predicted: real, expected: Tensor<Float>(ones: real.shape))
        let fakeLoss = meanSquaredError(predicted: fake, expected: Tensor<Float>(zeros: fake.shape))
        
        return (realLoss + fakeLoss) / 2
    }
}
