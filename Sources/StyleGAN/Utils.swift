import Foundation
import TensorFlow

public func sampleNoise(size: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, Config.latentSize])
}
