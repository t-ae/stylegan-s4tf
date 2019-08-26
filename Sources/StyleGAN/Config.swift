import Foundation

public enum Config {
    // MARK: Model settings
    public static let latentSize = 256
    public static let wsize = 256
    
    // Level 1 generates 4x4 images.
    // Level 7 generates 256x256 images.
    public static let startLevel = 2
    public static let maxLevel = 7
    
    public static let useFusedScale = true
    public static let useBlur = false
    public static let useNoise = false
    public static let normalizeLatent = true
    
    public static let loss: LossType = .lsgan
    
    // MARK: Training settings
    public static let mappingLearningRate: Float = 1e-5
    public static let synthesisLearningRate: Float = 1e-3
    public static let discriminatorLearningRate: Float = 1e-3
    
    // minibatch size for each level
    public static let minibatchSizeSchedule = [128, 64, 64, 32, 32, 16, 16]
    
    public static let numImagesPerPhase = 800_000
    
    public static let imageDirectory = URL(fileURLWithPath: "./images")
    public static let tensorboardOutputDirectory = URL(fileURLWithPath: "./tensorboard")
    
    public static let numStepsToInfer = 3_000
    
    public static let debugPrint = true
}
