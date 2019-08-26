import Foundation
import Python
import TensorFlow
import TensorBoardX
import StyleGAN

var generator = Generator()
var discriminator = Discriminator()

var optMap = Adam(for: generator.mapping, learningRate: Config.mappingLearningRate, beta1: 0, beta2: 0.99)
var optSynth = Adam(for: generator.synthesis, learningRate: Config.synthesisLearningRate, beta1: 0, beta2: 0.99)
var optDis = Adam(for: discriminator, learningRate: Config.discriminatorLearningRate, beta1: 0, beta2: 0.99)

func grow() {
    generator.grow()
    discriminator.grow()
    
    optMap = Adam(for: generator.mapping, learningRate: Config.mappingLearningRate, beta1: 0, beta2: 0.99)
    optSynth = Adam(for: generator.synthesis, learningRate: Config.synthesisLearningRate, beta1: 0, beta2: 0.99)
    optDis = Adam(for: discriminator, learningRate: Config.discriminatorLearningRate, beta1: 0, beta2: 0.99)
}

func setAlpha(_ alpha: Float) {
    generator.alpha = alpha
    discriminator.alpha = alpha
}

let imageLoader = try ImageLoader(imageDirectory: Config.imageDirectory)

let loss = Config.loss.createLoss()

func train(minibatch: Tensor<Float>) -> (lossG: Tensor<Float>, lossD: Tensor<Float>){
    Context.local.learningPhase = .training
    let minibatchSize = minibatch.shape[0]
    // Update generator
    let noise1 = sampleNoise(size: minibatchSize)
    
    let (lossG, 𝛁generator) = generator.valueWithGradient { generator ->Tensor<Float> in
        let images = generator(noise1)
        let scores = discriminator(images)
        
        // update output mean
        discriminator.outputMean.value = 0.9*discriminator.outputMean.value
            + 0.1*withoutDerivative(at: scores).mean()
        
        return loss.generatorLoss(fake: scores)
    }
    optMap.update(&generator.mapping, along: 𝛁generator.mapping)
    optSynth.update(&generator.synthesis, along: 𝛁generator.synthesis)
    
    
    // Update discriminator
    let noise2 = sampleNoise(size: minibatchSize)
    let fakeImages = generator(noise2)
    let (lossD, 𝛁discriminator) = discriminator.valueWithGradient { discriminator -> Tensor<Float> in
        let realScores = discriminator(minibatch)
        let fakeScores = discriminator(fakeImages)
        
        // update output mean
        discriminator.outputMean.value = 0.9*discriminator.outputMean.value + 0.1*fakeScores.mean()
        
        return loss.discriminatorLoss(real: realScores, fake: fakeScores)
    }
    optDis.update(&discriminator, along: 𝛁discriminator)

    return (lossG, lossD)
}

// Plot
let writer = SummaryWriter(logdir: Config.tensorboardOutputDirectory, flushSecs: 10)

// Test
let testNoise = sampleNoise(size: 64)
func infer(level: Int, step: Int) {
    print("infer...")
    Context.local.learningPhase = .inference
    
    var images = generator(testNoise)
    images = images.padded(forSizes: [(0, 0), (1, 1), (1, 1), (0, 0)], with: 0)
    let (height, width) = (images.shape[1], images.shape[2])
    images = images.reshaped(to: [8, 8, height, width, 3])
    images = images.transposed(withPermutations: [0, 2, 1, 3, 4])
    images = images.reshaped(to: [8*height, 8*width, 3])
    
    // [0, 1] range
    images = (images + 1) / 2
    images = images.clipped(min: Tensor(0), max: Tensor(1))
    
    writer.addImage(tag: "lv\(level)", image: images, globalStep: step)
}

func addHistograms(step: Int) {
    for (k, v) in generator.getHistogramWeights() {
        writer.addHistogram(tag: k, values: v, globalStep: step)
    }
    for (k, v) in discriminator.getHistogramWeights() {
        writer.addHistogram(tag: k, values: v, globalStep: step)
    }
}

enum Phase {
    case fading, stabilizing
}

var phase: Phase = .stabilizing
var imageCount = 0

// Grow to start level
for _ in 1..<Config.startLevel {
    grow()
}

// Initial histogram
addHistograms(step: 0)

for step in 1... {
    if phase == .fading {
        setAlpha(Float(imageCount) / Float(Config.numImagesPerPhase))
    }
    print("step: \(step), alpha: \(generator.alpha)")
    
    let level = generator.synthesis.level
    
    let minibatchSize = Config.minibatchSizeSchedule[level - 1]
    let imageSize = 2 * Int(powf(2, Float(level)))

    let minibatch = measureTime(label: "minibatch load") {
        imageLoader.minibatch(size: minibatchSize, imageSize: (imageSize, imageSize))
    }
    
    let (lossG, lossD) = measureTime(label: "train") {
        train(minibatch: minibatch)
    }
    
    writer.addScalar(tag: "lv\(level)/lossG", scalar: lossG.scalar!, globalStep: step)
    writer.addScalar(tag: "lv\(level)/lossD", scalar: lossD.scalar!, globalStep: step)
    if Config.loss == .lsgan {
        writer.addScalar(tag: "lv\(level)/dout_mean", scalar: discriminator.outputMean.value.scalar!, globalStep: step)
    }
    
    imageCount += minibatchSize
    
    if imageCount >= Config.numImagesPerPhase {
        imageCount = 0
        
        switch (phase, level) {
        case (.fading, _):
            phase = .stabilizing
            setAlpha(1)
            print("Start stabilizing lv: \(generator.synthesis.level)")
        case (.stabilizing, Config.maxLevel):
            break
        case (.stabilizing, _):
            phase = .fading
            setAlpha(0)
            grow()
            print("Start fading lv: \(generator.synthesis.level)")
            
            infer(level: level, step: step)
            addHistograms(step: step)
        }
    }
    
    if step.isMultiple(of: Config.numStepsToInfer) {
        infer(level: level, step: step)
        addHistograms(step: step)
    }
}
