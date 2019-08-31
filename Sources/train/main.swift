import Foundation
import Python
import TensorFlow
import TensorBoardX
import StyleGAN

// Plot
let writer = SummaryWriter(logdir: Config.tensorboardOutputDirectory, flushSecs: 10)

func plotImage(tag: String, images: Tensor<Float>, rows: Int, cols: Int, step: Int) {
    var images = images.padded(forSizes: [(0, 0), (1, 1), (1, 1), (0, 0)], with: 0)
    let (height, width) = (images.shape[1], images.shape[2])
    images = images.reshaped(to: [rows, cols, height, width, 3])
    images = images.transposed(withPermutations: [0, 2, 1, 3, 4])
    images = images.reshaped(to: [rows*height, cols*width, 3])
    
    // [0, 1] range
    images = (images + 1) / 2
    images = images.clipped(min: Tensor(0), max: Tensor(1))
    
    writer.addImage(tag: tag, image: images, globalStep: step)
}

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
print("image count: \(imageLoader.urls.count)")

let loss = Config.loss.createLoss()

func train(minibatch: Tensor<Float>, step: Int) -> (lossG: Tensor<Float>, lossD: Tensor<Float>){
    Context.local.learningPhase = .training
    let minibatchSize = minibatch.shape[0]
    
    // Differentiate generator
    let noise = sampleNoise(size: minibatchSize)
    let (lossG, 𝛁generator) = generator.valueWithGradient { generator ->Tensor<Float> in
        let images = generator(noise)
        let scores = discriminator(images)
        
        // update output mean
        discriminator.outputMean.value = 0.9*discriminator.outputMean.value
            + 0.1*withoutDerivative(at: scores).mean()
        
        return loss.generatorLoss(fake: scores)
    }
    
    // Differentiate discriminator
    let fakeImages = generator(noise)
    let (lossD, 𝛁discriminator) = discriminator.valueWithGradient { discriminator -> Tensor<Float> in
        let realScores = discriminator(minibatch)
        let fakeScores = discriminator(fakeImages)
        
        // update output mean
        discriminator.outputMean.value = 0.9*discriminator.outputMean.value + 0.1*fakeScores.mean()
        
        return loss.discriminatorLoss(real: realScores, fake: fakeScores)
    }
    
    if lossG.scalarized() > 10 || lossD.scalarized() > 10 {
        // Occasionally gets too large loss in early steps.
        // Don't train then.
        print("skip training")
        // record large loss images for debugging
        plotImage(tag: "Large_loss/real", images: minibatch, rows: 4, cols: minibatchSize/4, step: step)
        plotImage(tag: "Large_loss/fake", images: fakeImages, rows: 4, cols: minibatchSize/4, step: step)
    } else {
        // Update
        optMap.update(&generator.mapping, along: 𝛁generator.mapping)
        optSynth.update(&generator.synthesis, along: 𝛁generator.synthesis)
        optDis.update(&discriminator, along: 𝛁discriminator)
    }
    
    return (lossG, lossD)
}

// Test
let testNoise = sampleNoise(size: 64)
func infer(level: Int, step: Int) {
    print("infer...")
    Context.local.learningPhase = .inference
    
    let images = generator(testNoise)
    plotImage(tag: "lv\(level)", images: images, rows: 8, cols: 8, step: step)
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
    
    let level = generator.synthesis.level
    
    let minibatchSize = Config.minibatchSizeSchedule[level - 1]
    let imageSize = 2 * Int(powf(2, Float(level)))

    let minibatch = measureTime(label: "minibatch load") {
        imageLoader.minibatch(size: minibatchSize, imageSize: (imageSize, imageSize))
    }
    
    let (lossG, lossD) = measureTime(label: "train") {
        train(minibatch: minibatch, step: step)
    }

    print("step: \(step), alpha: \(generator.alpha), g: \(lossG), d: \(lossD)")
    
    writer.addScalar(tag: "lv\(level)/lossG", scalar: lossG.scalarized(), globalStep: step)
    writer.addScalar(tag: "lv\(level)/lossD", scalar: lossD.scalarized(), globalStep: step)
    if Config.loss == .lsgan {
        writer.addScalar(tag: "lv\(level)/dout_mean", scalar: discriminator.outputMean.value.scalarized(), globalStep: step)
    }
    
    imageCount += minibatchSize
    
    var shouldInfer = step.isMultiple(of: Config.numStepsToInfer)
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
            shouldInfer = true
        }
    }
    
    if shouldInfer {
        let level = generator.synthesis.level
        infer(level: level, step: step)
        addHistograms(step: step)
    }
}
