import Foundation
import TensorFlow
import Swim

class ImageLoader {
    var imageDirectory: URL
    var fileNames: [String]
    
    var index = 0
    
    var multiThread: Bool
    
    let appendQueue = DispatchQueue(label: "ImageLoader.appendQueue")
    
    init(imageDirectory: URL, multiThread: Bool = true) throws {
        self.imageDirectory = imageDirectory
        fileNames = try FileManager.default.contentsOfDirectory(atPath: imageDirectory.path)
            .filter { $0.hasSuffix(".png") }
        self.multiThread = multiThread
    }
    
    func shuffle() {
        fileNames.shuffle()
    }
    
    func resetIndex() {
        index = 0
    }
    
    func minibatch(size: Int, imageSize: (height: Int, width: Int)) -> Tensor<Float> {
        if fileNames.count >= index+size {
            resetIndex()
            shuffle()
        }
        
        var tensors: [Tensor<Float>]
        let fileNames = self.fileNames[index..<index+size]
        
        if multiThread {
            tensors = []
            DispatchQueue.concurrentPerform(iterations: size) { i in
                let url = imageDirectory.appendingPathComponent(fileNames[i])
                let image = try! Image<RGB, Float>(contentsOf: url)
                let resized = image.resize(width: imageSize.width, height: imageSize.height)
                
                let tensor = Tensor<Float>(resized.getData())
                appendQueue.sync {
                    tensors.append(tensor)
                }
            }
        } else {
            let images = fileNames.map { fileName -> Image<RGB, Float> in
                let url = imageDirectory.appendingPathComponent(fileName)
                let image = try! Image<RGB, Float>(contentsOf: url)
                return image.resize(width: imageSize.width, height: imageSize.height)
            }
            tensors = images.map { image in
                image.withUnsafeBufferPointer { bp in
                    Tensor<Float>(Array(bp))
                }
            }
        }
        
        let tensor = Tensor<Float>(stacking: tensors)
        
        // [-1, 1] range
        return tensor.reshaped(to: [size, imageSize.height, imageSize.width, 3]) * 2 - 1
    }
}
