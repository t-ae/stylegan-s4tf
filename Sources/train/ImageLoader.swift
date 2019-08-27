import Foundation
import TensorFlow
import Swim

class ImageLoader {
    var imageDirectory: URL
    var urls: [URL]
    
    var index = 0
    
    var multiThread: Bool
    
    let appendQueue = DispatchQueue(label: "ImageLoader.appendQueue")
    
    init(imageDirectory: URL, multiThread: Bool = true) throws {
        self.imageDirectory = imageDirectory
        
        let enumerator = FileManager.default.enumerator(at: imageDirectory, includingPropertiesForKeys: nil)!
        var urls: [URL] = []
        for entry in enumerator {
            guard let url = entry as? URL else {
                continue
            }
            guard url.pathExtension == "png" else {
                continue
            }
            urls.append(url)
        }
        self.urls = urls
        
        self.multiThread = multiThread
    }
    
    func shuffle() {
        urls.shuffle()
    }
    
    func resetIndex() {
        index = 0
    }
    
    func minibatch(size: Int, imageSize: (height: Int, width: Int)) -> Tensor<Float> {
        if urls.count < index+size {
            resetIndex()
            shuffle()
        }
        
        var tensors: [Tensor<Float>]
        let urls = self.urls[index..<index+size]
        defer { index += size }
        
        if multiThread {
            tensors = []
            DispatchQueue.concurrentPerform(iterations: size) { i in
                let url = urls[i+index]
                let image = try! Image<RGB, Float>(contentsOf: url)
                let resized = image.resize(width: imageSize.width, height: imageSize.height)
                
                let tensor = Tensor<Float>(resized.getData())
                appendQueue.sync {
                    tensors.append(tensor)
                }
            }
        } else {
            let images = urls.map { url -> Image<RGB, Float> in
                let image = try! Image<RGB, Float>(contentsOf: url)
                return image.resize(width: imageSize.width, height: imageSize.height, method: .areaAverage)
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
