import Foundation
import StyleGAN

func debugPrint(_ text: String) {
    guard Config.debugPrint else {
        return
    }
    print(text)
}

func measureTime<R>(label: String, f: ()->R) -> R {
    let start = Date()
    defer { debugPrint("\(label): \(Date().timeIntervalSince(start))sec") }
    
    return f()
}
