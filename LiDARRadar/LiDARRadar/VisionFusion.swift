import Foundation
import ImageIO
import Vision
import simd
import RadarCore

/// Runs Vision people / animal detection on the RGB frame and back-projects each bounding
/// box into world space using the LiDAR depth under the box.
final class VisionFusion {
    private let queue = DispatchQueue(label: "radar.vision", qos: .userInteractive)
    private let lock = NSLock()
    private var busy = false

    var isBusy: Bool {
        lock.lock(); defer { lock.unlock() }
        return busy
    }

    func submit(pixelBuffer: CVPixelBuffer, depth: DepthFrame, completion: @escaping ([TargetMeasurement]) -> Void) {
        lock.lock()
        if busy { lock.unlock(); return }
        busy = true
        lock.unlock()
        queue.async {
            let result = Self.detect(pixelBuffer: pixelBuffer, depth: depth)
            self.lock.lock(); self.busy = false; self.lock.unlock()
            completion(result)
        }
    }

    private static func detect(pixelBuffer: CVPixelBuffer, depth: DepthFrame) -> [TargetMeasurement] {
        let orientation = SensorImageOrientation.from(cameraTransform: depth.cameraTransform)
        let animalRequest = VNRecognizeAnimalsRequest()
        let humanRequest = VNDetectHumanRectanglesRequest()
        humanRequest.upperBodyOnly = false

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation.cgOrientation, options: [:])
        do {
            try handler.perform([animalRequest, humanRequest])
        } catch {
            return []
        }

        var detections: [(label: String, box: CGRect)] = []
        for animal in animalRequest.results ?? [] {
            detections.append((animal.labels.first?.identifier ?? "Animal", animal.boundingBox))
        }
        for human in humanRequest.results ?? [] where human.confidence > 0.5 {
            detections.append(("Person", human.boundingBox))
        }

        let w = Float(depth.width), h = Float(depth.height)
        return detections.compactMap { det in
            let r = orientation.sensorRect(fromVision: Float(det.box.minX), minY: Float(det.box.minY), width: Float(det.box.width), height: Float(det.box.height))
            let px = (minX: r.minX * w, minY: r.minY * h, maxX: r.maxX * w, maxY: r.maxY * h)
            guard let z = PointCloudBuilder.robustDepth(in: depth, rect: px) else { return nil }
            let u = (px.minX + px.maxX) / 2, v = (px.minY + px.maxY) / 2
            let center = Unprojection.world(u: u, v: v, depth: z, intrinsics: depth.intrinsics, cameraTransform: depth.cameraTransform)
            let widthMeters = (px.maxX - px.minX) * z / depth.intrinsics.fx
            let heightMeters = (px.maxY - px.minY) * z / depth.intrinsics.fy
            let size = SIMD3<Float>(widthMeters, heightMeters, min(widthMeters, heightMeters))
            return TargetMeasurement(position: center, label: det.label, size: size, source: .vision)
        }
    }
}

extension SensorImageOrientation {
    var cgOrientation: CGImagePropertyOrientation {
        switch self {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        }
    }
}
