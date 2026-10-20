import Foundation
import ARKit
import Combine
import simd
import RadarCore

enum RadarSourceMode: String, CaseIterable, Identifiable {
    case lidar = "LiDAR"
    case simulation = "Simulation"

    var id: String { rawValue }
}

/// Owns the sensor source (ARKit LiDAR or the synthetic room) and publishes radar snapshots.
@MainActor
final class LiDARRadarEngine: ObservableObject {
    @Published private(set) var snapshot = RadarSnapshot()
    @Published private(set) var mode: RadarSourceMode
    @Published private(set) var statusMessage: String?
    @Published var settings = RadarSettings() {
        didSet {
            guard settings != oldValue else { return }
            pipeline.update(settings: settings)
            receiver.visionEnabled = settings.visionEnabled
            simulator.pixelStep = settings.pointStep
        }
    }

    let isLiDARAvailable = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    private let session = ARSession()
    private let pipeline = RadarPipeline()
    private let receiver: ARFrameReceiver
    private let simulator: SimulationDriver
    private var running = false

    init() {
        let forceSimulation = ProcessInfo.processInfo.arguments.contains("-simulate")
        mode = (isLiDARAvailable && !forceSimulation) ? .lidar : .simulation
        receiver = ARFrameReceiver(pipeline: pipeline)
        simulator = SimulationDriver(pipeline: pipeline)
        session.delegate = receiver
        session.delegateQueue = DispatchQueue(label: "radar.arsession", qos: .userInteractive)
        pipeline.update(settings: settings)
        pipeline.onSnapshot = { [weak self] snap in
            MainActor.assumeIsolated { self?.snapshot = snap }
        }
        receiver.onStatus = { [weak self] message in
            DispatchQueue.main.async { self?.statusMessage = message }
        }
        if !isLiDARAvailable {
            statusMessage = "No LiDAR scanner detected — running the synthetic room demo."
        }
    }

    func start() {
        guard !running else { return }
        running = true
        switch mode {
        case .lidar:
            let config = ARWorldTrackingConfiguration()
            config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
            config.environmentTexturing = .none
            config.worldAlignment = .gravity
            session.run(config, options: [.resetTracking, .removeExistingAnchors])
        case .simulation:
            simulator.pixelStep = settings.pointStep
            simulator.start()
        }
    }

    func stop() {
        guard running else { return }
        running = false
        session.pause()
        simulator.stop()
    }

    func setMode(_ newMode: RadarSourceMode) {
        guard newMode != mode, newMode == .simulation || isLiDARAvailable else { return }
        stop()
        mode = newMode
        resetTracking()
        start()
    }

    func resetTracking() {
        pipeline.reset()
        if mode == .lidar && running {
            session.run(session.configuration ?? ARWorldTrackingConfiguration(), options: [.resetTracking, .removeExistingAnchors])
        }
    }
}

/// ARSession delegate living off the main actor. Copies depth out of each ARFrame so ARKit
/// can recycle frames immediately, then hands work to the pipeline / Vision queues.
final class ARFrameReceiver: NSObject, ARSessionDelegate {
    var onStatus: ((String?) -> Void)?
    var visionEnabled: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _visionEnabled }
        set { lock.lock(); _visionEnabled = newValue; lock.unlock() }
    }

    private let lock = NSLock()
    private var _visionEnabled = true

    private let pipeline: RadarPipeline
    private let vision = VisionFusion()
    private var frameIndex = 0
    private var lastTrackingDescription: String?

    init(pipeline: RadarPipeline) {
        self.pipeline = pipeline
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameIndex += 1
        let wantsVision = visionEnabled && frameIndex % 3 == 0 && !vision.isBusy
        guard !pipeline.isBusy || wantsVision else { return }
        guard let depth = Self.makeDepthFrame(from: frame) else { return }
        pipeline.submit(depth)
        if wantsVision {
            let pipeline = pipeline
            vision.submit(pixelBuffer: frame.capturedImage, depth: depth) { measurements in
                pipeline.submitVision(measurements, timestamp: depth.timestamp, observer: depth.cameraPosition)
            }
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let text: String?
        switch camera.trackingState {
        case .normal: text = nil
        case .notAvailable: text = "Tracking unavailable"
        case .limited(.initializing): text = "Initializing — move the device slowly"
        case .limited(.excessiveMotion): text = "Slow down — excessive motion"
        case .limited(.insufficientFeatures): text = "Low detail — point at a textured area"
        case .limited(.relocalizing): text = "Relocalizing…"
        case .limited: text = "Tracking limited"
        }
        if text != lastTrackingDescription {
            lastTrackingDescription = text
            onStatus?(text)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        onStatus?("AR session failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        onStatus?("Session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        onStatus?(nil)
    }

    static func makeDepthFrame(from frame: ARFrame) -> DepthFrame? {
        guard let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
        let depthMap = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        let w = CVPixelBufferGetWidth(depthMap), h = CVPixelBufferGetHeight(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        var depth = [Float](repeating: 0, count: w * h)
        depth.withUnsafeMutableBytes { dst in
            for y in 0..<h {
                memcpy(dst.baseAddress! + y * w * 4, base + y * rowBytes, w * 4)
            }
        }

        var confidence: [UInt8]?
        if let conf = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            if let cbase = CVPixelBufferGetBaseAddress(conf), CVPixelBufferGetWidth(conf) == w, CVPixelBufferGetHeight(conf) == h {
                let crow = CVPixelBufferGetBytesPerRow(conf)
                var c = [UInt8](repeating: 0, count: w * h)
                c.withUnsafeMutableBytes { dst in
                    for y in 0..<h { memcpy(dst.baseAddress! + y * w, cbase + y * crow, w) }
                }
                confidence = c
            }
            CVPixelBufferUnlockBaseAddress(conf, .readOnly)
        }

        let res = frame.camera.imageResolution
        let k = DepthIntrinsics(cameraIntrinsics: frame.camera.intrinsics, imageWidth: Float(res.width), imageHeight: Float(res.height), depthWidth: w, depthHeight: h)
        return DepthFrame(width: w, height: h, depth: depth, confidence: confidence, intrinsics: k, cameraTransform: frame.camera.transform, timestamp: frame.timestamp)
    }
}

/// Drives the pipeline with ray-cast depth frames from `SyntheticScene` at 30 Hz.
final class SimulationDriver {
    private let pipeline: RadarPipeline
    private let queue = DispatchQueue(label: "radar.simulation", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private let scene = SyntheticScene()
    private var startTime: CFTimeInterval = 0

    init(pipeline: RadarPipeline) {
        self.pipeline = pipeline
    }

    var pixelStep: Int {
        get { lock.lock(); defer { lock.unlock() }; return _pixelStep }
        set { lock.lock(); _pixelStep = max(1, newValue); lock.unlock() }
    }

    private let lock = NSLock()
    private var _pixelStep = 3

    func start() {
        stop()
        startTime = CACurrentMediaTime()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: .milliseconds(33))
        t.setEventHandler { [weak self] in
            guard let self, !self.pipeline.isBusy else { return }
            let time = CACurrentMediaTime() - self.startTime
            let frame = self.scene.renderDepthFrame(at: time, pixelStep: self.pixelStep, seed: 7)
            self.pipeline.submit(frame)
        }
        timer = t
        t.resume()
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }
}
