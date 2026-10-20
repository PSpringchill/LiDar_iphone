import Foundation
import QuartzCore
import simd
import RadarCore

/// Everything the UI needs to draw one radar frame.
struct RadarSnapshot {
    var livePoints: [PointCloudVertex] = []
    var mapPoints: [PointCloudVertex] = []
    var mapVersion = 0
    var targets: [TrackedTarget] = []
    var clusters: [ObstacleCluster] = []
    var cameraTransform = matrix_identity_float4x4
    var floorY: Float = -1.3
    var processingMs: Double = 0
    var fps: Double = 0
    var frameCount = 0

    var cameraPosition: SIMD3<Float> {
        SIMD3(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
    }
}

struct RadarSettings: Equatable {
    var clustering = ClusteringParameters()
    var tracker = TrackerParameters()
    var pointStep = 3
    var colorMode: PointColorMode = .height
    var lidarClusteringEnabled = true
    var visionEnabled = true
    var accumulateMap = true
    var showClusterBoxes = true
}

/// Serial background pipeline: depth frame → point cloud → floor estimate → voxel clustering
/// → α-β tracking → map accumulation → snapshot. Thread-safe; `submit` drops frames while busy.
final class RadarPipeline {
    var onSnapshot: ((RadarSnapshot) -> Void)?

    private let queue = DispatchQueue(label: "radar.pipeline", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false

    // Accessed only on `queue`.
    private var settings = RadarSettings()
    private var floor = FloorEstimator()
    private let lidarTracker = AlphaBetaTracker(firstID: 1)
    private let visionTracker: AlphaBetaTracker
    private var map = PointMapAccumulator()
    private var frameCount = 0
    private var lastFrameTime: CFTimeInterval?
    private var fps: Double = 0

    init() {
        var vp = TrackerParameters()
        vp.gateRadius = 0.8
        vp.confirmationHits = 2
        vp.maxMisses = 6
        visionTracker = AlphaBetaTracker(parameters: vp, firstID: 10_000)
    }

    var isBusy: Bool {
        lock.lock(); defer { lock.unlock() }
        return busy
    }

    func update(settings newValue: RadarSettings) {
        queue.async {
            let old = self.settings
            self.settings = newValue
            self.lidarTracker.parameters = newValue.tracker
            if old.colorMode != newValue.colorMode || (!newValue.accumulateMap && old.accumulateMap) {
                self.map.reset()
            }
        }
    }

    func reset() {
        queue.async {
            self.floor.reset()
            self.lidarTracker.reset()
            self.visionTracker.reset()
            self.map.reset()
        }
    }

    /// Returns `false` when the frame was dropped because the previous one is still processing.
    @discardableResult
    func submit(_ frame: DepthFrame) -> Bool {
        lock.lock()
        if busy { lock.unlock(); return false }
        busy = true
        lock.unlock()
        queue.async {
            self.process(frame)
            self.lock.lock(); self.busy = false; self.lock.unlock()
        }
        return true
    }

    func submitVision(_ measurements: [TargetMeasurement], timestamp: TimeInterval, observer: SIMD3<Float>) {
        queue.async {
            guard self.settings.visionEnabled else { return }
            self.visionTracker.update(measurements: measurements, timestamp: timestamp, observer: observer)
        }
    }

    private func process(_ frame: DepthFrame) {
        let start = CACurrentMediaTime()
        let s = settings
        let camPos = frame.cameraPosition

        let positions = PointCloudBuilder.positions(from: frame, options: PointCloudOptions(step: s.pointStep))
        let floorY = floor.update(with: positions) ?? (camPos.y - 1.3)

        var clusters: [ObstacleCluster] = []
        if s.lidarClusteringEnabled {
            clusters = VoxelClusterer.cluster(points: positions, floorY: floorY, parameters: s.clustering).clusters
            lidarTracker.update(measurements: clusters.map { TargetMeasurement(cluster: $0) }, timestamp: frame.timestamp, observer: camPos)
        } else if !lidarTracker.tracks.isEmpty {
            lidarTracker.reset()
        }
        if !s.visionEnabled && !visionTracker.tracks.isEmpty { visionTracker.reset() }

        let vertices = positions.map {
            PointCloudVertex(position: $0, color: TacticalColorMap.color(for: $0, mode: s.colorMode, floorY: floorY, cameraPosition: camPos))
        }

        frameCount += 1
        if s.accumulateMap && frameCount % 4 == 0 && !map.isFull {
            // Keep obstacles out of the persistent map so moving targets don't leave smears.
            let staticPoints = clusters.isEmpty ? vertices : vertices.filter { v in
                !clusters.contains { $0.contains(v.position, margin: 0.08) }
            }
            map.insert(staticPoints)
        }

        let now = CACurrentMediaTime()
        if let last = lastFrameTime {
            let inst = 1 / max(now - last, 1e-3)
            fps = fps == 0 ? inst : fps * 0.9 + inst * 0.1
        }
        lastFrameTime = now

        let targets = (lidarTracker.confirmedTracks + visionTracker.confirmedTracks)
            .sorted { $0.distanceMeters < $1.distanceMeters }

        let snapshot = RadarSnapshot(
            livePoints: vertices,
            mapPoints: s.accumulateMap ? map.points : [],
            mapVersion: map.version,
            targets: targets,
            clusters: clusters,
            cameraTransform: frame.cameraTransform,
            floorY: floorY,
            processingMs: (CACurrentMediaTime() - start) * 1000,
            fps: fps,
            frameCount: frameCount)
        DispatchQueue.main.async { self.onSnapshot?(snapshot) }
    }
}
