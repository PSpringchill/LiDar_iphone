import Foundation
import simd

public enum TargetSource: String, Equatable {
    case lidarCluster = "LiDAR"
    case vision = "Vision"
}

public struct TargetMeasurement: Equatable {
    public var position: SIMD3<Float>
    public var label: String
    public var size: SIMD3<Float>
    public var source: TargetSource

    public init(position: SIMD3<Float>, label: String, size: SIMD3<Float> = .init(repeating: 0.3), source: TargetSource) {
        self.position = position
        self.label = label
        self.size = size
        self.source = source
    }

    public init(cluster: ObstacleCluster, label: String = "Obstacle") {
        self.init(position: cluster.center, label: label, size: cluster.size, source: .lidarCluster)
    }
}

public struct TrackedTarget: Identifiable, Equatable {
    public let id: Int
    public var label: String
    public var source: TargetSource
    public var position: SIMD3<Float>
    public var velocity: SIMD3<Float>
    public var size: SIMD3<Float>
    public var distanceMeters: Float
    public var trajectory: [SIMD3<Float>]
    public var hits: Int
    public var misses: Int
    public var isConfirmed: Bool
    public var firstSeen: TimeInterval
    public var lastSeen: TimeInterval

    public var speed: Float { simd_length(velocity) }
    public var displayName: String { "\(label.uppercased())-\(id)" }
}

public struct TrackerParameters: Equatable {
    /// Position gain α in `P̂_t = P̂_t|t−1 + α (Z_t − P̂_t|t−1)`.
    public var alpha: Float = 0.35
    /// Velocity gain β in `V_t = V_t−1 + (β / Δt) (Z_t − P̂_t|t−1)`.
    public var beta: Float = 0.08
    /// `r_gate`: max distance between prediction and measurement for association.
    public var gateRadius: Float = 0.5
    public var maxMisses: Int = 10
    public var confirmationHits: Int = 3
    public var maxTrajectoryLength: Int = 60
    public var minTrajectorySpacing: Float = 0.02
    /// Velocity decay per missed update while coasting.
    public var coastVelocityDamping: Float = 0.7
    public var maxSpeed: Float = 6
    public var requireLabelMatch = true

    public init() {}
}

/// Nearest-neighbour gated multi-target α-β tracker.
public final class AlphaBetaTracker {
    public var parameters: TrackerParameters
    public private(set) var tracks: [TrackedTarget] = []
    private var nextID: Int
    private let firstID: Int
    private var lastTimestamp: TimeInterval?

    public init(parameters: TrackerParameters = .init(), firstID: Int = 1) {
        self.parameters = parameters
        self.firstID = firstID
        self.nextID = firstID
    }

    public var confirmedTracks: [TrackedTarget] { tracks.filter(\.isConfirmed) }

    public func reset() {
        tracks.removeAll()
        nextID = firstID
        lastTimestamp = nil
    }

    @discardableResult
    public func update(measurements: [TargetMeasurement], timestamp: TimeInterval, observer: SIMD3<Float>) -> [TrackedTarget] {
        let p = parameters
        let dt: Float = lastTimestamp.map { Float(max(timestamp - $0, 1e-3)) } ?? 0
        lastTimestamp = timestamp

        let predicted = tracks.map { $0.position + $0.velocity * dt }

        var pairs: [(t: Int, m: Int, d: Float)] = []
        for (ti, track) in tracks.enumerated() {
            for (mi, m) in measurements.enumerated() {
                if p.requireLabelMatch && track.label != m.label { continue }
                let d = simd_distance(predicted[ti], m.position)
                if d < p.gateRadius { pairs.append((ti, mi, d)) }
            }
        }
        pairs.sort { $0.d < $1.d }

        var trackHit = [Bool](repeating: false, count: tracks.count)
        var measUsed = [Bool](repeating: false, count: measurements.count)
        for pair in pairs where !trackHit[pair.t] && !measUsed[pair.m] {
            trackHit[pair.t] = true
            measUsed[pair.m] = true
            let z = measurements[pair.m]
            var t = tracks[pair.t]
            let pred = predicted[pair.t]
            let residual = z.position - pred
            t.position = pred + p.alpha * residual
            if dt > 0 {
                t.velocity += (p.beta / dt) * residual
                let s = simd_length(t.velocity)
                if s > p.maxSpeed { t.velocity *= p.maxSpeed / s }
            }
            t.size = simd_mix(t.size, z.size, SIMD3(repeating: 0.3))
            t.hits += 1
            t.misses = 0
            t.lastSeen = timestamp
            if t.hits >= p.confirmationHits { t.isConfirmed = true }
            if let last = t.trajectory.last {
                if simd_distance(last, t.position) >= p.minTrajectorySpacing { t.trajectory.append(t.position) }
            } else {
                t.trajectory.append(t.position)
            }
            if t.trajectory.count > p.maxTrajectoryLength {
                t.trajectory.removeFirst(t.trajectory.count - p.maxTrajectoryLength)
            }
            tracks[pair.t] = t
        }

        for i in tracks.indices where !trackHit[i] {
            tracks[i].misses += 1
            tracks[i].position = predicted[i]
            tracks[i].velocity *= p.coastVelocityDamping
        }
        tracks.removeAll { $0.misses > p.maxMisses || (!$0.isConfirmed && $0.misses > 1) }

        for (mi, m) in measurements.enumerated() where !measUsed[mi] {
            tracks.append(TrackedTarget(
                id: nextID, label: m.label, source: m.source, position: m.position, velocity: .zero, size: m.size,
                distanceMeters: 0, trajectory: [m.position], hits: 1, misses: 0,
                isConfirmed: p.confirmationHits <= 1, firstSeen: timestamp, lastSeen: timestamp))
            nextID += 1
        }

        for i in tracks.indices {
            tracks[i].distanceMeters = simd_distance(tracks[i].position, observer)
        }
        return tracks
    }
}
