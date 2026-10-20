import simd

/// Tunables for the pure-LiDAR obstacle detector.
public struct ClusteringParameters: Equatable {
    /// Edge length `v` of a cubic voxel.
    public var voxelSize: Float = 0.07
    /// `d_th`: max distance between neighbouring voxel centroids to be joined.
    public var clusterTolerance: Float = 0.14
    public var minPointsPerCluster: Int = 18
    public var maxPointsPerCluster: Int = 25_000
    /// Points must be at least this far above the floor (removes floor + rugs).
    public var floorClearanceMeters: Float = 0.12
    /// Points higher than this above the floor (ceiling, lamps) are ignored.
    public var maxHeightAboveFloor: Float = 2.0
    /// Clusters whose horizontal footprint `max(ΔX, ΔZ)` reaches this are treated as walls / large furniture.
    public var maxWallSpanMeters: Float = 1.8

    public init() {}
}

public struct ObstacleCluster: Equatable {
    public var centroid: SIMD3<Float>
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>
    public var pointCount: Int
    public var voxelCount: Int

    public init(centroid: SIMD3<Float>, boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>, pointCount: Int, voxelCount: Int) {
        self.centroid = centroid
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.pointCount = pointCount
        self.voxelCount = voxelCount
    }

    public var size: SIMD3<Float> { boundsMax - boundsMin }
    public var center: SIMD3<Float> { (boundsMin + boundsMax) / 2 }
    public var footprint: Float { max(size.x, size.z) }

    public func contains(_ p: SIMD3<Float>, margin: Float = 0) -> Bool {
        all(p .>= boundsMin - margin) && all(p .<= boundsMax + margin)
    }
}

public struct ClusteringResult: Equatable {
    public var floorY: Float
    public var clusters: [ObstacleCluster]
    /// Points that survived ground / ceiling removal.
    public var obstaclePointCount: Int
    public var occupiedVoxelCount: Int
    public var rejectedClusterCount: Int
}

/// Ground removal → spatial-hashed voxel grid → 26-connected BFS Euclidean clustering →
/// size / footprint rejection. Linear in the number of points.
public enum VoxelClusterer {
    private struct Voxel {
        var sum: SIMD3<Float>
        var count: Int32
        var minP: SIMD3<Float>
        var maxP: SIMD3<Float>
        var label: Int32 = -1

        var mean: SIMD3<Float> { sum / Float(count) }
    }

    private static let neighborOffsets: [SIMD3<Int32>] = {
        var o: [SIMD3<Int32>] = []
        for dx in -1...1 { for dy in -1...1 { for dz in -1...1 where !(dx == 0 && dy == 0 && dz == 0) {
            o.append(SIMD3(Int32(dx), Int32(dy), Int32(dz)))
        } } }
        return o
    }()

    @inline(__always)
    static func pack(_ k: SIMD3<Int32>) -> Int64 {
        let m: Int64 = 0x1F_FFFF
        return ((Int64(k.x) & m) << 42) | ((Int64(k.y) & m) << 21) | (Int64(k.z) & m)
    }

    @inline(__always)
    static func voxelKey(_ p: SIMD3<Float>, inverseSize inv: Float) -> SIMD3<Int32> {
        SIMD3(Int32((p.x * inv).rounded(.down)), Int32((p.y * inv).rounded(.down)), Int32((p.z * inv).rounded(.down)))
    }

    public static func cluster(points: [SIMD3<Float>], floorY: Float, parameters p: ClusteringParameters = .init()) -> ClusteringResult {
        let lo = floorY + p.floorClearanceMeters
        let hi = floorY + p.maxHeightAboveFloor
        let inv = 1 / max(p.voxelSize, 0.005)

        var index: [Int64: Int32] = [:]
        index.reserveCapacity(points.count / 2)
        var voxels: [Voxel] = []
        var coords: [SIMD3<Int32>] = []
        var kept = 0

        // 1–2. Gravity-aligned ground removal + voxel hashing.
        for pt in points where pt.y > lo && pt.y < hi {
            kept += 1
            let k = voxelKey(pt, inverseSize: inv)
            let key = pack(k)
            if let i = index[key] {
                let i = Int(i)
                voxels[i].sum += pt
                voxels[i].count += 1
                voxels[i].minP = simd_min(voxels[i].minP, pt)
                voxels[i].maxP = simd_max(voxels[i].maxP, pt)
            } else {
                index[key] = Int32(voxels.count)
                voxels.append(Voxel(sum: pt, count: 1, minP: pt, maxP: pt))
                coords.append(k)
            }
        }

        // 3. BFS over the 26-connected neighbourhood.
        var clusters: [ObstacleCluster] = []
        var rejected = 0
        var queue: [Int] = []
        let tol2 = p.clusterTolerance * p.clusterTolerance
        var label: Int32 = 0
        for seed in voxels.indices where voxels[seed].label < 0 {
            voxels[seed].label = label
            queue.removeAll(keepingCapacity: true)
            queue.append(seed)
            var head = 0
            var sum = SIMD3<Float>(repeating: 0)
            var count = 0
            var bmin = voxels[seed].minP
            var bmax = voxels[seed].maxP
            while head < queue.count {
                let j = queue[head]
                head += 1
                let vj = voxels[j]
                sum += vj.sum
                count += Int(vj.count)
                bmin = simd_min(bmin, vj.minP)
                bmax = simd_max(bmax, vj.maxP)
                let mj = vj.mean
                let cj = coords[j]
                for off in neighborOffsets {
                    guard let n = index[pack(cj &+ off)] else { continue }
                    let n32 = Int(n)
                    if voxels[n32].label >= 0 { continue }
                    if simd_distance_squared(mj, voxels[n32].mean) <= tol2 {
                        voxels[n32].label = label
                        queue.append(n32)
                    }
                }
            }
            label += 1

            // 4. Noise & wall rejection.
            let c = ObstacleCluster(centroid: sum / Float(count), boundsMin: bmin, boundsMax: bmax, pointCount: count, voxelCount: queue.count)
            if count >= p.minPointsPerCluster && count <= p.maxPointsPerCluster && c.footprint < p.maxWallSpanMeters {
                clusters.append(c)
            } else {
                rejected += 1
            }
        }
        clusters.sort { $0.pointCount > $1.pointCount }
        return ClusteringResult(floorY: floorY, clusters: clusters, obstaclePointCount: kept, occupiedVoxelCount: voxels.count, rejectedClusterCount: rejected)
    }
}

/// Estimates floor height from the 5th percentile of world-space Y, smoothed over time:
/// drops quickly (new, lower floor seen) and rises slowly (floor temporarily out of view).
public struct FloorEstimator {
    public var percentile: Float = 0.05
    public var dropRate: Float = 0.5
    public var riseRate: Float = 0.02
    public var minimumPoints = 200
    public private(set) var floorY: Float?

    public init() {}

    public mutating func reset() { floorY = nil }

    @discardableResult
    public mutating func update(with points: [SIMD3<Float>]) -> Float? {
        guard let y = Self.percentileY(points, percentile: percentile, minimumPoints: minimumPoints) else { return floorY }
        if let f = floorY {
            floorY = f + (y - f) * (y < f ? dropRate : riseRate)
        } else {
            floorY = y
        }
        return floorY
    }

    public static func percentileY(_ points: [SIMD3<Float>], percentile: Float, minimumPoints: Int = 1) -> Float? {
        guard points.count >= max(1, minimumPoints) else { return nil }
        let stride = max(1, points.count / 4000)
        var ys: [Float] = []
        ys.reserveCapacity(points.count / stride + 1)
        var i = 0
        while i < points.count { ys.append(points[i].y); i += stride }
        ys.sort()
        let k = Int((Float(ys.count - 1) * simd_clamp(percentile, 0, 1)).rounded())
        return ys[k]
    }
}

/// Persistent voxel-downsampled world map for the "accumulated room" view.
public struct PointMapAccumulator {
    public var voxelSize: Float
    public var capacity: Int
    public private(set) var points: [PointCloudVertex] = []
    public private(set) var version = 0
    private var occupied = Set<Int64>()

    public init(voxelSize: Float = 0.05, capacity: Int = 80_000) {
        self.voxelSize = voxelSize
        self.capacity = capacity
    }

    public var isFull: Bool { points.count >= capacity }

    @discardableResult
    public mutating func insert(_ vertices: [PointCloudVertex]) -> Int {
        let inv = 1 / voxelSize
        var added = 0
        for v in vertices {
            if points.count >= capacity { break }
            let key = VoxelClusterer.pack(VoxelClusterer.voxelKey(v.position, inverseSize: inv))
            if occupied.insert(key).inserted {
                points.append(v)
                added += 1
            }
        }
        if added > 0 { version += 1 }
        return added
    }

    public mutating func reset() {
        points.removeAll(keepingCapacity: true)
        occupied.removeAll(keepingCapacity: true)
        version += 1
    }
}
