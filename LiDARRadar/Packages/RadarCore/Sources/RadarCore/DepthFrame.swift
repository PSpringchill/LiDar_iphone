import Foundation
import simd

/// A single colored point of the tactical point cloud.
/// `SIMD3<Float>` has a 16-byte stride, so position is at offset 0 and color at offset 16.
public struct PointCloudVertex: Equatable {
    public var position: SIMD3<Float>
    public var color: SIMD3<Float>

    public init(position: SIMD3<Float>, color: SIMD3<Float>) {
        self.position = position
        self.color = color
    }
}

/// Pinhole intrinsics expressed in depth-map pixel coordinates.
public struct DepthIntrinsics: Equatable {
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float

    public init(fx: Float, fy: Float, cx: Float, cy: Float) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }

    /// Scales ARKit's `ARCamera.intrinsics` (defined for `imageResolution`) down to the
    /// LiDAR depth-map resolution (typically 256×192).
    public init(cameraIntrinsics k: simd_float3x3, imageWidth: Float, imageHeight: Float, depthWidth: Int, depthHeight: Int) {
        let sx = Float(depthWidth) / imageWidth
        let sy = Float(depthHeight) / imageHeight
        self.init(fx: k[0][0] * sx, fy: k[1][1] * sy, cx: k[2][0] * sx, cy: k[2][1] * sy)
    }
}

/// A CPU copy of one LiDAR depth frame plus the camera pose it was captured with.
/// Copying out of the `CVPixelBuffer` lets ARKit recycle its `ARFrame` immediately.
public struct DepthFrame {
    public var width: Int
    public var height: Int
    /// Row-major depth in meters. `0` means "no sample".
    public var depth: [Float]
    /// Row-major `ARConfidenceLevel` raw values (0 low, 1 medium, 2 high).
    public var confidence: [UInt8]?
    public var intrinsics: DepthIntrinsics
    /// Camera-to-world transform (`ARCamera.transform`).
    public var cameraTransform: simd_float4x4
    public var timestamp: TimeInterval

    public init(width: Int, height: Int, depth: [Float], confidence: [UInt8]?, intrinsics: DepthIntrinsics, cameraTransform: simd_float4x4, timestamp: TimeInterval) {
        precondition(depth.count == width * height, "depth buffer size mismatch")
        self.width = width
        self.height = height
        self.depth = depth
        self.confidence = confidence
        self.intrinsics = intrinsics
        self.cameraTransform = cameraTransform
        self.timestamp = timestamp
    }

    public var cameraPosition: SIMD3<Float> {
        SIMD3(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
    }

    /// Lens direction (camera −Z) in world space.
    public var cameraForward: SIMD3<Float> {
        -SIMD3(cameraTransform.columns.2.x, cameraTransform.columns.2.y, cameraTransform.columns.2.z)
    }
}

public enum Unprojection {
    /// ARKit camera space: +X right, +Y up, −Z out of the back lens. Pixel `v` grows downward.
    @inlinable
    public static func cameraSpace(u: Float, v: Float, depth z: Float, intrinsics k: DepthIntrinsics) -> SIMD3<Float> {
        SIMD3((u - k.cx) * z / k.fx, -(v - k.cy) * z / k.fy, -z)
    }

    /// `P_world = T × P_cam`
    @inlinable
    public static func world(u: Float, v: Float, depth z: Float, intrinsics k: DepthIntrinsics, cameraTransform t: simd_float4x4) -> SIMD3<Float> {
        let p = cameraSpace(u: u, v: v, depth: z, intrinsics: k)
        let w = t * SIMD4<Float>(p, 1)
        return SIMD3(w.x, w.y, w.z)
    }

    /// Inverse of `world(...)`: projects a world point into depth-map pixel coordinates.
    public static func project(world p: SIMD3<Float>, intrinsics k: DepthIntrinsics, cameraTransform t: simd_float4x4) -> (u: Float, v: Float, depth: Float)? {
        let c = t.inverse * SIMD4<Float>(p, 1)
        let z = -c.z
        guard z > 0 else { return nil }
        return (c.x * k.fx / z + k.cx, -c.y * k.fy / z + k.cy, z)
    }
}

public struct PointCloudOptions: Equatable {
    /// Sample every `step`-th pixel in both directions (1 = all 49,152 pixels of a 256×192 map).
    public var step: Int
    /// Minimum `ARConfidenceLevel` raw value to keep (spec skips only `low` = 0).
    public var minConfidence: UInt8
    public var minDepth: Float
    public var maxDepth: Float

    public init(step: Int = 3, minConfidence: UInt8 = 1, minDepth: Float = 0.15, maxDepth: Float = 5.0) {
        self.step = max(1, step)
        self.minConfidence = minConfidence
        self.minDepth = minDepth
        self.maxDepth = maxDepth
    }
}

public enum PointCloudBuilder {
    /// Unprojects valid depth pixels into world-space positions.
    public static func positions(from frame: DepthFrame, options: PointCloudOptions = .init()) -> [SIMD3<Float>] {
        let w = frame.width, h = frame.height, step = options.step
        let k = frame.intrinsics
        let t = frame.cameraTransform
        var out: [SIMD3<Float>] = []
        out.reserveCapacity((w / step + 1) * (h / step + 1))
        frame.depth.withUnsafeBufferPointer { depth in
            let conf = frame.confidence
            for y in stride(from: 0, to: h, by: step) {
                let row = y * w
                for x in stride(from: 0, to: w, by: step) {
                    let idx = row + x
                    if let conf, conf[idx] < options.minConfidence { continue }
                    let z = depth[idx]
                    guard z > options.minDepth && z < options.maxDepth else { continue }
                    out.append(Unprojection.world(u: Float(x), v: Float(y), depth: z, intrinsics: k, cameraTransform: t))
                }
            }
        }
        return out
    }

    /// Robust depth of a region (depth-map pixel rect): median of valid samples on a grid
    /// over the central part of the rect. Used to fuse Vision bounding boxes with LiDAR.
    public static func robustDepth(in frame: DepthFrame, rect: (minX: Float, minY: Float, maxX: Float, maxY: Float), centralFraction: Float = 0.5, grid: Int = 9, minDepth: Float = 0.15, maxDepth: Float = 5.0) -> Float? {
        let cxr = (rect.minX + rect.maxX) / 2, cyr = (rect.minY + rect.maxY) / 2
        let hw = (rect.maxX - rect.minX) * centralFraction / 2
        let hh = (rect.maxY - rect.minY) * centralFraction / 2
        var samples: [Float] = []
        samples.reserveCapacity(grid * grid)
        for gy in 0..<grid {
            for gx in 0..<grid {
                let fx = grid == 1 ? 0 : Float(gx) / Float(grid - 1) * 2 - 1
                let fy = grid == 1 ? 0 : Float(gy) / Float(grid - 1) * 2 - 1
                let x = Int((cxr + fx * hw).rounded())
                let y = Int((cyr + fy * hh).rounded())
                guard x >= 0, y >= 0, x < frame.width, y < frame.height else { continue }
                let idx = y * frame.width + x
                if let conf = frame.confidence, conf[idx] == 0 { continue }
                let z = frame.depth[idx]
                if z > minDepth && z < maxDepth { samples.append(z) }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        return samples[samples.count / 2]
    }
}
