import Foundation
import simd

/// Deterministic virtual room that ray-casts a fake LiDAR depth map. Lets the full pipeline
/// (unprojection → clustering → tracking → rendering) run in the iOS Simulator and in tests.
public struct SyntheticScene {
    public enum Shape: Equatable {
        case box(center: SIMD3<Float>, halfExtents: SIMD3<Float>)
        case cylinder(base: SIMD3<Float>, radius: Float, height: Float)
        case ellipsoid(center: SIMD3<Float>, radii: SIMD3<Float>)
    }

    public struct Object: Equatable {
        public var name: String
        public var shape: Shape
        public var isDynamic: Bool

        public var center: SIMD3<Float> {
            switch shape {
            case let .box(c, _): return c
            case let .cylinder(b, _, h): return b + SIMD3(0, h / 2, 0)
            case let .ellipsoid(c, _): return c
            }
        }

        public var footprint: Float {
            switch shape {
            case let .box(_, h): return 2 * max(h.x, h.z)
            case let .cylinder(_, r, _): return 2 * r
            case let .ellipsoid(_, r): return 2 * max(r.x, r.z)
            }
        }
    }

    public var floorY: Float = -1.35
    public var roomMin = SIMD3<Float>(-3.5, -1.35, -5.0)
    public var roomMax = SIMD3<Float>(3.5, 1.25, 2.5)
    public var imageWidth: Float = 1920
    public var imageHeight: Float = 1440
    public var focalLength: Float = 1450
    public var cameraMotion = true

    public init() {}

    public func objects(at t: Double) -> [Object] {
        let a = Float(t * 0.7)
        let dog = SIMD3<Float>(0.3 + 0.6 * cos(a), floorY + 0.32, -3.1 + 0.6 * sin(a))
        let personX = Float(1.3 * sin(t * 0.45))
        return [
            Object(name: "sofa", shape: .box(center: SIMD3(-1.85, floorY + 0.4, -3.3), halfExtents: SIMD3(1.1, 0.4, 0.45)), isDynamic: false),
            Object(name: "crate", shape: .box(center: SIMD3(1.7, floorY + 0.25, -2.4), halfExtents: SIMD3(repeating: 0.25)), isDynamic: false),
            Object(name: "person", shape: .cylinder(base: SIMD3(personX, floorY, -1.6), radius: 0.22, height: 1.7), isDynamic: true),
            Object(name: "dog", shape: .ellipsoid(center: dog, radii: SIMD3(0.3, 0.2, 0.3)), isDynamic: true),
        ]
    }

    public func cameraTransform(at t: Double) -> simd_float4x4 {
        let yaw = cameraMotion ? Float(0.42 * sin(t * 0.22)) : 0
        let pitch: Float = -0.30 + (cameraMotion ? Float(0.04 * sin(t * 0.5)) : 0)
        let pos = SIMD3<Float>(cameraMotion ? Float(0.25 * sin(t * 0.13)) : 0, 0, 1.2)
        var m = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))) * simd_float4x4(simd_quatf(angle: pitch, axis: SIMD3(1, 0, 0)))
        m.columns.3 = SIMD4(pos, 1)
        return m
    }

    public func intrinsics(depthWidth: Int, depthHeight: Int) -> DepthIntrinsics {
        let k = simd_float3x3(SIMD3(focalLength, 0, 0), SIMD3(0, focalLength, 0), SIMD3(imageWidth / 2, imageHeight / 2, 1))
        return DepthIntrinsics(cameraIntrinsics: k, imageWidth: imageWidth, imageHeight: imageHeight, depthWidth: depthWidth, depthHeight: depthHeight)
    }

    /// Renders a z-depth map. Only every `pixelStep`-th pixel is ray-cast; the rest stay 0 (invalid).
    public func renderDepthFrame(at t: Double, width: Int = 256, height: Int = 192, pixelStep: Int = 1, noise: Float = 0.004, seed: UInt64 = 1) -> DepthFrame {
        let k = intrinsics(depthWidth: width, depthHeight: height)
        let m = cameraTransform(at: t)
        let r = simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                              SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                              SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        let origin = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        let objs = objects(at: t)
        var rng = SplitMix64(seed: seed &+ UInt64(max(0, t) * 1000))
        var depth = [Float](repeating: 0, count: width * height)
        var conf = [UInt8](repeating: 0, count: width * height)
        let step = max(1, pixelStep)
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let dc = SIMD3<Float>((Float(x) - k.cx) / k.fx, -(Float(y) - k.cy) / k.fy, -1)
                guard var z = raycast(origin: origin, direction: r * dc, objects: objs) else { continue }
                if noise > 0 { z *= 1 + noise * rng.nextGaussian() }
                let i = y * width + x
                depth[i] = z
                conf[i] = rng.nextFloat() < 0.005 ? 0 : (z < 3.5 ? 2 : 1)
            }
        }
        return DepthFrame(width: width, height: height, depth: depth, confidence: conf, intrinsics: k, cameraTransform: m, timestamp: t)
    }

    /// Returns the ray parameter of the nearest hit. With `direction.z_cam == −1` this equals z-depth.
    public func raycast(origin o: SIMD3<Float>, direction d: SIMD3<Float>, objects: [Object]) -> Float? {
        var best = Self.roomInterior(o, d, roomMin, roomMax) ?? .infinity
        for obj in objects {
            let hit: Float?
            switch obj.shape {
            case let .box(c, h): hit = Self.box(o, d, c - h, c + h)
            case let .cylinder(b, rad, h): hit = Self.cylinder(o, d, b, rad, h)
            case let .ellipsoid(c, rad): hit = Self.ellipsoid(o, d, c, rad)
            }
            if let hit, hit < best { best = hit }
        }
        return best.isFinite ? best : nil
    }

    static func roomInterior(_ o: SIMD3<Float>, _ d: SIMD3<Float>, _ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Float? {
        var t = Float.infinity
        for a in 0..<3 {
            if d[a] > 1e-6 { t = min(t, (hi[a] - o[a]) / d[a]) }
            else if d[a] < -1e-6 { t = min(t, (lo[a] - o[a]) / d[a]) }
        }
        return t.isFinite && t > 0 ? t : nil
    }

    static func box(_ o: SIMD3<Float>, _ d: SIMD3<Float>, _ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Float? {
        var tmin: Float = -.infinity, tmax: Float = .infinity
        for a in 0..<3 {
            if abs(d[a]) < 1e-8 {
                if o[a] < lo[a] || o[a] > hi[a] { return nil }
                continue
            }
            let t1 = (lo[a] - o[a]) / d[a], t2 = (hi[a] - o[a]) / d[a]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        }
        return tmax >= tmin && tmin > 0 ? tmin : nil
    }

    static func cylinder(_ o: SIMD3<Float>, _ d: SIMD3<Float>, _ base: SIMD3<Float>, _ r: Float, _ h: Float) -> Float? {
        var best = Float.infinity
        let ox = o.x - base.x, oz = o.z - base.z
        let a = d.x * d.x + d.z * d.z
        if a > 1e-10 {
            let b = 2 * (ox * d.x + oz * d.z)
            let c = ox * ox + oz * oz - r * r
            let disc = b * b - 4 * a * c
            if disc >= 0 {
                let t = (-b - disc.squareRoot()) / (2 * a)
                let y = o.y + t * d.y
                if t > 0 && y >= base.y && y <= base.y + h { best = t }
            }
        }
        if abs(d.y) > 1e-8 {
            let t = (base.y + h - o.y) / d.y
            let px = ox + t * d.x, pz = oz + t * d.z
            if t > 0 && px * px + pz * pz <= r * r { best = min(best, t) }
        }
        return best.isFinite ? best : nil
    }

    static func ellipsoid(_ o: SIMD3<Float>, _ d: SIMD3<Float>, _ c: SIMD3<Float>, _ r: SIMD3<Float>) -> Float? {
        let oo = (o - c) / r, dd = d / r
        let a = simd_dot(dd, dd), b = 2 * simd_dot(oo, dd), cc = simd_dot(oo, oo) - 1
        let disc = b * b - 4 * a * cc
        guard disc >= 0 else { return nil }
        let t = (-b - disc.squareRoot()) / (2 * a)
        return t > 0 ? t : nil
    }
}

/// Small deterministic PRNG (SplitMix64).
public struct SplitMix64 {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func nextFloat() -> Float { Float(next() >> 40) / Float(1 << 24) }

    public mutating func nextGaussian() -> Float {
        let u1 = max(nextFloat(), 1e-7), u2 = nextFloat()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}
