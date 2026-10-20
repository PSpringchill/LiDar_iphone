import simd

/// Orientation of the upright image relative to the raw (landscape-right) sensor buffer,
/// mirroring the subset of `CGImagePropertyOrientation` we need.
public enum SensorImageOrientation: String, Equatable {
    case up, down, left, right

    /// Infers how the device is held from gravity expressed in ARKit camera space.
    /// Portrait (home indicator at the bottom) maps to `.right`.
    public static func from(cameraTransform t: simd_float4x4) -> SensorImageOrientation {
        let r = simd_float3x3(SIMD3(t.columns.0.x, t.columns.0.y, t.columns.0.z),
                              SIMD3(t.columns.1.x, t.columns.1.y, t.columns.1.z),
                              SIMD3(t.columns.2.x, t.columns.2.y, t.columns.2.z))
        let g = r.transpose * SIMD3<Float>(0, -1, 0)
        if abs(g.x) > abs(g.y) {
            return g.x > 0 ? .right : .left
        }
        return g.y < 0 ? .up : .down
    }

    /// Converts a Vision normalized point (origin bottom-left of the *oriented* image) to
    /// a normalized sensor-buffer point (origin top-left of the raw buffer).
    public func sensorPoint(fromVision p: SIMD2<Float>) -> SIMD2<Float> {
        switch self {
        case .up: return SIMD2(p.x, 1 - p.y)
        case .down: return SIMD2(1 - p.x, p.y)
        case .right: return SIMD2(1 - p.y, 1 - p.x)
        case .left: return SIMD2(p.y, p.x)
        }
    }

    /// Converts a Vision normalized rect into a normalized sensor-space rect.
    public func sensorRect(fromVision minX: Float, minY: Float, width: Float, height: Float) -> (minX: Float, minY: Float, maxX: Float, maxY: Float) {
        let a = sensorPoint(fromVision: SIMD2(minX, minY))
        let b = sensorPoint(fromVision: SIMD2(minX + width, minY + height))
        return (min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))
    }
}
