import simd

public enum PointColorMode: String, CaseIterable, Identifiable {
    case height = "Height"
    case distance = "Distance"

    public var id: String { rawValue }
}

public enum TacticalColorMap {
    /// Deep Blue → Cyan → Neon Green → Yellow for `t ∈ [0, 1]`.
    @inlinable
    public static func gradient(_ t: Float) -> SIMD3<Float> {
        let t = simd_clamp(t, 0, 1)
        if t < 0.35 {
            let l = t / 0.35
            return SIMD3(0, l * 0.8, 1)
        } else if t < 0.7 {
            let l = (t - 0.35) / 0.35
            return SIMD3(0, 0.8 + 0.2 * l, 1 - l)
        } else {
            let l = (t - 0.7) / 0.3
            return SIMD3(l, 1, 0.1 * l)
        }
    }

    /// Height mode spans 2.2 m above the floor; distance mode spans 0–5 m from the device.
    @inlinable
    public static func color(for p: SIMD3<Float>, mode: PointColorMode, floorY: Float, cameraPosition: SIMD3<Float>) -> SIMD3<Float> {
        switch mode {
        case .height: return gradient((p.y - floorY) / 2.2)
        case .distance: return gradient(simd_distance(p, cameraPosition) / 5.0)
        }
    }
}
