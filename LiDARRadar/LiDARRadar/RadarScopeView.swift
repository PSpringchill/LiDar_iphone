import SwiftUI
import simd
import RadarCore

/// 2D plan-position-indicator (PPI) scope: device at the center, heading up, rotating sweep.
struct RadarScopeView: View {
    let targets: [TrackedTarget]
    let cameraTransform: simd_float4x4
    var range: Float = 5

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { ctx, size in
                let r = min(size.width, size.height) / 2
                let c = CGPoint(x: size.width / 2, y: size.height / 2)
                let circle = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r))
                ctx.fill(circle, with: .color(Color(red: 0.02, green: 0.08, blue: 0.06).opacity(0.85)))
                let green = Color(red: 0.2, green: 1, blue: 0.5)
                for i in 1...4 {
                    let rr = r * CGFloat(i) / 4
                    ctx.stroke(Path(ellipseIn: CGRect(x: c.x - rr, y: c.y - rr, width: 2 * rr, height: 2 * rr)), with: .color(green.opacity(0.25)), lineWidth: 0.7)
                }
                var cross = Path()
                cross.move(to: CGPoint(x: c.x - r, y: c.y)); cross.addLine(to: CGPoint(x: c.x + r, y: c.y))
                cross.move(to: CGPoint(x: c.x, y: c.y - r)); cross.addLine(to: CGPoint(x: c.x, y: c.y + r))
                ctx.stroke(cross, with: .color(green.opacity(0.2)), lineWidth: 0.7)

                // Camera field of view wedge (~67° horizontal).
                var fov = Path()
                fov.move(to: c)
                fov.addArc(center: c, radius: r, startAngle: .degrees(-90 - 33), endAngle: .degrees(-90 + 33), clockwise: false)
                fov.closeSubpath()
                ctx.fill(fov, with: .color(.cyan.opacity(0.08)))

                // Sweep.
                let t = timeline.date.timeIntervalSinceReferenceDate
                let angle = Angle.radians((t.truncatingRemainder(dividingBy: 3)) / 3 * 2 * .pi - .pi / 2)
                var sweep = Path()
                sweep.move(to: c)
                sweep.addArc(center: c, radius: r, startAngle: angle - .degrees(35), endAngle: angle, clockwise: false)
                sweep.closeSubpath()
                ctx.fill(sweep, with: .linearGradient(Gradient(colors: [green.opacity(0), green.opacity(0.35)]),
                                                     startPoint: c, endPoint: CGPoint(x: c.x + cos(angle.radians) * r, y: c.y + sin(angle.radians) * r)))

                let (pos, fwd, right) = basis()
                for target in targets {
                    let rel = target.position - pos
                    let x = simd_dot(rel, right), y = simd_dot(rel, fwd)
                    guard (x * x + y * y).squareRoot() <= range else { continue }
                    let p = CGPoint(x: c.x + CGFloat(x / range) * r, y: c.y - CGFloat(y / range) * r)
                    let color: Color = target.source == .vision ? .orange : .pink
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)), with: .color(color))
                    ctx.draw(Text("\(target.id)").font(.system(size: 8, design: .monospaced)).foregroundColor(.white), at: CGPoint(x: p.x, y: p.y - 9))
                }
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(.white))
                ctx.stroke(circle, with: .color(green.opacity(0.7)), lineWidth: 1.2)
            }
        }
        .accessibilityLabel("Radar scope, \(targets.count) targets")
    }

    private func basis() -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        let m = cameraTransform
        let pos = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        var fwd = -SIMD3(m.columns.2.x, 0, m.columns.2.z)
        fwd = simd_length(fwd) > 1e-3 ? simd_normalize(fwd) : SIMD3(0, 0, -1)
        let right = simd_cross(fwd, SIMD3(0, 1, 0))
        return (pos, fwd, right)
    }
}
