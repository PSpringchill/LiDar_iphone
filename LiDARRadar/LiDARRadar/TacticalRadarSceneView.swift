import SwiftUI
import SceneKit
import simd
import RadarCore

enum RadarViewMode: String, CaseIterable, Identifiable {
    case follow = "Follow"
    case orbit = "Orbit"
    case top = "Top"

    var id: String { rawValue }
}

/// Dark tactical 3D viewport: reference grid, range rings, device frustum + axes, live and
/// accumulated LiDAR point clouds, cluster boxes, magenta target reticles and trajectories.
struct TacticalRadarSceneView: UIViewRepresentable {
    let snapshot: RadarSnapshot
    let viewMode: RadarViewMode
    let showClusterBoxes: Bool
    let recenterToken: Int

    func makeCoordinator() -> TacticalSceneController { TacticalSceneController() }

    func makeUIView(context: Context) -> SCNView { context.coordinator.makeView() }

    func updateUIView(_ view: SCNView, context: Context) {
        context.coordinator.apply(snapshot: snapshot, viewMode: viewMode, showClusterBoxes: showClusterBoxes, recenterToken: recenterToken)
    }
}

final class TacticalSceneController {
    static let background = UIColor(red: 0.08, green: 0.09, blue: 0.16, alpha: 1)
    static let magenta = UIColor(red: 1.0, green: 0.2, blue: 0.8, alpha: 1)

    let scene = SCNScene()
    let cameraNode = SCNNode()
    private let liveCloud = SCNNode()
    private let mapCloud = SCNNode()
    private let gridNode = SCNNode()
    private let rangeRings = SCNNode()
    private let deviceNode = SCNNode()
    private let targetsRoot = SCNNode()
    private let clusterNode = SCNNode()
    private var targetNodes: [Int: TargetNodes] = [:]
    private var lastMapVersion = -1
    private var currentMode: RadarViewMode?
    private var lastRecenter = 0
    private var headingH = SIMD3<Float>(0, 0, -1)
    private weak var view: SCNView?

    private struct TargetNodes {
        let root: SCNNode
        let ring: SCNTorus
        let label: SCNText
        let labelNode: SCNNode
        let stalk: SCNNode
        let velocity: SCNNode
        let trail: SCNNode
        var labelText = ""
    }

    init() {
        let camera = SCNCamera()
        camera.zNear = 0.05
        camera.zFar = 40
        camera.fieldOfView = 60
        cameraNode.camera = camera
        cameraNode.name = "tacticalCamera"
        cameraNode.simdPosition = SIMD3(0, 1.2, 1.8)
        cameraNode.simdEulerAngles = SIMD3(-.pi / 6, 0, 0)

        scene.rootNode.addChildNode(cameraNode)
        gridNode.geometry = Self.gridGeometry(size: 6, step: 0.5)
        scene.rootNode.addChildNode(gridNode)
        rangeRings.geometry = Self.rangeRingGeometry(radii: [1, 2, 3, 4, 5])
        scene.rootNode.addChildNode(rangeRings)
        scene.rootNode.addChildNode(Self.axesNode(length: 0.25, opacity: 0.5))
        mapCloud.name = "mapCloud"
        scene.rootNode.addChildNode(mapCloud)
        liveCloud.name = "pointCloudContainer"
        scene.rootNode.addChildNode(liveCloud)
        scene.rootNode.addChildNode(clusterNode)
        targetsRoot.name = "targetsContainer"
        scene.rootNode.addChildNode(targetsRoot)

        deviceNode.addChildNode(SCNNode(geometry: Self.frustumGeometry()))
        deviceNode.addChildNode(Self.axesNode(length: 0.12, opacity: 1))
        scene.rootNode.addChildNode(deviceNode)
    }

    func makeView() -> SCNView {
        let v = SCNView(frame: .zero)
        v.scene = scene
        v.backgroundColor = Self.background
        v.autoenablesDefaultLighting = false
        v.antialiasingMode = .multisampling4X
        v.preferredFramesPerSecond = 60
        v.pointOfView = cameraNode
        view = v
        return v
    }

    func apply(snapshot s: RadarSnapshot, viewMode: RadarViewMode, showClusterBoxes: Bool, recenterToken: Int) {
        SCNTransaction.begin()
        SCNTransaction.disableActions = true
        defer { SCNTransaction.commit() }

        liveCloud.geometry = Self.pointCloudGeometry(s.livePoints, pointSize: 4, minRadius: 1.5, maxRadius: 6, opacity: 1)
        if s.mapVersion != lastMapVersion || (s.mapPoints.isEmpty && mapCloud.geometry != nil) {
            lastMapVersion = s.mapVersion
            mapCloud.geometry = Self.pointCloudGeometry(s.mapPoints, pointSize: 2.5, minRadius: 1, maxRadius: 4, opacity: 0.45)
        }

        gridNode.simdPosition.y = s.floorY
        let camPos = s.cameraPosition
        rangeRings.simdPosition = SIMD3(camPos.x, s.floorY + 0.005, camPos.z)
        deviceNode.simdTransform = s.cameraTransform

        let fwd = -SIMD3(s.cameraTransform.columns.2.x, 0, s.cameraTransform.columns.2.z)
        if simd_length(fwd) > 0.15 { headingH = simd_normalize(fwd) }

        updateTargets(s.targets, floorY: s.floorY)
        clusterNode.geometry = showClusterBoxes ? Self.boxesGeometry(s.clusters) : nil
        updateCamera(mode: viewMode, devicePosition: camPos, recenterToken: recenterToken)
    }

    // MARK: - Targets

    private func updateTargets(_ targets: [TrackedTarget], floorY: Float) {
        var alive = Set<Int>()
        for t in targets {
            alive.insert(t.id)
            var nodes = targetNodes[t.id] ?? makeTargetNodes()
            nodes.root.simdPosition = t.position
            let ringRadius = CGFloat(max(0.16, min(0.6, max(t.size.x, t.size.z) / 2 + 0.06)))
            if abs(nodes.ring.ringRadius - ringRadius) > 0.02 { nodes.ring.ringRadius = ringRadius }

            let text = String(format: "%@  %.2fm", t.displayName, t.distanceMeters)
            if text != nodes.labelText {
                nodes.label.string = text
                nodes.labelText = text
            }
            nodes.labelNode.simdPosition = SIMD3(0, max(0.2, t.size.y / 2 + 0.12), 0)
            nodes.stalk.geometry = Self.lineGeometry([.zero, SIMD3(0, floorY - t.position.y, 0)], color: Self.magenta.withAlphaComponent(0.5))
            if t.speed > 0.08 {
                nodes.velocity.geometry = Self.lineGeometry([.zero, t.velocity * 0.6], color: .systemYellow)
            } else {
                nodes.velocity.geometry = nil
            }
            nodes.trail.geometry = t.trajectory.count > 1 ? Self.polylineGeometry(t.trajectory, color: .systemPink) : nil
            targetNodes[t.id] = nodes
        }
        for (id, nodes) in targetNodes where !alive.contains(id) {
            nodes.root.removeFromParentNode()
            nodes.trail.removeFromParentNode()
            targetNodes[id] = nil
        }
    }

    private func makeTargetNodes() -> TargetNodes {
        let root = SCNNode()
        let sphere = SCNSphere(radius: 0.05)
        sphere.firstMaterial = Self.flatMaterial(Self.magenta)
        root.addChildNode(SCNNode(geometry: sphere))

        let spinner = SCNNode()
        let ring = SCNTorus(ringRadius: 0.16, pipeRadius: 0.008)
        ring.firstMaterial = Self.flatMaterial(Self.magenta)
        spinner.addChildNode(SCNNode(geometry: ring))
        for i in 0..<4 {
            let tick = SCNBox(width: 0.07, height: 0.012, length: 0.012, chamferRadius: 0)
            tick.firstMaterial = Self.flatMaterial(Self.magenta)
            let n = SCNNode(geometry: tick)
            let a = Float(i) * .pi / 2
            n.simdPosition = SIMD3(cos(a) * 0.2, 0, sin(a) * 0.2)
            n.simdEulerAngles = SIMD3(0, -a, 0)
            spinner.addChildNode(n)
        }
        spinner.runAction(.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 3)))
        root.addChildNode(spinner)

        let label = SCNText(string: "", extrusionDepth: 0)
        label.font = UIFont.monospacedSystemFont(ofSize: 10, weight: .bold)
        label.flatness = 0.4
        label.firstMaterial = Self.flatMaterial(.white)
        let labelNode = SCNNode(geometry: label)
        labelNode.simdScale = SIMD3(repeating: 0.008)
        labelNode.pivot = SCNMatrix4MakeTranslation(0, 0, 0)
        labelNode.constraints = [SCNBillboardConstraint()]
        let labelHolder = SCNNode()
        labelHolder.addChildNode(labelNode)
        root.addChildNode(labelHolder)

        let stalk = SCNNode()
        let velocity = SCNNode()
        root.addChildNode(stalk)
        root.addChildNode(velocity)
        targetsRoot.addChildNode(root)

        let trail = SCNNode()
        targetsRoot.addChildNode(trail)
        return TargetNodes(root: root, ring: ring, label: label, labelNode: labelHolder, stalk: stalk, velocity: velocity, trail: trail)
    }

    // MARK: - Camera

    private func updateCamera(mode: RadarViewMode, devicePosition p: SIMD3<Float>, recenterToken: Int) {
        guard let view else { return }
        let modeChanged = mode != currentMode
        let recenter = recenterToken != lastRecenter
        currentMode = mode
        lastRecenter = recenterToken

        switch mode {
        case .orbit:
            if modeChanged || recenter {
                view.pointOfView = cameraNode
                cameraNode.simdPosition = p - headingH * 2.2 + SIMD3(0, 1.6, 0)
                cameraNode.simdLook(at: p + headingH * 1.0 + SIMD3(0, -0.6, 0), up: SIMD3(0, 1, 0), localFront: SIMD3(0, 0, -1))
                view.allowsCameraControl = true
                view.defaultCameraController.interactionMode = .orbitTurntable
                view.defaultCameraController.target = SCNVector3(p.x + headingH.x, p.y - 0.6, p.z + headingH.z)
            }
        case .follow, .top:
            if view.allowsCameraControl { view.allowsCameraControl = false }
            if view.pointOfView !== cameraNode { view.pointOfView = cameraNode }
            let eye: SIMD3<Float>, look: SIMD3<Float>, up: SIMD3<Float>
            if mode == .follow {
                eye = p - headingH * 1.8 + SIMD3(0, 1.1, 0)
                look = p + headingH * 1.4 + SIMD3(0, -0.5, 0)
                up = SIMD3(0, 1, 0)
            } else {
                eye = p + SIMD3(0, 6.5, 0) + headingH * 1.5
                look = p + headingH * 1.5
                up = headingH
            }
            let k: Float = (modeChanged || recenter) ? 1 : 0.18
            cameraNode.simdPosition = simd_mix(cameraNode.simdPosition, eye, SIMD3(repeating: k))
            cameraNode.simdLook(at: look, up: up, localFront: SIMD3(0, 0, -1))
        }
    }

    // MARK: - Geometry builders

    static func flatMaterial(_ color: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.diffuse.contents = color
        m.lightingModel = .constant
        m.isDoubleSided = true
        return m
    }

    static func pointCloudGeometry(_ vertices: [PointCloudVertex], pointSize: CGFloat, minRadius: CGFloat, maxRadius: CGFloat, opacity: CGFloat) -> SCNGeometry? {
        guard !vertices.isEmpty else { return nil }
        let stride = MemoryLayout<PointCloudVertex>.stride
        let data = vertices.withUnsafeBytes { Data($0) }
        let positions = SCNGeometrySource(data: data, semantic: .vertex, vectorCount: vertices.count, usesFloatComponents: true,
                                          componentsPerVector: 3, bytesPerComponent: MemoryLayout<Float>.size,
                                          dataOffset: MemoryLayout<PointCloudVertex>.offset(of: \.position)!, dataStride: stride)
        let colors = SCNGeometrySource(data: data, semantic: .color, vectorCount: vertices.count, usesFloatComponents: true,
                                       componentsPerVector: 3, bytesPerComponent: MemoryLayout<Float>.size,
                                       dataOffset: MemoryLayout<PointCloudVertex>.offset(of: \.color)!, dataStride: stride)
        let element = SCNGeometryElement(data: nil, primitiveType: .point, primitiveCount: vertices.count, bytesPerIndex: 0)
        element.pointSize = pointSize
        element.minimumPointScreenSpaceRadius = minRadius
        element.maximumPointScreenSpaceRadius = maxRadius
        let g = SCNGeometry(sources: [positions, colors], elements: [element])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.transparency = opacity
        if opacity < 1 { m.writesToDepthBuffer = false }
        g.materials = [m]
        return g
    }

    static func lineGeometry(_ points: [SIMD3<Float>], color: UIColor) -> SCNGeometry {
        let src = SCNGeometrySource(vertices: points.map { SCNVector3($0.x, $0.y, $0.z) })
        let idx = (0..<Int32(points.count)).map { $0 }
        let g = SCNGeometry(sources: [src], elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        g.firstMaterial = flatMaterial(color)
        return g
    }

    static func polylineGeometry(_ points: [SIMD3<Float>], color: UIColor) -> SCNGeometry {
        var idx: [Int32] = []
        idx.reserveCapacity(points.count * 2)
        for i in 0..<(points.count - 1) { idx.append(Int32(i)); idx.append(Int32(i + 1)) }
        let src = SCNGeometrySource(vertices: points.map { SCNVector3($0.x, $0.y, $0.z) })
        let g = SCNGeometry(sources: [src], elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        g.firstMaterial = flatMaterial(color)
        return g
    }

    static func coloredLines(_ points: [SIMD3<Float>], colors: [SIMD3<Float>], opacity: CGFloat) -> SCNGeometry {
        let pos = SCNGeometrySource(vertices: points.map { SCNVector3($0.x, $0.y, $0.z) })
        let colorData = colors.map { SIMD4<Float>($0, 1) }.withUnsafeBytes { Data($0) }
        let col = SCNGeometrySource(data: colorData, semantic: .color, vectorCount: colors.count, usesFloatComponents: true,
                                    componentsPerVector: 4, bytesPerComponent: 4, dataOffset: 0, dataStride: 16)
        let idx = (0..<Int32(points.count)).map { $0 }
        let g = SCNGeometry(sources: [pos, col], elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        let m = flatMaterial(.white)
        m.transparency = opacity
        g.firstMaterial = m
        return g
    }

    static func axesNode(length l: Float, opacity: CGFloat) -> SCNNode {
        let pts: [SIMD3<Float>] = [.zero, SIMD3(l, 0, 0), .zero, SIMD3(0, l, 0), .zero, SIMD3(0, 0, l)]
        let cols: [SIMD3<Float>] = [SIMD3(1, 0.2, 0.2), SIMD3(1, 0.2, 0.2), SIMD3(0.2, 1, 0.2), SIMD3(0.2, 1, 0.2), SIMD3(0.3, 0.5, 1), SIMD3(0.3, 0.5, 1)]
        return SCNNode(geometry: coloredLines(pts, colors: cols, opacity: opacity))
    }

    static func frustumGeometry() -> SCNGeometry {
        let d: Float = -0.22, hw: Float = 0.12, hh: Float = 0.09
        let c = [SIMD3(-hw, hh, d), SIMD3(hw, hh, d), SIMD3(hw, -hh, d), SIMD3(-hw, -hh, d)]
        var pts: [SIMD3<Float>] = []
        for i in 0..<4 {
            pts += [.zero, c[i]]
            pts += [c[i], c[(i + 1) % 4]]
        }
        return lineGeometry(pts, color: UIColor(white: 1, alpha: 0.9))
    }

    static func gridGeometry(size: Float, step: Float) -> SCNGeometry {
        var pts: [SIMD3<Float>] = []
        for i in Swift.stride(from: -size, through: size, by: step) {
            pts += [SIMD3(i, 0, -size), SIMD3(i, 0, size), SIMD3(-size, 0, i), SIMD3(size, 0, i)]
        }
        return lineGeometry(pts, color: UIColor(white: 1, alpha: 0.18))
    }

    static func rangeRingGeometry(radii: [Float]) -> SCNGeometry {
        var pts: [SIMD3<Float>] = []
        let segments = 72
        for r in radii {
            for i in 0..<segments {
                let a0 = Float(i) / Float(segments) * 2 * .pi, a1 = Float(i + 1) / Float(segments) * 2 * .pi
                pts += [SIMD3(cos(a0) * r, 0, sin(a0) * r), SIMD3(cos(a1) * r, 0, sin(a1) * r)]
            }
        }
        return lineGeometry(pts, color: UIColor(red: 0.2, green: 1, blue: 0.5, alpha: 0.28))
    }

    static func boxesGeometry(_ clusters: [ObstacleCluster]) -> SCNGeometry? {
        guard !clusters.isEmpty else { return nil }
        var pts: [SIMD3<Float>] = []
        pts.reserveCapacity(clusters.count * 24)
        for c in clusters {
            let a = c.boundsMin, b = c.boundsMax
            let v = [SIMD3(a.x, a.y, a.z), SIMD3(b.x, a.y, a.z), SIMD3(b.x, a.y, b.z), SIMD3(a.x, a.y, b.z),
                     SIMD3(a.x, b.y, a.z), SIMD3(b.x, b.y, a.z), SIMD3(b.x, b.y, b.z), SIMD3(a.x, b.y, b.z)]
            for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
                pts += [v[i], v[j]]
            }
        }
        return lineGeometry(pts, color: UIColor(red: 0.2, green: 0.95, blue: 1, alpha: 0.7))
    }
}
