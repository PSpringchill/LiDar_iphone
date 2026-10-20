import XCTest
import simd
@testable import RadarCore

final class UnprojectionTests: XCTestCase {
    func testProjectUnprojectRoundTrip() {
        let scene = SyntheticScene()
        let k = scene.intrinsics(depthWidth: 256, depthHeight: 192)
        let t = scene.cameraTransform(at: 3.7)
        let world = SIMD3<Float>(0.4, -0.8, -2.0)
        let px = try! XCTUnwrap(Unprojection.project(world: world, intrinsics: k, cameraTransform: t))
        let back = Unprojection.world(u: px.u, v: px.v, depth: px.depth, intrinsics: k, cameraTransform: t)
        XCTAssertLessThan(simd_distance(world, back), 1e-4)
    }

    func testCameraSpaceConvention() {
        let k = DepthIntrinsics(fx: 200, fy: 200, cx: 128, cy: 96)
        let center = Unprojection.cameraSpace(u: 128, v: 96, depth: 2, intrinsics: k)
        XCTAssertEqual(center, SIMD3(0, 0, -2))
        let topRight = Unprojection.cameraSpace(u: 228, v: 0, depth: 2, intrinsics: k)
        XCTAssertGreaterThan(topRight.x, 0)
        XCTAssertGreaterThan(topRight.y, 0)
    }

    func testIntrinsicsScaling() {
        let kk = simd_float3x3(SIMD3(1450, 0, 0), SIMD3(0, 1450, 0), SIMD3(960, 720, 1))
        let k = DepthIntrinsics(cameraIntrinsics: kk, imageWidth: 1920, imageHeight: 1440, depthWidth: 256, depthHeight: 192)
        XCTAssertEqual(k.cx, 128, accuracy: 1e-4)
        XCTAssertEqual(k.cy, 96, accuracy: 1e-4)
        XCTAssertEqual(k.fx, 1450 * 256 / 1920, accuracy: 1e-3)
    }

    func testPointCloudSkipsInvalidAndLowConfidence() {
        let k = DepthIntrinsics(fx: 2, fy: 2, cx: 1, cy: 1)
        let frame = DepthFrame(width: 2, height: 2, depth: [1, 0, 9, 2], confidence: [2, 2, 2, 0], intrinsics: k, cameraTransform: matrix_identity_float4x4, timestamp: 0)
        let pts = PointCloudBuilder.positions(from: frame, options: .init(step: 1))
        XCTAssertEqual(pts.count, 1)
    }

    func testRobustDepthIgnoresBackgroundHoles() {
        var depth = [Float](repeating: 1.5, count: 20 * 20)
        depth[10 * 20 + 10] = 4.9
        let frame = DepthFrame(width: 20, height: 20, depth: depth, confidence: nil, intrinsics: .init(fx: 10, fy: 10, cx: 10, cy: 10), cameraTransform: matrix_identity_float4x4, timestamp: 0)
        let z = PointCloudBuilder.robustDepth(in: frame, rect: (4, 4, 16, 16))
        XCTAssertEqual(z, 1.5)
    }
}

final class ColorMapTests: XCTestCase {
    func testGradientEndpoints() {
        XCTAssertEqual(TacticalColorMap.gradient(0), SIMD3(0, 0, 1))
        let top = TacticalColorMap.gradient(1)
        XCTAssertEqual(top.x, 1, accuracy: 1e-5)
        XCTAssertEqual(top.y, 1, accuracy: 1e-5)
        XCTAssertEqual(TacticalColorMap.gradient(-3), TacticalColorMap.gradient(0))
    }
}

final class ClusteringTests: XCTestCase {
    private func blob(center: SIMD3<Float>, size: SIMD3<Float>, spacing: Float = 0.03) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = []
        for x in stride(from: -size.x / 2, through: size.x / 2, by: spacing) {
            for y in stride(from: -size.y / 2, through: size.y / 2, by: spacing) {
                for z in stride(from: -size.z / 2, through: size.z / 2, by: spacing) {
                    pts.append(center + SIMD3(x, y, z))
                }
            }
        }
        return pts
    }

    private func floorPlane(y: Float) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = []
        for x in stride(from: Float(-3), through: 3, by: 0.05) {
            for z in stride(from: Float(-4), through: 0, by: 0.05) { pts.append(SIMD3(x, y, z)) }
        }
        return pts
    }

    func testFindsSeparateObstaclesAndRemovesFloor() {
        let floorY: Float = -1.3
        var pts = floorPlane(y: floorY)
        pts += blob(center: SIMD3(-1, floorY + 0.4, -2), size: SIMD3(0.4, 0.6, 0.4))
        pts += blob(center: SIMD3(1, floorY + 0.8, -2.5), size: SIMD3(0.4, 1.4, 0.4))
        let r = VoxelClusterer.cluster(points: pts, floorY: floorY)
        XCTAssertEqual(r.clusters.count, 2)
        let xs = r.clusters.map(\.center.x).sorted()
        XCTAssertEqual(xs[0], -1, accuracy: 0.05)
        XCTAssertEqual(xs[1], 1, accuracy: 0.05)
    }

    func testRejectsWallsAndSparkleNoise() {
        let floorY: Float = -1.3
        var pts = blob(center: SIMD3(0, floorY + 1, -3), size: SIMD3(3.0, 1.6, 0.05))
        pts.append(SIMD3(1.5, floorY + 0.5, -1))
        pts.append(SIMD3(1.52, floorY + 0.5, -1))
        let r = VoxelClusterer.cluster(points: pts, floorY: floorY)
        XCTAssertTrue(r.clusters.isEmpty)
        XCTAssertEqual(r.rejectedClusterCount, 2)
    }

    func testFloorEstimatorUsesLowPercentile() {
        var pts = floorPlane(y: -1.2)
        pts += blob(center: SIMD3(0, 0, -2), size: SIMD3(1, 1, 1))
        pts.append(SIMD3(0, -9, 0))
        var est = FloorEstimator()
        let f = try! XCTUnwrap(est.update(with: pts))
        XCTAssertEqual(f, -1.2, accuracy: 0.02)
    }

    func testMapAccumulatorDeduplicatesVoxels() {
        var map = PointMapAccumulator(voxelSize: 0.1, capacity: 10)
        let v = PointCloudVertex(position: SIMD3(0.01, 0.01, 0.01), color: .zero)
        XCTAssertEqual(map.insert([v, v, PointCloudVertex(position: SIMD3(0.5, 0, 0), color: .zero)]), 2)
        XCTAssertEqual(map.insert([v]), 0)
        XCTAssertEqual(map.points.count, 2)
    }
}

final class TrackerTests: XCTestCase {
    func testConstantVelocityTargetKeepsIDAndEstimatesVelocity() {
        let tracker = AlphaBetaTracker()
        let v = SIMD3<Float>(0.8, 0, 0)
        var id: Int?
        for i in 0..<90 {
            let t = Double(i) / 30
            let pos = SIMD3<Float>(-1, -1, -2) + v * Float(t)
            let tracks = tracker.update(measurements: [TargetMeasurement(position: pos, label: "Obstacle", source: .lidarCluster)], timestamp: t, observer: .zero)
            XCTAssertEqual(tracks.count, 1)
            if let id { XCTAssertEqual(tracks[0].id, id) } else { id = tracks[0].id }
        }
        let track = tracker.tracks[0]
        XCTAssertTrue(track.isConfirmed)
        XCTAssertEqual(track.velocity.x, 0.8, accuracy: 0.08)
        XCTAssertGreaterThan(track.trajectory.count, 10)
        XCTAssertLessThanOrEqual(track.trajectory.count, 60)
    }

    func testTwoTargetsAndDeletionAfterMisses() {
        let tracker = AlphaBetaTracker()
        for i in 0..<10 {
            let t = Double(i) / 30
            _ = tracker.update(measurements: [
                TargetMeasurement(position: SIMD3(-1, 0, -2), label: "Obstacle", source: .lidarCluster),
                TargetMeasurement(position: SIMD3(1, 0, -2), label: "Obstacle", source: .lidarCluster),
            ], timestamp: t, observer: .zero)
        }
        XCTAssertEqual(tracker.confirmedTracks.count, 2)
        XCTAssertEqual(tracker.tracks.first { $0.position.x < 0 }!.distanceMeters, simd_length(SIMD3<Float>(-1, 0, -2)), accuracy: 0.01)
        for i in 10..<40 { _ = tracker.update(measurements: [], timestamp: Double(i) / 30, observer: .zero) }
        XCTAssertTrue(tracker.tracks.isEmpty)
    }

    func testLabelsAreNotMixed() {
        let tracker = AlphaBetaTracker()
        _ = tracker.update(measurements: [TargetMeasurement(position: .zero, label: "Dog", source: .vision)], timestamp: 0, observer: .zero)
        let tracks = tracker.update(measurements: [TargetMeasurement(position: .zero, label: "Person", source: .vision)], timestamp: 0.1, observer: .zero)
        XCTAssertEqual(Set(tracks.map(\.label)), ["Dog", "Person"])
    }
}

final class VisionGeometryTests: XCTestCase {
    func testOrientationFromGravity() {
        XCTAssertEqual(SensorImageOrientation.from(cameraTransform: matrix_identity_float4x4), .up)
        // Portrait: device rolled so camera +X points down.
        let portrait = simd_float4x4(simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1)))
        XCTAssertEqual(SensorImageOrientation.from(cameraTransform: portrait), .right)
    }

    func testVisionToSensorMapping() {
        let up = SensorImageOrientation.up.sensorPoint(fromVision: SIMD2(0.2, 0.9))
        XCTAssertEqual(up.x, 0.2, accuracy: 1e-6)
        XCTAssertEqual(up.y, 0.1, accuracy: 1e-6)
        // Upright top-left corner of a portrait image is the raw buffer's bottom-left.
        let p = SensorImageOrientation.right.sensorPoint(fromVision: SIMD2(0, 1))
        XCTAssertEqual(p, SIMD2(0, 1))
        let r = SensorImageOrientation.right.sensorRect(fromVision: 0.25, minY: 0.25, width: 0.5, height: 0.5)
        XCTAssertEqual(r.minX, 0.25, accuracy: 1e-6)
        XCTAssertEqual(r.maxY, 0.75, accuracy: 1e-6)
    }
}

final class SyntheticPipelineTests: XCTestCase {
    func testEndToEndDetectsCompactObstaclesAndRejectsSofa() {
        var scene = SyntheticScene()
        scene.cameraMotion = false
        let t = 2.0
        let frame = scene.renderDepthFrame(at: t, pixelStep: 3)
        let pts = PointCloudBuilder.positions(from: frame, options: .init(step: 3))
        XCTAssertGreaterThan(pts.count, 3000)
        var floor = FloorEstimator()
        let floorY = try! XCTUnwrap(floor.update(with: pts))
        XCTAssertEqual(floorY, scene.floorY, accuracy: 0.05)

        let result = VoxelClusterer.cluster(points: pts, floorY: floorY)
        let objects = scene.objects(at: t)
        for name in ["person", "dog", "crate"] {
            let obj = objects.first { $0.name == name }!
            let nearest = result.clusters.map { simd_distance(SIMD2($0.center.x, $0.center.z), SIMD2(obj.center.x, obj.center.z)) }.min() ?? .infinity
            XCTAssertLessThan(nearest, 0.35, "\(name) not detected")
        }
        let sofa = objects.first { $0.name == "sofa" }!
        XCTAssertFalse(result.clusters.contains { simd_distance(SIMD2($0.center.x, $0.center.z), SIMD2(sofa.center.x, sofa.center.z)) < 0.4 })
    }

    func testTrackingMovingPersonOverTime() {
        var scene = SyntheticScene()
        scene.cameraMotion = false
        let tracker = AlphaBetaTracker()
        var floor = FloorEstimator()
        for i in 0..<60 {
            let t = Double(i) / 15
            let frame = scene.renderDepthFrame(at: t, pixelStep: 3)
            let pts = PointCloudBuilder.positions(from: frame, options: .init(step: 3))
            let f = floor.update(with: pts)!
            let clusters = VoxelClusterer.cluster(points: pts, floorY: f).clusters
            tracker.update(measurements: clusters.map { TargetMeasurement(cluster: $0) }, timestamp: t, observer: frame.cameraPosition)
        }
        let person = scene.objects(at: 59.0 / 15).first { $0.name == "person" }!
        let track = tracker.confirmedTracks.min { simd_distance($0.position, person.center) < simd_distance($1.position, person.center) }
        XCTAssertNotNil(track)
        XCTAssertLessThan(simd_distance(SIMD2(track!.position.x, track!.position.z), SIMD2(person.center.x, person.center.z)), 0.3)
        XCTAssertGreaterThan(track!.trajectory.count, 5)
    }
}
