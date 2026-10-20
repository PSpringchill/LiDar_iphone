import SwiftUI
import RadarCore

struct ContentView: View {
    @StateObject private var engine = LiDARRadarEngine()
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewMode: RadarViewMode = .follow
    @State private var showSettings = false
    @State private var recenterToken = 0

    var body: some View {
        ZStack {
            TacticalRadarSceneView(snapshot: engine.snapshot, viewMode: viewMode,
                                   showClusterBoxes: engine.settings.showClusterBoxes, recenterToken: recenterToken)
                .ignoresSafeArea()

            VStack(spacing: 10) {
                HStack(alignment: .top) {
                    telemetryPanel
                    Spacer(minLength: 8)
                    RadarScopeView(targets: engine.snapshot.targets, cameraTransform: engine.snapshot.cameraTransform)
                        .frame(width: 128, height: 128)
                }
                if let status = engine.statusMessage {
                    Text(status)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.yellow)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
                }
                Spacer()
                targetList
                controlBar
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .onAppear { engine.start() }
        .onDisappear { engine.stop() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { engine.start() } else if phase == .background { engine.stop() }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(engine: engine)
                .presentationDetents([.medium, .large])
        }
    }

    private var telemetryPanel: some View {
        let s = engine.snapshot
        return VStack(alignment: .leading, spacing: 4) {
            Text("3D LiDAR RADAR FUSION")
                .font(.system(.headline, design: .monospaced))
                .foregroundColor(.green)
            HStack(spacing: 6) {
                Text(engine.mode == .lidar ? "● LIDAR" : "● SIMULATION")
                    .foregroundColor(engine.mode == .lidar ? .green : .orange)
                Text(String(format: "%.0f FPS", s.fps))
                Text(String(format: "%.1f ms", s.processingMs))
            }
            .font(.system(.caption2, design: .monospaced))
            .foregroundColor(.white.opacity(0.8))
            Text("Points: \(s.livePoints.count)  Map: \(s.mapPoints.count)")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.cyan)
            Text(String(format: "Clusters: %d  Targets: %d  Floor: %.2fm", s.clusters.count, s.targets.count, s.floorY))
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
        }
        .lineLimit(1)
        .minimumScaleFactor(0.7)
        .padding(10)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
    }

    private var targetList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(engine.snapshot.targets.prefix(5)) { target in
                HStack(spacing: 8) {
                    Circle()
                        .fill(target.source == .vision ? Color.orange : Color.pink)
                        .frame(width: 10, height: 10)
                    Text(target.displayName)
                        .foregroundColor(.white)
                    Spacer(minLength: 4)
                    Text(String(format: "%.2fm", target.distanceMeters))
                        .foregroundColor(.green)
                    Text(String(format: "%.1fm/s", target.speed))
                        .foregroundColor(.yellow)
                    Text(String(format: "(%.1f, %.1f, %.1f)", target.position.x, target.position.y, target.position.z))
                        .foregroundColor(.white.opacity(0.6))
                }
                .font(.system(.caption, design: .monospaced))
                .lineLimit(1)
                .padding(6)
                .background(Color.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private var controlBar: some View {
        HStack(spacing: 10) {
            Picker("View", selection: $viewMode) {
                ForEach(RadarViewMode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            Button { recenterToken += 1 } label: { Image(systemName: "scope") }
                .accessibilityLabel("Recenter view")
            Button { engine.resetTracking() } label: { Image(systemName: "arrow.counterclockwise") }
                .accessibilityLabel("Reset map and tracks")
            Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
                .accessibilityLabel("Settings")
        }
        .font(.title3)
        .buttonStyle(.bordered)
        .tint(.green)
        .padding(8)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
    }
}
