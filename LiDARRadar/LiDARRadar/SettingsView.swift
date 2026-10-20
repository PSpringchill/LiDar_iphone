import SwiftUI
import RadarCore

struct SettingsView: View {
    @ObservedObject var engine: LiDARRadarEngine
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Source", selection: Binding(get: { engine.mode }, set: { engine.setMode($0) })) {
                        ForEach(RadarSourceMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(!engine.isLiDARAvailable)
                } header: {
                    Text("Sensor")
                } footer: {
                    Text(engine.isLiDARAvailable ? "Simulation renders a synthetic room with a walking person and a dog." : "This device has no LiDAR scanner, so only the synthetic room is available.")
                }

                Section("Detection") {
                    Toggle("LiDAR Euclidean clustering", isOn: $engine.settings.lidarClusteringEnabled)
                    Toggle("Vision people / animal fusion", isOn: $engine.settings.visionEnabled)
                        .disabled(engine.mode == .simulation)
                }

                Section {
                    ParamSlider(title: "Voxel size", value: $engine.settings.clustering.voxelSize, range: 0.03...0.15, format: "%.2f m")
                    ParamSlider(title: "Cluster tolerance", value: $engine.settings.clustering.clusterTolerance, range: 0.05...0.30, format: "%.2f m")
                    ParamSlider(title: "Min points / cluster", value: Binding(
                        get: { Float(engine.settings.clustering.minPointsPerCluster) },
                        set: { engine.settings.clustering.minPointsPerCluster = Int($0.rounded()) }), range: 5...100, format: "%.0f")
                    ParamSlider(title: "Floor clearance", value: $engine.settings.clustering.floorClearanceMeters, range: 0.02...0.40, format: "%.2f m")
                    ParamSlider(title: "Max wall span", value: $engine.settings.clustering.maxWallSpanMeters, range: 0.5...4.0, format: "%.1f m")
                } header: {
                    Text("Clustering")
                } footer: {
                    Text("Smaller voxels separate small objects; larger voxels merge limbs into one cluster. Clusters wider than the wall span are treated as walls / large furniture.")
                }

                Section("Tracker (α-β)") {
                    ParamSlider(title: "α position gain", value: $engine.settings.tracker.alpha, range: 0.1...0.9, format: "%.2f")
                    ParamSlider(title: "β velocity gain", value: $engine.settings.tracker.beta, range: 0.01...0.5, format: "%.2f")
                    ParamSlider(title: "Gate radius", value: $engine.settings.tracker.gateRadius, range: 0.2...1.5, format: "%.2f m")
                }

                Section("Display") {
                    Stepper("Point step: \(engine.settings.pointStep) (\(49_152 / (engine.settings.pointStep * engine.settings.pointStep)) pts max)",
                            value: $engine.settings.pointStep, in: 1...4)
                    Picker("Color by", selection: $engine.settings.colorMode) {
                        ForEach(PointColorMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Toggle("Accumulate world map", isOn: $engine.settings.accumulateMap)
                    Toggle("Show cluster boxes", isOn: $engine.settings.showClusterBoxes)
                }

                Section {
                    Button("Reset map & tracks") { engine.resetTracking() }
                    Button("Restore defaults", role: .destructive) {
                        let vision = engine.settings.visionEnabled
                        engine.settings = RadarSettings()
                        engine.settings.visionEnabled = vision
                    }
                }
            }
            .navigationTitle("Radar Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}

private struct ParamSlider: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value)).font(.system(.body, design: .monospaced)).foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range)
        }
    }
}
