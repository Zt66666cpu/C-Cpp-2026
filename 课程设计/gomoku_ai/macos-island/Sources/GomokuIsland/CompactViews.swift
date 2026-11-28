import SwiftUI

let islandRunningColor = Color(red: 0x34 / 255.0, green: 0xD3 / 255.0, blue: 0x99 / 255.0)
let islandPausedColor = Color(red: 0xFB / 255.0, green: 0xBF / 255.0, blue: 0x24 / 255.0)
let islandOfflineColor = Color(red: 0xF8 / 255.0, green: 0x71 / 255.0, blue: 0x71 / 255.0)
let islandAccentGold = Color(red: 0xE2 / 255.0, green: 0xB6 / 255.0, blue: 0x6D / 255.0)

struct CompactLeadingView: View {
    @ObservedObject var controller: IslandController
    @ObservedObject var poller: MonitorPoller

    init(controller: IslandController) {
        self._controller = ObservedObject(wrappedValue: controller)
        self._poller = ObservedObject(wrappedValue: controller.poller)
    }

    @ViewBuilder
    var body: some View {
        if controller.compactLayout.leading == .hidden || poller.compactState == .hidden {
            EmptyView()
        } else {
            content
                .padding(.trailing, MenuBarSpaceProbe.notchInnerGap)
                .transition(.opacity)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch controller.compactLayout.leading {
        case .hidden:
            EmptyView()
        case .iconOnly:
            compactIconView(for: poller.compactState, showsRoundInRing: showsRoundInRing)
                .transition(.opacity)
        case .iconAndFullText:
            HStack(spacing: 4) {
                compactIconView(for: poller.compactState, showsRoundInRing: showsRoundInRing)
                compactFullTextView(for: poller.compactState)
            }
            .fixedSize()
            .transition(.opacity)
        case .fullTextOnly:
            compactFullTextView(for: poller.compactState)
                .transition(.opacity)
        }
    }

    private var showsRoundInRing: Bool {
        controller.compactLayout.tier == .leftIconOnly
            || (controller.compactLayout.leading == .iconOnly && controller.compactLayout.trailing == .hidden)
    }
}

struct CompactTrailingView: View {
    @ObservedObject var controller: IslandController
    @ObservedObject var poller: MonitorPoller

    init(controller: IslandController) {
        self._controller = ObservedObject(wrappedValue: controller)
        self._poller = ObservedObject(wrappedValue: controller.poller)
    }

    @ViewBuilder
    var body: some View {
        if controller.compactLayout.trailing == .hidden || poller.compactState == .hidden {
            EmptyView()
        } else {
            content
                .padding(.leading, MenuBarSpaceProbe.notchInnerGap)
                .transition(.opacity)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch controller.compactLayout.trailing {
        case .hidden:
            EmptyView()
        case .iconOnly:
            compactIconView(for: poller.compactState)
                .transition(.opacity)
        case .fullText:
            compactFullTextView(for: poller.compactState)
                .transition(.opacity)
        case .shortText:
            compactShortTextView(for: poller.compactState)
                .transition(.opacity)
        }
    }
}

@ViewBuilder
private func compactIconView(
    for state: MonitorCompactState,
    showsRoundInRing: Bool = false
) -> some View {
    switch state {
    case .training(let round, _, _, _, let progress):
        ZStack {
            Circle()
                .inset(by: 1)
                .stroke(Color.white.opacity(0.22), lineWidth: 2)
            Circle()
                .inset(by: 1)
                .trim(from: 0, to: CGFloat(min(1.0, max(0.04, progress))))
                .stroke(islandRunningColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
            if showsRoundInRing {
                Text("\(round)")
                    .font(.system(size: round >= 100 ? 7 : 9, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(islandRunningColor)
                    .minimumScaleFactor(0.6)
                    .lineLimit(1)
                    .frame(maxWidth: 12)
            } else {
                Image(systemName: "cpu.fill")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(islandRunningColor)
            }
        }
        .frame(width: 18, height: 18)
        .animation(.linear(duration: 0.3), value: progress)
    case .paused:
        Image(systemName: "pause.circle.fill")
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(islandPausedColor)
    case .offline:
        Image(systemName: "bolt.slash.fill")
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(islandOfflineColor)
    case .hidden:
        EmptyView()
    }
}

@ViewBuilder
private func compactFullTextView(for state: MonitorCompactState) -> some View {
    switch state {
    case .training(let round, _, _, let heurPct, _):
        if let heurPct {
            Text("R\(round) \(heurPct)%")
                .font(.system(size: 12, weight: .medium))
                .monospacedDigit()
                .lineLimit(1)
                .foregroundStyle(islandRunningColor)
        } else {
            Text("R\(round) 训练")
                .font(.system(size: 12, weight: .medium))
                .monospacedDigit()
                .lineLimit(1)
                .foregroundStyle(.white)
        }
    case .paused(let round, let heurPct):
        if round > 0, let heurPct {
            Text("R\(round) \(heurPct)%")
                .font(.system(size: 12, weight: .medium))
                .monospacedDigit()
                .lineLimit(1)
                .foregroundStyle(islandPausedColor)
        } else {
            Text(round > 0 ? "R\(round) 暂停" : "待机")
                .font(.system(size: 12, weight: .medium))
                .monospacedDigit()
                .lineLimit(1)
                .foregroundStyle(islandPausedColor)
        }
    case .offline:
        Text("离线")
            .font(.system(size: 12, weight: .medium))
            .lineLimit(1)
            .foregroundStyle(islandOfflineColor)
    case .hidden:
        EmptyView()
    }
}

@ViewBuilder
private func compactShortTextView(for state: MonitorCompactState) -> some View {
    switch state {
    case .training(let round, _, _, _, _):
        Text("R\(round)")
            .font(.system(size: 12, weight: .medium))
            .monospacedDigit()
            .lineLimit(1)
            .foregroundStyle(islandRunningColor)
    case .paused:
        Text("暂停")
            .font(.system(size: 12, weight: .medium))
            .lineLimit(1)
            .foregroundStyle(islandPausedColor)
    case .offline:
        Text("离线")
            .font(.system(size: 12, weight: .medium))
            .lineLimit(1)
            .foregroundStyle(islandOfflineColor)
    case .hidden:
        EmptyView()
    }
}
