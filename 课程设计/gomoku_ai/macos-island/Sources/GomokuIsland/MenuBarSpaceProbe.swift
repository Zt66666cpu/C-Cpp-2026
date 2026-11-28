import AppKit
import ApplicationServices
import CoreGraphics

struct MenuBarSpace: Equatable, Sendable {
    let leftAvailable: CGFloat
    let rightAvailable: CGFloat
    let hasAccessibilityPermission: Bool
    let leftmostForeignStatusMinX: CGFloat?
    let overflowExpanded: Bool

    static let unconstrained = MenuBarSpace(
        leftAvailable: 1000,
        rightAvailable: 1000,
        hasAccessibilityPermission: true,
        leftmostForeignStatusMinX: nil,
        overflowExpanded: false
    )
}

struct ScreenNotchGeometry: Sendable {
    let screenFrame: CGRect
    let primaryScreenHeight: CGFloat
    let notchLeft: CGFloat
    let notchRight: CGFloat
    let menuBarHeight: CGFloat

    var screenTopCG: CGFloat {
        primaryScreenHeight - screenFrame.maxY
    }
}

struct CompactLayout: Equatable {
    enum LeadingContent: Equatable {
        case hidden
        case iconOnly
        case iconAndFullText
        case fullTextOnly
    }

    enum TrailingContent: Equatable {
        case hidden
        case iconOnly
        case shortText
        case fullText
    }

    enum Tier: Int, Equatable, CaseIterable {
        case splitFull = 0
        case leftCombined = 1
        case swapped = 2
        case splitShort = 3
        case leftIconOnly = 4
    }

    let leading: LeadingContent
    let trailing: TrailingContent
    let tier: Tier

    static let defaultLayout = CompactLayout(leading: .iconOnly, trailing: .fullText, tier: .splitFull)

    static func resolve(tier: Tier, mouseDodgedSide: MouseDodgeSide) -> CompactLayout {
        let baseLeading: LeadingContent
        let baseTrailing: TrailingContent

        switch tier {
        case .splitFull:
            baseLeading = .iconOnly
            baseTrailing = .fullText
        case .leftCombined:
            baseLeading = .iconAndFullText
            baseTrailing = .hidden
        case .swapped:
            baseLeading = .fullTextOnly
            baseTrailing = .iconOnly
        case .splitShort:
            baseLeading = .iconOnly
            baseTrailing = .shortText
        case .leftIconOnly:
            baseLeading = .iconOnly
            baseTrailing = .hidden
        }

        switch mouseDodgedSide {
        case .none:
            return CompactLayout(leading: baseLeading, trailing: baseTrailing, tier: tier)
        case .left:
            return CompactLayout(leading: .hidden, trailing: baseTrailing, tier: tier)
        case .right:
            return CompactLayout(leading: baseLeading, trailing: .hidden, tier: tier)
        }
    }
}

enum MouseDodgeSide: Equatable {
    case none
    case left
    case right
}

struct CompactStateMetrics {
    let iconWidth: CGFloat
    let fullTextWidth: CGFloat
    let shortTextWidth: CGFloat
    let supportsShortText: Bool
    let combinedSpacing: CGFloat = 4

    var combinedLeftWidth: CGFloat {
        iconWidth + combinedSpacing + fullTextWidth
    }

    static func forState(_ state: MonitorCompactState) -> CompactStateMetrics {
        let fullWidth = CompactTextMeasure.fullTextWidth(for: state)
        switch state {
        case .training:
            return CompactStateMetrics(iconWidth: 20, fullTextWidth: fullWidth, shortTextWidth: 28, supportsShortText: true)
        case .paused:
            return CompactStateMetrics(iconWidth: 18, fullTextWidth: fullWidth, shortTextWidth: 28, supportsShortText: true)
        case .offline:
            return CompactStateMetrics(iconWidth: 18, fullTextWidth: fullWidth, shortTextWidth: 24, supportsShortText: false)
        case .hidden:
            return CompactStateMetrics(iconWidth: 0, fullTextWidth: 0, shortTextWidth: 0, supportsShortText: false)
        }
    }
}

final class MenuBarSpaceProbe {
    static let safetyMargin: CGFloat = 12
    static let librarySideInset: CGFloat = 8 + 6 + notchInnerGap
    static let notchInnerGap: CGFloat = 6
    static let conservativeLeftWidth: CGFloat = 20
    static let hysteresisMargin: CGFloat = 10

    private let queue = DispatchQueue(label: "com.gomokuai.menubar-space-probe", qos: .utility)

    @MainActor
    static func notchGeometry(for screen: NSScreen) -> ScreenNotchGeometry? {
        guard let leftArea = screen.auxiliaryTopLeftArea,
              let rightArea = screen.auxiliaryTopRightArea else {
            return nil
        }
        let notchWidth = max(0, screen.frame.width - leftArea.width - rightArea.width)
        guard notchWidth > 0 else { return nil }

        let notchLeft = screen.frame.midX - notchWidth / 2
        let notchRight = screen.frame.midX + notchWidth / 2
        let menuBarHeight = max(24, max(screen.safeAreaInsets.top, screen.frame.maxY - screen.visibleFrame.maxY))
        let primaryHeight = NSScreen.screens.first(where: { $0.frame.origin == .zero })?.frame.height
            ?? NSScreen.screens.first?.frame.height
            ?? screen.frame.height

        return ScreenNotchGeometry(
            screenFrame: screen.frame,
            primaryScreenHeight: primaryHeight,
            notchLeft: notchLeft,
            notchRight: notchRight,
            menuBarHeight: menuBarHeight
        )
    }

    @MainActor
    func probe(on screen: NSScreen, completion: @escaping @MainActor (MenuBarSpace) -> Void) {
        guard let geometry = Self.notchGeometry(for: screen) else {
            completion(.unconstrained)
            return
        }

        let ownPID = ProcessInfo.processInfo.processIdentifier
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
        let pids = NSWorkspace.shared.runningApplications.map(\.processIdentifier)

        queue.async {
            let space = Self.measureSync(
                geometry: geometry,
                frontmostPID: frontmostPID,
                pids: pids,
                ownPID: ownPID
            )
            Task { @MainActor in
                completion(space)
            }
        }
    }

    static func selectTier(
        for state: MonitorCompactState,
        space: MenuBarSpace,
        currentTier: CompactLayout.Tier,
        hysteresis: CGFloat = hysteresisMargin
    ) -> CompactLayout.Tier {
        let metrics = CompactStateMetrics.forState(state)
        guard metrics.iconWidth > 0 else { return .splitFull }

        func fits(_ tier: CompactLayout.Tier, margin: CGFloat) -> Bool {
            switch tier {
            case .splitFull:
                return space.rightAvailable >= metrics.fullTextWidth + margin
            case .leftCombined:
                return space.leftAvailable >= metrics.combinedLeftWidth + margin
            case .swapped:
                return space.leftAvailable >= metrics.fullTextWidth + margin
                    && space.rightAvailable >= metrics.iconWidth + margin
            case .splitShort:
                return metrics.supportsShortText && space.rightAvailable >= metrics.shortTextWidth + margin
            case .leftIconOnly:
                return true
            }
        }

        for tier in CompactLayout.Tier.allCases where tier.rawValue < currentTier.rawValue {
            if fits(tier, margin: hysteresis) {
                return tier
            }
        }

        if fits(currentTier, margin: 0) {
            return currentTier
        }

        for tier in CompactLayout.Tier.allCases {
            if fits(tier, margin: 0) {
                return tier
            }
        }
        return .leftIconOnly
    }

    private static func measureSync(
        geometry: ScreenNotchGeometry,
        frontmostPID: pid_t,
        pids: [pid_t],
        ownPID: pid_t
    ) -> MenuBarSpace {
        let rightResult = measureRightAvailableAX(geometry: geometry, pids: pids, ownPID: ownPID)
            ?? measureRightAvailable(geometry: geometry, ownPID: ownPID)
        let hasAX = AXIsProcessTrusted()
        let leftAvailable: CGFloat = hasAX
            ? measureLeftAvailable(geometry: geometry, frontmostPID: frontmostPID, ownPID: ownPID)
            : conservativeLeftWidth

        return MenuBarSpace(
            leftAvailable: leftAvailable,
            rightAvailable: rightResult.available,
            hasAccessibilityPermission: hasAX,
            leftmostForeignStatusMinX: rightResult.minStatusX,
            overflowExpanded: rightResult.overflowExpanded
        )
    }

    private static func axFrame(of element: AXUIElement) -> CGRect? {
        var posRef: CFTypeRef?
        var sizeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &posRef) == .success,
              AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeRef) == .success,
              let posRef, let sizeRef,
              CFGetTypeID(posRef) == AXValueGetTypeID(),
              CFGetTypeID(sizeRef) == AXValueGetTypeID() else { return nil }
        var point = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(posRef as! AXValue, .cgPoint, &point),
              AXValueGetValue(sizeRef as! AXValue, .cgSize, &size) else { return nil }
        return CGRect(origin: point, size: size)
    }

    private static func measureRightAvailableAX(
        geometry: ScreenNotchGeometry,
        pids: [pid_t],
        ownPID: pid_t
    ) -> (available: CGFloat, minStatusX: CGFloat?, overflowExpanded: Bool)? {
        guard AXIsProcessTrusted() else { return nil }
        let topCG = geometry.screenTopCG
        let bottomCG = topCG + geometry.menuBarHeight
        var frames: [CGRect] = []
        var leftSideForeignCount = 0

        for pid in pids where pid != ownPID {
            let app = AXUIElementCreateApplication(pid)
            AXUIElementSetMessagingTimeout(app, 0.05)
            var barRef: CFTypeRef?
            guard AXUIElementCopyAttributeValue(app, kAXExtrasMenuBarAttribute as CFString, &barRef) == .success,
                  let barRef, CFGetTypeID(barRef) == AXUIElementGetTypeID() else { continue }
            var kidsRef: CFTypeRef?
            guard AXUIElementCopyAttributeValue(barRef as! AXUIElement, kAXChildrenAttribute as CFString, &kidsRef) == .success,
                  let items = kidsRef as? [AXUIElement] else { continue }
            for item in items {
                guard let f = axFrame(of: item) else { continue }
                if f.width >= 2, f.height >= 2,
                   f.minY >= topCG - 2, f.maxY <= bottomCG + 6,
                   f.maxX <= geometry.notchLeft + 2,
                   f.minX >= geometry.screenFrame.minX {
                    leftSideForeignCount += 1
                    continue
                }
                guard f.width >= 2, f.height >= 2,
                      f.minY >= topCG - 2, f.maxY <= bottomCG + 6,
                      f.maxX > geometry.notchRight + 2,
                      f.minX < geometry.screenFrame.maxX else { continue }
                frames.append(f)
            }
        }
        guard !frames.isEmpty else { return (available: 0, minStatusX: nil, overflowExpanded: leftSideForeignCount > 0) }

        var clusters: [[CGRect]] = []
        for f in frames.sorted(by: { $0.minX < $1.minX }) {
            if let last = clusters.last, let lastMaxX = last.map(\.maxX).max(), f.minX < lastMaxX - 2 {
                clusters[clusters.count - 1].append(f)
            } else {
                clusters.append([f])
            }
        }
        let visibleLeftEdges: [CGFloat] = clusters.map { cluster in
            if cluster.count == 1 { return cluster[0].minX }
            let maxX = cluster.map(\.maxX).max()!
            let chevron = cluster
                .filter { abs($0.maxX - maxX) <= 1.5 }
                .min(by: { $0.width < $1.width })
            return chevron?.minX ?? (maxX - 18)
        }
        let leftmost = max(geometry.notchRight, visibleLeftEdges.min()!)
        let available = leftmost - geometry.notchRight - safetyMargin - librarySideInset
        return (available: max(0, available), minStatusX: leftmost, overflowExpanded: leftSideForeignCount > 0)
    }

    private static func measureRightAvailable(
        geometry: ScreenNotchGeometry,
        ownPID: pid_t
    ) -> (available: CGFloat, minStatusX: CGFloat?, overflowExpanded: Bool) {
        let statusLevel = Int(CGWindowLevelForKey(.statusWindow))
        let topCG = geometry.screenTopCG
        let bottomCG = topCG + geometry.menuBarHeight
        var minStatusX: CGFloat?

        if let windowList = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] {
            for info in windowList {
                guard let layer = info[kCGWindowLayer as String] as? Int, layer == statusLevel else { continue }
                if let ownerPID = info[kCGWindowOwnerPID as String] as? pid_t, ownerPID == ownPID { continue }
                if let alpha = info[kCGWindowAlpha as String] as? Double, alpha <= 0.01 { continue }
                guard let boundsDict = info[kCGWindowBounds as String] as? [String: Any],
                      let bounds = CGRect(dictionaryRepresentation: boundsDict as CFDictionary) else { continue }
                guard bounds.width >= 2, bounds.height >= 2,
                      bounds.minY >= topCG - 2, bounds.maxY <= bottomCG + 6 else { continue }
                guard bounds.minX >= geometry.notchRight - 2,
                      bounds.minX < geometry.screenFrame.maxX else { continue }
                minStatusX = minStatusX.map { min($0, bounds.minX) } ?? bounds.minX
            }
        }

        guard let minStatusX else {
            return (available: 0, minStatusX: nil, overflowExpanded: false)
        }
        let available = minStatusX - geometry.notchRight - safetyMargin - librarySideInset
        return (available: max(0, available), minStatusX: minStatusX, overflowExpanded: false)
    }

    private static func measureLeftAvailable(
        geometry: ScreenNotchGeometry,
        frontmostPID: pid_t,
        ownPID: pid_t
    ) -> CGFloat {
        guard frontmostPID > 0, frontmostPID != ownPID else { return conservativeLeftWidth }
        let appElement = AXUIElementCreateApplication(frontmostPID)
        AXUIElementSetMessagingTimeout(appElement, 0.15)

        var menuBarRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(appElement, kAXMenuBarAttribute as CFString, &menuBarRef) == .success,
              let menuBarObj = menuBarRef,
              CFGetTypeID(menuBarObj) == AXUIElementGetTypeID() else { return conservativeLeftWidth }
        let menuBarElement = menuBarObj as! AXUIElement

        var childrenRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(menuBarElement, kAXChildrenAttribute as CFString, &childrenRef) == .success,
              let children = childrenRef as? [AXUIElement], !children.isEmpty else { return conservativeLeftWidth }

        let topCG = geometry.screenTopCG
        let bottomCG = topCG + geometry.menuBarHeight
        var maxMenuRight: CGFloat?

        for child in children {
            guard let pos = axPoint(child, attribute: kAXPositionAttribute),
                  let size = axSize(child, attribute: kAXSizeAttribute),
                  size.width > 0, size.height > 0 else { continue }
            guard pos.y >= topCG - 4, pos.y <= bottomCG + 4,
                  pos.x >= geometry.screenFrame.minX - 2, pos.x < geometry.notchRight else { continue }
            let itemRight = pos.x + size.width
            maxMenuRight = maxMenuRight.map { max($0, itemRight) } ?? itemRight
        }

        guard let menuRight = maxMenuRight else { return conservativeLeftWidth }
        return max(0, geometry.notchLeft - menuRight - safetyMargin - librarySideInset)
    }

    private static func axPoint(_ element: AXUIElement, attribute: String) -> CGPoint? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &ref) == .success,
              let valueObj = ref, CFGetTypeID(valueObj) == AXValueGetTypeID() else { return nil }
        var point = CGPoint.zero
        guard AXValueGetType(valueObj as! AXValue) == .cgPoint,
              AXValueGetValue(valueObj as! AXValue, .cgPoint, &point) else { return nil }
        return point
    }

    private static func axSize(_ element: AXUIElement, attribute: String) -> CGSize? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &ref) == .success,
              let valueObj = ref, CFGetTypeID(valueObj) == AXValueGetTypeID() else { return nil }
        var size = CGSize.zero
        guard AXValueGetType(valueObj as! AXValue) == .cgSize,
              AXValueGetValue(valueObj as! AXValue, .cgSize, &size) else { return nil }
        return size
    }
}
