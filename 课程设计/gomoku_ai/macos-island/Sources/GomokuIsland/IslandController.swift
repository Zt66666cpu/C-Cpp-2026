import AppKit
import Combine
import SwiftUI
import DynamicNotchKit

enum IslandSurface: String {
    case summary
    case full

    var preferredSize: CGSize {
        switch self {
        case .summary:
            return CGSize(width: 480, height: 540)
        case .full:
            return CGSize(width: 960, height: 620)
        }
    }
}

enum IslandPresentationMode: Equatable {
    case hidden
    case compact
    case prompt(text: String, targetSurface: String)
    case expanded
}

@MainActor
final class IslandController: ObservableObject, WebBridgeDelegate {
    static let hoverExpandDefaultsKey = "gomokuHoverExpandEnabled"

    @Published private(set) var mode: IslandPresentationMode = .hidden
    @Published private(set) var contentMode: IslandPresentationMode = .hidden
    @Published private(set) var surface: IslandSurface = .summary
    @Published private(set) var isFloatingScreen: Bool = false
    @Published private(set) var compactLayout: CompactLayout = .defaultLayout

    var hoverExpandEnabled: Bool {
        get { UserDefaults.standard.object(forKey: Self.hoverExpandDefaultsKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: Self.hoverExpandDefaultsKey) }
    }

    let poller: MonitorPoller
    let webBridge: WebBridge
    var onStateChanged: (() -> Void)?

    private var transitionID = 0
    private var notch: DynamicNotch<ExpandedIslandView, CompactLeadingView, CompactTrailingView>!
    private var cancellables = Set<AnyCancellable>()

    private let spaceProbe = MenuBarSpaceProbe()
    let notchSpacer = NotchSpacerItem()
    private var spacerProbeTimer: Timer?
    private var lastMeasuredSpace: MenuBarSpace = .unconstrained
    private var hasMeasuredSpace = false
    private var committedCompactTier: CompactLayout.Tier = .splitFull
    private var pendingCompactTier: CompactLayout.Tier?
    private var spaceProbeGeneration = 0
    private var spaceProbeTimer: Timer?
    private var layoutDebounceTimer: Timer?
    private var activationFollowUpTimer: Timer?
    private var mouseDodgedSide: MouseDodgeSide = .none
    private var dodgeRestoreTimer: Timer?

    private var isInputFocused = false
    private var pinnedByHotkey = false
    private var expandTimer: Timer?
    private var collapseTimer: Timer?
    private var promptTimer: Timer?
    private var localKeyMonitor: Any?
    private var globalKeyMonitor: Any?
    private var localMouseMonitor: Any?
    private var globalMouseMonitor: Any?
    private var menuBarClickMonitor: Any?
    private var clickProbeTimers: [Timer] = []

    init(poller: MonitorPoller, webBridge: WebBridge) {
        self.poller = poller
        self.webBridge = webBridge
    }

    func start() {
        createNotchIfNeeded()
        applyTransitionConfiguration()
        installKeyMonitors()
        installMouseMonitors()

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screenParametersDidChange),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )

        let workspaceCenter = NSWorkspace.shared.notificationCenter
        for name in [
            NSWorkspace.didActivateApplicationNotification,
            NSWorkspace.didLaunchApplicationNotification,
            NSWorkspace.didTerminateApplicationNotification
        ] {
            workspaceCenter.addObserver(
                self,
                selector: #selector(workspaceApplicationsDidChange(_:)),
                name: name,
                object: nil
            )
        }

        notchSpacer.onClick = { [weak self] in
            self?.expand(surface: .summary, activateKeyboard: false, pin: true)
        }
        notchSpacer.onLengthApplied = { [weak self] in
            self?.scheduleProbeAfterSpacerChange()
        }
        notchSpacer.installIfEnabled()

        updateScreenState()
        refreshCompactOrHidden()
    }

    // MARK: - WebBridgeDelegate

    var currentSurfaceRawValue: String { surface.rawValue }
    var currentSurfaceStyleRawValue: String { isFloatingScreen ? "floating" : "notch" }

    func webBridgeDidRequestSurface(_ surfaceName: String) {
        guard let requested = IslandSurface(rawValue: surfaceName) else { return }
        if requested == .full {
            webBridge.ensureFullMonitorLoaded()
        }
        withAnimation(surfaceAnimation) {
            surface = requested
        }
        applyWindowAppearance()
        onStateChanged?()
    }

    func webBridgeDidRequestCollapse() {
        collapse()
    }

    func webBridgeDidUpdateInputFocus(_ focused: Bool) {
        isInputFocused = focused
        if focused {
            collapseTimer?.invalidate()
            collapseTimer = nil
            focusPanel()
        }
    }

    // MARK: - Screen state

    static func screenHasNotch(_ screen: NSScreen) -> Bool {
        screen.auxiliaryTopLeftArea?.width != nil && screen.auxiliaryTopRightArea?.width != nil
    }

    private func targetScreen() -> NSScreen {
        NSScreen.main ?? NSScreen.screens.first ?? { fatalError("No screen available") }()
    }

    private func updateScreenState() {
        let floating = !Self.screenHasNotch(targetScreen())
        isFloatingScreen = floating
        webBridge.setSurfaceStyle(floating ? "floating" : "notch")

        if floating {
            stopSpaceProbingAndResetLayout()
        }
        if mode == .compact || mode == .hidden {
            refreshCompactOrHidden()
        }
        updateNotchSpacer()
    }

    @objc private func screenParametersDidChange() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.updateScreenState()
            if self.mode == .compact, !self.isFloatingScreen {
                self.probeMenuBarSpace(immediate: false)
            }
        }
    }

    @objc private func workspaceApplicationsDidChange(_ notification: Notification) {
        let isActivation = notification.name == NSWorkspace.didActivateApplicationNotification
        Task { @MainActor [weak self] in
            guard let self, self.mode == .compact, !self.isFloatingScreen else { return }
            self.probeMenuBarSpace(immediate: false)

            if isActivation {
                self.activationFollowUpTimer?.invalidate()
                self.activationFollowUpTimer = Timer(timeInterval: 0.15, repeats: false) { [weak self] _ in
                    Task { @MainActor in
                        guard let self, self.mode == .compact, !self.isFloatingScreen else { return }
                        self.activationFollowUpTimer = nil
                        self.probeMenuBarSpace(immediate: false)
                    }
                }
                RunLoop.main.add(self.activationFollowUpTimer!, forMode: .common)
            }
        }
    }

    // MARK: - Appearance and animation

    static func panelSize(for screen: NSScreen) -> CGSize {
        let surfaces: [IslandSurface] = [.summary, .full]
        let maxW = surfaces.map(\.preferredSize.width).max() ?? 960
        let maxH = surfaces.map(\.preferredSize.height).max() ?? 620
        let contentW = min(maxW, screen.visibleFrame.width)
        let contentH = min(maxH, screen.visibleFrame.height)
        let notchH = max(screen.safeAreaInsets.top, screen.frame.maxY - screen.visibleFrame.maxY)
        let shadowMargin: CGFloat = 24
        let floatingPadding: CGFloat = screenHasNotch(screen) ? 0 : 40
        return CGSize(
            width: contentW + 15 * 2 + 15 * 2 + shadowMargin * 2,
            height: notchH + contentH + 15 + shadowMargin + floatingPadding
        )
    }

    private var surfaceAnimation: Animation {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            ? .easeInOut(duration: 0.20)
            : .spring(response: 0.42, dampingFraction: 0.80)
    }

    private var collapseAnimation: Animation {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            ? .easeInOut(duration: 0.20)
            : .easeOut(duration: 0.30)
    }

    private var collapseAnimationDuration: Double {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion ? 0.20 : 0.30
    }

    private func applyTransitionConfiguration() {
        let reduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        if reduceMotion {
            let eased = Animation.easeInOut(duration: 0.20)
            notch.transitionConfiguration = DynamicNotchTransitionConfiguration(
                openingAnimation: eased,
                closingAnimation: eased,
                conversionAnimation: eased,
                skipIntermediateHides: true
            )
        } else {
            notch.transitionConfiguration = DynamicNotchTransitionConfiguration(
                openingAnimation: .spring(response: 0.42, dampingFraction: 0.80),
                closingAnimation: .easeOut(duration: 0.30),
                conversionAnimation: .spring(response: 0.42, dampingFraction: 0.80),
                skipIntermediateHides: true
            )
        }
    }

    private func applyWindowAppearance() {
        notch.windowController?.window?.appearance = NSAppearance(named: .darkAqua)
    }

    private func focusPanel() {
        guard let panel = notch.windowController?.window else { return }
        NSApp.activate(ignoringOtherApps: true)
        panel.makeKey()
    }

    private func createNotchIfNeeded() {
        guard notch == nil else { return }
        let notch = DynamicNotch(
            hoverBehavior: [],
            style: .auto,
            expanded: { ExpandedIslandView(controller: self) },
            compactLeading: { CompactLeadingView(controller: self) },
            compactTrailing: { CompactTrailingView(controller: self) }
        )
        self.notch = notch
        notch.panelSize = { screen in IslandController.panelSize(for: screen) }

        notch.$isHovering
            .removeDuplicates()
            .receive(on: RunLoop.main)
            .sink { [weak self] hovering in
                self?.handleHoverChange(hovering)
            }
            .store(in: &cancellables)

        applyWindowAppearance()
    }

    // MARK: - Mouse / Hover

    private func installMouseMonitors() {
        guard localMouseMonitor == nil, globalMouseMonitor == nil else { return }
        let mask: NSEvent.EventTypeMask = [.mouseMoved, .leftMouseDragged, .otherMouseDragged]

        localMouseMonitor = NSEvent.addLocalMonitorForEvents(matching: mask) { [weak self] event in
            Task { @MainActor in self?.handleMouseLocation(NSEvent.mouseLocation) }
            return event
        }
        globalMouseMonitor = NSEvent.addGlobalMonitorForEvents(matching: mask) { [weak self] _ in
            Task { @MainActor in self?.handleMouseLocation(NSEvent.mouseLocation) }
        }
        menuBarClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseUp]) { [weak self] _ in
            Task { @MainActor in self?.handleMenuBarClick(NSEvent.mouseLocation) }
        }
    }

    private func handleMenuBarClick(_ point: NSPoint) {
        guard mode == .compact, !isFloatingScreen else { return }
        let screen = screen(containing: point) ?? targetScreen()
        guard let g = MenuBarSpaceProbe.notchGeometry(for: screen),
              point.y >= screen.frame.maxY - g.menuBarHeight - 2 else { return }
        clickProbeTimers.forEach { $0.invalidate() }
        clickProbeTimers = [0.35, 1.0].map { delay in
            let t = Timer(timeInterval: delay, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.mode == .compact, !self.isFloatingScreen else { return }
                    self.probeMenuBarSpace(immediate: false)
                }
            }
            RunLoop.main.add(t, forMode: .common)
            return t
        }
    }

    private func screen(containing point: NSPoint) -> NSScreen? {
        NSScreen.screens.first { $0.frame.contains(point) }
    }

    private func notchHoverRect(on screen: NSScreen) -> NSRect {
        if let leftArea = screen.auxiliaryTopLeftArea,
           let rightArea = screen.auxiliaryTopRightArea {
            let notchWidth = max(160, screen.frame.width - leftArea.width - rightArea.width)
            let notchHeight = max(screen.safeAreaInsets.top, screen.frame.maxY - screen.visibleFrame.maxY)
            let width = notchWidth + 40
            let height = notchHeight + 8
            return NSRect(
                x: screen.frame.midX - width / 2,
                y: screen.frame.maxY - height,
                width: width,
                height: height
            )
        }
        let menubarHeight = max(24, screen.frame.maxY - screen.visibleFrame.maxY)
        let width: CGFloat = 200
        let height = menubarHeight + 38
        return NSRect(
            x: screen.frame.midX - width / 2,
            y: screen.frame.maxY - height,
            width: width,
            height: height
        )
    }

    private func expandedHoverRect(on screen: NSScreen) -> NSRect {
        let preferred = surface.preferredSize
        let size = CGSize(
            width: min(preferred.width, screen.visibleFrame.width),
            height: min(preferred.height, screen.visibleFrame.height)
        )
        let notchHeight = max(screen.safeAreaInsets.top, screen.frame.maxY - screen.visibleFrame.maxY)
        let width = size.width + 48
        let height = notchHeight + size.height + 44
        return NSRect(
            x: screen.frame.midX - width / 2,
            y: screen.frame.maxY - height,
            width: width,
            height: height
        )
    }

    private func handleMouseLocation(_ point: NSPoint) {
        let screen = screen(containing: point) ?? targetScreen()

        if mode == .compact, !isFloatingScreen {
            let dodgingSide = updateMouseDodge(for: point, on: screen)
            if dodgingSide != .none {
                expandTimer?.invalidate()
                expandTimer = nil
                return
            }
        }

        if mode == .hidden || mode == .compact || isPromptMode {
            guard hoverExpandEnabled else {
                expandTimer?.invalidate()
                expandTimer = nil
                return
            }
            guard notchHoverRect(on: screen).contains(point) else {
                expandTimer?.invalidate()
                expandTimer = nil
                return
            }
            collapseTimer?.invalidate()
            collapseTimer = nil

            guard expandTimer == nil else { return }
            expandTimer = Timer(timeInterval: 0.22, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    self.expandTimer = nil
                    guard self.hoverExpandEnabled,
                          self.mode == .hidden || self.mode == .compact || self.isPromptMode else { return }
                    let current = NSEvent.mouseLocation
                    let currentScreen = self.screen(containing: current) ?? self.targetScreen()
                    guard self.detectMouseDodgeSide(at: current, on: currentScreen) == .none else { return }
                    guard self.notchHoverRect(on: currentScreen).contains(current) else { return }
                    self.expand(surface: .summary, activateKeyboard: false, pin: false)
                }
            }
            RunLoop.main.add(expandTimer!, forMode: .common)
            return
        }

        if mode == .expanded {
            if expandedHoverRect(on: screen).contains(point) || notch.isHovering {
                collapseTimer?.invalidate()
                collapseTimer = nil
                return
            }
            guard !isInputFocused, !pinnedByHotkey else { return }
            guard collapseTimer == nil else { return }
            collapseTimer = Timer(timeInterval: 0.32, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    self.collapseTimer = nil
                    guard self.mode == .expanded else { return }
                    let current = NSEvent.mouseLocation
                    let currentScreen = self.screen(containing: current) ?? self.targetScreen()
                    guard !self.expandedHoverRect(on: currentScreen).contains(current) else { return }
                    guard !self.isInputFocused, !self.pinnedByHotkey else { return }
                    self.collapse()
                }
            }
            RunLoop.main.add(collapseTimer!, forMode: .common)
        }
    }

    private func handleHoverChange(_ hovering: Bool) {
        _ = hovering
        handleMouseLocation(NSEvent.mouseLocation)
    }

    // MARK: - Menu bar space probing & mouse dodge

    private func startSpaceProbingIfNeeded() {
        guard mode == .compact, !isFloatingScreen else { return }
        if spaceProbeTimer == nil {
            let timer = Timer(timeInterval: 4.0, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.mode == .compact, !self.isFloatingScreen else { return }
                    self.probeMenuBarSpace(immediate: false)
                }
            }
            RunLoop.main.add(timer, forMode: .common)
            spaceProbeTimer = timer
        }
    }

    private func stopSpaceProbingAndResetLayout() {
        spaceProbeGeneration += 1
        spaceProbeTimer?.invalidate()
        spaceProbeTimer = nil
        layoutDebounceTimer?.invalidate()
        layoutDebounceTimer = nil
        activationFollowUpTimer?.invalidate()
        activationFollowUpTimer = nil
        dodgeRestoreTimer?.invalidate()
        dodgeRestoreTimer = nil
        spacerProbeTimer?.invalidate()
        spacerProbeTimer = nil
        clickProbeTimers.forEach { $0.invalidate() }
        clickProbeTimers = []
        pendingCompactTier = nil
        let wasDodging = (mouseDodgedSide != .none)
        mouseDodgedSide = .none

        if isFloatingScreen {
            committedCompactTier = .splitFull
            applyEffectiveCompactLayout(animated: false)
        } else if wasDodging {
            applyEffectiveCompactLayout(animated: false)
        }
    }

    private func probeMenuBarSpace(immediate: Bool) {
        guard mode == .compact, !isFloatingScreen else { return }
        spaceProbeGeneration += 1
        let generation = spaceProbeGeneration
        let screen = targetScreen()

        spaceProbe.probe(on: screen) { [weak self] space in
            guard let self,
                  generation == self.spaceProbeGeneration,
                  self.mode == .compact,
                  !self.isFloatingScreen else { return }

            let isFirstProbe = !self.hasMeasuredSpace
            self.hasMeasuredSpace = true
            let overflowChanged = space.overflowExpanded != self.lastMeasuredSpace.overflowExpanded
            self.lastMeasuredSpace = space

            if let geometry = MenuBarSpaceProbe.notchGeometry(for: screen) {
                self.notchSpacer.ensureLeftmost(foreignLeftmostMinX: space.leftmostForeignStatusMinX, geometry: geometry)
            }

            let candidateTier = MenuBarSpaceProbe.selectTier(
                for: self.poller.compactState,
                space: space,
                currentTier: self.committedCompactTier
            )

            if immediate || isFirstProbe {
                self.layoutDebounceTimer?.invalidate()
                self.layoutDebounceTimer = nil
                self.pendingCompactTier = nil
                self.committedCompactTier = candidateTier
                self.applyEffectiveCompactLayout(animated: !isFirstProbe)
            } else {
                self.scheduleTierTransitionIfNeeded(to: candidateTier)
            }

            if overflowChanged {
                self.applyEffectiveCompactLayout(animated: true, fast: true)
            }
        }
    }

    private func scheduleTierTransitionIfNeeded(to candidateTier: CompactLayout.Tier) {
        if candidateTier == committedCompactTier {
            layoutDebounceTimer?.invalidate()
            layoutDebounceTimer = nil
            pendingCompactTier = nil
            return
        }
        guard pendingCompactTier != candidateTier else { return }
        layoutDebounceTimer?.invalidate()
        pendingCompactTier = candidateTier

        layoutDebounceTimer = Timer(timeInterval: 0.5, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.layoutDebounceTimer = nil
                guard self.mode == .compact, !self.isFloatingScreen,
                      let targetTier = self.pendingCompactTier else {
                    self.pendingCompactTier = nil
                    return
                }
                self.pendingCompactTier = nil
                self.committedCompactTier = targetTier
                self.applyEffectiveCompactLayout(animated: true)
            }
        }
        RunLoop.main.add(layoutDebounceTimer!, forMode: .common)
    }

    private func recomputeCompactTierForStateChange() {
        guard !isFloatingScreen else {
            committedCompactTier = .splitFull
            applyEffectiveCompactLayout(animated: false)
            return
        }
        guard hasMeasuredSpace else { return }
        let candidateTier = MenuBarSpaceProbe.selectTier(
            for: poller.compactState,
            space: lastMeasuredSpace,
            currentTier: committedCompactTier
        )
        layoutDebounceTimer?.invalidate()
        layoutDebounceTimer = nil
        pendingCompactTier = nil
        committedCompactTier = candidateTier
        applyEffectiveCompactLayout(animated: true)
    }

    private func applyEffectiveCompactLayout(animated: Bool, fast: Bool = false) {
        let next: CompactLayout
        if isFloatingScreen {
            next = .defaultLayout
        } else {
            var resolved = CompactLayout.resolve(tier: committedCompactTier, mouseDodgedSide: mouseDodgedSide)
            if lastMeasuredSpace.overflowExpanded {
                resolved = CompactLayout(leading: .hidden, trailing: .hidden, tier: committedCompactTier)
            }
            next = resolved
        }
        guard next != compactLayout else { return }

        if animated {
            let reduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            let duration: Double = reduceMotion ? 0.12 : (fast ? 0.16 : 0.22)
            withAnimation(.easeInOut(duration: duration)) {
                compactLayout = next
            }
        } else {
            compactLayout = next
        }
    }

    private func detectMouseDodgeSide(at point: NSPoint, on screen: NSScreen) -> MouseDodgeSide {
        guard mode == .compact, !isFloatingScreen,
              let geometry = MenuBarSpaceProbe.notchGeometry(for: screen) else { return .none }
        let inMenuBarY = point.y >= screen.frame.maxY - geometry.menuBarHeight - 2
            && point.y <= screen.frame.maxY + 2
        guard inMenuBarY else { return .none }
        let edgeClearance: CGFloat = 4
        let dodgeReach: CGFloat = 150
        if point.x > geometry.notchRight + edgeClearance && point.x <= geometry.notchRight + dodgeReach {
            return .right
        }
        if point.x < geometry.notchLeft - edgeClearance && point.x >= geometry.notchLeft - dodgeReach {
            return .left
        }
        return .none
    }

    private func updateMouseDodge(for point: NSPoint, on screen: NSScreen) -> MouseDodgeSide {
        let detected = detectMouseDodgeSide(at: point, on: screen)
        if detected != .none {
            dodgeRestoreTimer?.invalidate()
            dodgeRestoreTimer = nil
            if mouseDodgedSide != detected {
                mouseDodgedSide = detected
                applyEffectiveCompactLayout(animated: true, fast: true)
            }
        } else if mouseDodgedSide != .none, dodgeRestoreTimer == nil {
            dodgeRestoreTimer = Timer(timeInterval: 1.0, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    self.dodgeRestoreTimer = nil
                    guard self.mode == .compact, !self.isFloatingScreen else {
                        self.mouseDodgedSide = .none
                        return
                    }
                    let current = NSEvent.mouseLocation
                    let currentScreen = self.screen(containing: current) ?? self.targetScreen()
                    let currentDetected = self.detectMouseDodgeSide(at: current, on: currentScreen)
                    if currentDetected == .none {
                        self.restoreMouseDodgeWithFreshProbe()
                    } else if self.mouseDodgedSide != currentDetected {
                        self.mouseDodgedSide = currentDetected
                        self.applyEffectiveCompactLayout(animated: true, fast: true)
                    }
                }
            }
            RunLoop.main.add(dodgeRestoreTimer!, forMode: .common)
        }
        return detected
    }

    private func restoreMouseDodgeWithFreshProbe() {
        guard mode == .compact, !isFloatingScreen else {
            mouseDodgedSide = .none
            return
        }
        spaceProbeGeneration += 1
        let generation = spaceProbeGeneration
        spaceProbe.probe(on: targetScreen()) { [weak self] space in
            guard let self,
                  generation == self.spaceProbeGeneration,
                  self.mode == .compact,
                  !self.isFloatingScreen else { return }
            let current = NSEvent.mouseLocation
            let currentScreen = self.screen(containing: current) ?? self.targetScreen()
            guard self.detectMouseDodgeSide(at: current, on: currentScreen) == .none else { return }

            self.hasMeasuredSpace = true
            self.lastMeasuredSpace = space
            self.layoutDebounceTimer?.invalidate()
            self.layoutDebounceTimer = nil
            self.pendingCompactTier = nil
            self.committedCompactTier = MenuBarSpaceProbe.selectTier(
                for: self.poller.compactState,
                space: space,
                currentTier: self.committedCompactTier
            )
            self.mouseDodgedSide = .none
            self.applyEffectiveCompactLayout(animated: true)
        }
    }

    private func updateNotchSpacer() {
        guard notchSpacer.isEnabled else { return }
        if isFloatingScreen || mode == .hidden || poller.compactState == .hidden {
            notchSpacer.requestLength(0)
            return
        }
        guard mode == .compact else { return }
        let textWidth = CompactTextMeasure.fullTextWidth(for: poller.compactState)
        notchSpacer.requestLength(textWidth > 0 ? textWidth + NotchSpacerItem.extraWidth : 0)
    }

    func refreshNotchSpacer() {
        updateNotchSpacer()
    }

    private func scheduleProbeAfterSpacerChange() {
        spacerProbeTimer?.invalidate()
        let timer = Timer(timeInterval: 0.3, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.spacerProbeTimer = nil
                guard self.mode == .compact, !self.isFloatingScreen else { return }
                self.probeMenuBarSpace(immediate: false)
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        spacerProbeTimer = timer
    }

    // MARK: - Presentation transitions

    func refreshCompactOrHidden() {
        guard mode != .expanded, !isPromptMode else { return }
        applyCompactOrHidden()
    }

    private func enterPresentedMode(_ newMode: IslandPresentationMode) {
        stopSpaceProbingAndResetLayout()
        transitionID += 1
        mode = newMode
        contentMode = newMode
    }

    private func applyCompactOrHidden() {
        expandTimer?.invalidate()
        expandTimer = nil

        transitionID += 1
        let id = transitionID
        let wasShowingWeb = (contentMode == .expanded)
        let target: IslandPresentationMode = (poller.compactState == .hidden) ? .hidden : .compact
        let screen = targetScreen()

        mode = target

        if target == .compact, !isFloatingScreen {
            recomputeCompactTierForStateChange()
            startSpaceProbingIfNeeded()
            probeMenuBarSpace(immediate: !hasMeasuredSpace)
        } else {
            stopSpaceProbingAndResetLayout()
        }
        updateNotchSpacer()

        Task { @MainActor [weak self] in
            guard let self else { return }
            if wasShowingWeb {
                self.webBridge.fadeWebViews(to: 0, duration: 0.22)
            }
            if target == .hidden {
                await self.notch.hide()
            } else if self.isFloatingScreen {
                if self.contentMode != .compact {
                    let animatedFromWeb = (self.contentMode == .expanded)
                    withAnimation(self.collapseAnimation) {
                        self.contentMode = .compact
                    }
                    if animatedFromWeb {
                        try? await Task.sleep(for: .seconds(self.collapseAnimationDuration))
                    }
                }
                await self.notch.expand(on: screen)
            } else {
                await self.notch.compact(on: screen)
            }

            guard id == self.transitionID else { return }
            self.contentMode = target
            if wasShowingWeb {
                self.webBridge.attachToKeepAliveWindow()
                self.webBridge.fadeWebViews(to: 1, duration: 0)
            }
            self.applyWindowAppearance()
        }
        onStateChanged?()
    }

    func expand(surface: IslandSurface, activateKeyboard: Bool, pin: Bool) {
        expandTimer?.invalidate()
        expandTimer = nil
        collapseTimer?.invalidate()
        collapseTimer = nil
        promptTimer?.invalidate()
        promptTimer = nil

        self.surface = surface
        pinnedByHotkey = pin
        if surface == .full {
            webBridge.ensureFullMonitorLoaded()
        }
        enterPresentedMode(.expanded)
        webBridge.fadeWebViews(to: 1, duration: 0)

        let screen = targetScreen()
        Task { @MainActor [weak self] in
            guard let self else { return }
            let expandTask = Task { await self.notch.expand(on: screen) }
            await Task.yield()
            self.applyWindowAppearance()
            if activateKeyboard { self.focusPanel() }
            await expandTask.value
            self.applyWindowAppearance()
            if activateKeyboard { self.focusPanel() }
        }
        onStateChanged?()
    }

    func collapse() {
        expandTimer?.invalidate()
        expandTimer = nil
        collapseTimer?.invalidate()
        collapseTimer = nil
        promptTimer?.invalidate()
        promptTimer = nil
        pinnedByHotkey = false
        isInputFocused = false
        applyCompactOrHidden()
    }

    func toggleFromHotkeyOrMenu() {
        if mode == .expanded {
            collapse()
        } else {
            expand(surface: .summary, activateKeyboard: false, pin: true)
        }
    }

    func openFullMonitorInIsland() {
        expand(surface: .full, activateKeyboard: true, pin: true)
    }

    func showPrompt(text: String, targetSurface: String) {
        guard mode != .expanded else { return }
        let screen = targetScreen()
        guard !(screen.frame.height - screen.visibleFrame.height <= 1) else { return }

        promptTimer?.invalidate()
        promptTimer = nil
        collapseTimer?.invalidate()
        collapseTimer = nil
        expandTimer?.invalidate()
        expandTimer = nil

        enterPresentedMode(.prompt(text: text, targetSurface: targetSurface))

        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.notch.expand(on: screen)
            self.applyWindowAppearance()
        }

        promptTimer = Timer(timeInterval: 4.0, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.promptTimer = nil
                guard self.isPromptMode else { return }
                self.mode = .hidden
                self.refreshCompactOrHidden()
            }
        }
        RunLoop.main.add(promptTimer!, forMode: .common)
    }

    private var isPromptMode: Bool {
        if case .prompt = mode { return true }
        return false
    }

    private func installKeyMonitors() {
        guard localKeyMonitor == nil, globalKeyMonitor == nil else { return }
        localKeyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, event.keyCode == 53, self.mode == .expanded else { return event }
            self.handleEscapeKey()
            return nil
        }
        globalKeyMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard event.keyCode == 53 else { return }
            Task { @MainActor in
                guard let self, self.mode == .expanded else { return }
                self.handleEscapeKey()
            }
        }
    }

    private func handleEscapeKey() {
        guard mode == .expanded else { return }
        if surface == .full {
            withAnimation(surfaceAnimation) {
                surface = .summary
            }
            onStateChanged?()
        } else {
            collapse()
        }
    }
}

struct ExpandedIslandView: View {
    @ObservedObject var controller: IslandController

    var body: some View {
        Group {
            switch controller.contentMode {
            case .compact:
                HStack(spacing: 8) {
                    CompactLeadingView(controller: controller)
                    CompactTrailingView(controller: controller)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .contentShape(Rectangle())
                .onTapGesture {
                    controller.expand(surface: .summary, activateKeyboard: false, pin: true)
                }

            case .prompt(let text, let targetSurface):
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(islandRunningColor)
                    Text(text)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
                .onTapGesture {
                    let target = IslandSurface(rawValue: targetSurface) ?? .summary
                    controller.expand(surface: target, activateKeyboard: false, pin: true)
                }

            case .expanded:
                let size = clampedSize(for: controller.surface)
                if controller.surface == .full {
                    VStack(spacing: 0) {
                        HStack(spacing: 10) {
                            Button {
                                controller.webBridgeDidRequestSurface("summary")
                            } label: {
                                Label("精简面板", systemImage: "chevron.left")
                                    .font(.system(size: 12, weight: .medium))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(islandAccentGold)

                            Spacer()
                            Text("五子棋 AI 完整监控台 · 127.0.0.1:8766")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.white.opacity(0.85))
                            Spacer()

                            Button {
                                controller.webBridge.reloadFullMonitor()
                            } label: {
                                Image(systemName: "arrow.clockwise")
                                    .font(.system(size: 12))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.white.opacity(0.75))

                            Button {
                                if let url = URL(string: controller.poller.monitorBaseURL) {
                                    NSWorkspace.shared.open(url)
                                }
                            } label: {
                                Image(systemName: "safari")
                                    .font(.system(size: 12))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.white.opacity(0.75))

                            Button {
                                controller.collapse()
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 13))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.white.opacity(0.65))
                        }
                        .padding(.horizontal, 12)
                        .frame(height: 34)
                        .background(Color.black)

                        WebViewHostRepresentable(bridge: controller.webBridge, kind: .full)
                            .frame(width: size.width, height: max(200, size.height - 34))
                            .clipped()
                    }
                    .frame(width: size.width, height: size.height)
                } else {
                    WebViewHostRepresentable(bridge: controller.webBridge, kind: .summary)
                        .frame(width: size.width, height: size.height)
                        .clipped()
                }

            case .hidden:
                EmptyView()
            }
        }
    }

    private func clampedSize(for surface: IslandSurface) -> CGSize {
        let preferred = surface.preferredSize
        guard let visibleFrame = NSScreen.main?.visibleFrame else { return preferred }
        return CGSize(
            width: min(preferred.width, visibleFrame.width),
            height: min(preferred.height, visibleFrame.height)
        )
    }
}
