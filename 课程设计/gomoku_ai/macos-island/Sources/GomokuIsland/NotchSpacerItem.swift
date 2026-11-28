import AppKit

@MainActor
final class NotchSpacerItem: NSObject {
    static let autosaveName = "GomokuIslandNotchSpacer"
    static let enabledDefaultsKey = "gomokuReserveNotchSpace"
    static var preferredPositionKey: String { "NSStatusItem Preferred Position \(autosaveName)" }

    static var extraWidth: CGFloat {
        MenuBarSpaceProbe.safetyMargin + MenuBarSpaceProbe.librarySideInset + MenuBarSpaceProbe.hysteresisMargin
    }

    private var statusItem: NSStatusItem?
    private(set) var appliedLength: CGFloat = 0
    private var pendingLength: CGFloat?
    private var debounceTimer: Timer?
    private var lastRepositionAt: Date = .distantPast

    var onClick: (() -> Void)?
    var onLengthApplied: (() -> Void)?

    var isEnabled: Bool {
        get { UserDefaults.standard.object(forKey: Self.enabledDefaultsKey) as? Bool ?? true }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.enabledDefaultsKey)
            if newValue { install() } else { uninstall() }
            onLengthApplied?()
        }
    }

    func installIfEnabled() {
        if isEnabled { install() }
    }

    private func install() {
        guard statusItem == nil else { return }
        let item = NSStatusBar.system.statusItem(withLength: 0)
        item.autosaveName = Self.autosaveName
        if let button = item.button {
            button.title = ""
            button.image = nil
            button.isBordered = false
            button.setAccessibilityLabel("五子棋监控灵动岛")
            button.toolTip = nil
            button.target = self
            button.action = #selector(handleClick)
        }
        statusItem = item
        appliedLength = 0
    }

    private func uninstall() {
        debounceTimer?.invalidate()
        debounceTimer = nil
        pendingLength = nil
        if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
        }
        statusItem = nil
        appliedLength = 0
    }

    @objc private func handleClick() {
        onClick?()
    }

    func requestLength(_ length: CGFloat) {
        guard let item = statusItem else { return }
        let target = max(0, length.rounded(.up))
        let isZeroing = (target == 0 && appliedLength != 0)
        if abs(target - appliedLength) < 4 && !isZeroing {
            debounceTimer?.invalidate()
            debounceTimer = nil
            pendingLength = nil
            return
        }
        if target > appliedLength {
            debounceTimer?.invalidate()
            debounceTimer = nil
            pendingLength = nil
            apply(target, to: item)
            return
        }
        guard pendingLength != target else { return }
        pendingLength = target
        debounceTimer?.invalidate()
        let timer = Timer(timeInterval: 0.5, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, let item = self.statusItem, let pending = self.pendingLength else { return }
                self.debounceTimer = nil
                self.pendingLength = nil
                self.apply(pending, to: item)
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        debounceTimer = timer
    }

    private func apply(_ length: CGFloat, to item: NSStatusItem) {
        item.length = length
        appliedLength = length
        onLengthApplied?()
    }

    var windowFrame: CGRect? {
        guard let window = statusItem?.button?.window, window.isVisible else { return nil }
        return window.frame
    }

    func ensureLeftmost(foreignLeftmostMinX: CGFloat?, geometry: ScreenNotchGeometry) {
        guard isEnabled, appliedLength > 0,
              let foreignX = foreignLeftmostMinX,
              let frame = windowFrame,
              foreignX >= geometry.notchRight - 2,
              foreignX < frame.minX - 1,
              Date().timeIntervalSince(lastRepositionAt) > 30 else { return }
        lastRepositionAt = Date()
        let length = appliedLength
        uninstall()
        let position = (geometry.screenFrame.maxX - foreignX) + length + 20
        UserDefaults.standard.set(Double(position), forKey: Self.preferredPositionKey)
        install()
        if let item = statusItem {
            apply(length, to: item)
        }
    }
}
