import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let poller = MonitorPoller()
    private var webBridge: WebBridge!
    private var islandController: IslandController!
    private let hotkeys = Hotkeys()
    private var statusMenu: StatusMenu!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        webBridge = WebBridge(poller: poller)
        islandController = IslandController(poller: poller, webBridge: webBridge)
        webBridge.delegate = islandController

        poller.onCompactStateChanged = { [weak self] _ in
            self?.islandController.refreshCompactOrHidden()
        }
        poller.onSnapshotUpdated = { [weak self] rawJson in
            self?.webBridge.pushSnapshot(rawJson)
        }
        poller.onPromptRequested = { [weak self] text, targetSurface in
            self?.islandController.showPrompt(text: text, targetSurface: targetSurface)
        }

        hotkeys.register(
            onToggleIsland: { [weak self] in
                self?.islandController.toggleFromHotkeyOrMenu()
            },
            onOpenWebGui: {
                if let url = URL(string: "http://127.0.0.1:8768") {
                    NSWorkspace.shared.open(url)
                }
            }
        )

        statusMenu = StatusMenu(islandController: islandController, poller: poller)
        islandController.onStateChanged = { [weak self] in
            guard let self else { return }
            self.statusMenu.refreshState(isExpanded: self.islandController.mode == .expanded)
        }

        webBridge.loadUI()
        poller.start()
        islandController.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        poller.stop()
    }
}
