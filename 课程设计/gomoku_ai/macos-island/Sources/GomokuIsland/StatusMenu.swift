import AppKit

@MainActor
final class StatusMenu: NSObject {
    private let statusItem: NSStatusItem
    private let toggleMenuItem = NSMenuItem()
    private let hoverExpandItem = NSMenuItem()
    private let hideIdleItem = NSMenuItem()
    private let reserveSpaceItem = NSMenuItem()

    private let islandController: IslandController
    private let poller: MonitorPoller

    init(islandController: IslandController, poller: MonitorPoller) {
        self.islandController = islandController
        self.poller = poller
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        super.init()
        configureStatusItem()
        configureMenu()
    }

    func refreshState(isExpanded: Bool) {
        toggleMenuItem.title = isExpanded ? "收起监控岛 (⌃⌥M)" : "展开监控岛 (⌃⌥M)"
    }

    private func configureStatusItem() {
        guard let button = statusItem.button else { return }
        let image = NSImage(
            systemSymbolName: "waveform.path.ecg",
            accessibilityDescription: "五子棋训练监控灵动岛"
        )
        image?.isTemplate = true
        button.image = image
        button.toolTip = "五子棋 AI 训练监控灵动岛"
    }

    private func configureMenu() {
        let menu = NSMenu()

        toggleMenuItem.title = "展开监控岛 (⌃⌥M)"
        toggleMenuItem.target = self
        toggleMenuItem.action = #selector(handleToggle)
        menu.addItem(toggleMenuItem)

        let fullItem = NSMenuItem(
            title: "打开完整监控台（灵动岛大窗）",
            action: #selector(handleOpenFull),
            keyEquivalent: ""
        )
        fullItem.target = self
        menu.addItem(fullItem)

        let webGuiItem = NSMenuItem(
            title: "打开五子棋对弈棋盘 (⌃⌥G)",
            action: #selector(handleOpenWebGui),
            keyEquivalent: ""
        )
        webGuiItem.target = self
        menu.addItem(webGuiItem)

        let browserItem = NSMenuItem(
            title: "在浏览器打开监控台 (8766)",
            action: #selector(handleOpenMonitorBrowser),
            keyEquivalent: ""
        )
        browserItem.target = self
        menu.addItem(browserItem)

        menu.addItem(.separator())

        let refreshItem = NSMenuItem(
            title: "立即刷新数据",
            action: #selector(handleRefresh),
            keyEquivalent: "r"
        )
        refreshItem.target = self
        menu.addItem(refreshItem)

        hoverExpandItem.title = "悬停刘海自动展开"
        hoverExpandItem.target = self
        hoverExpandItem.action = #selector(handleToggleHoverExpand)
        hoverExpandItem.state = islandController.hoverExpandEnabled ? .on : .off
        menu.addItem(hoverExpandItem)

        hideIdleItem.title = "仅训练中显示紧凑胶囊"
        hideIdleItem.target = self
        hideIdleItem.action = #selector(handleToggleHideWhenIdle)
        hideIdleItem.state = poller.hideWhenIdle ? .on : .off
        menu.addItem(hideIdleItem)

        reserveSpaceItem.title = "为灵动岛预留状态栏空间"
        reserveSpaceItem.target = self
        reserveSpaceItem.action = #selector(handleToggleReserveSpace)
        reserveSpaceItem.state = islandController.notchSpacer.isEnabled ? .on : .off
        menu.addItem(reserveSpaceItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(
            title: "退出五子棋监控岛",
            action: #selector(handleQuit),
            keyEquivalent: "q"
        )
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    @objc private func handleToggle() {
        islandController.toggleFromHotkeyOrMenu()
    }

    @objc private func handleOpenFull() {
        islandController.openFullMonitorInIsland()
    }

    @objc private func handleOpenWebGui() {
        if let url = URL(string: "http://127.0.0.1:8768") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func handleOpenMonitorBrowser() {
        if let url = URL(string: poller.monitorBaseURL) {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func handleRefresh() {
        poller.pollNow()
    }

    @objc private func handleToggleHoverExpand() {
        islandController.hoverExpandEnabled.toggle()
        hoverExpandItem.state = islandController.hoverExpandEnabled ? .on : .off
    }

    @objc private func handleToggleHideWhenIdle() {
        poller.hideWhenIdle.toggle()
        hideIdleItem.state = poller.hideWhenIdle ? .on : .off
        islandController.refreshCompactOrHidden()
        islandController.refreshNotchSpacer()
    }

    @objc private func handleToggleReserveSpace() {
        let spacer = islandController.notchSpacer
        spacer.isEnabled.toggle()
        reserveSpaceItem.state = spacer.isEnabled ? .on : .off
        islandController.refreshNotchSpacer()
    }

    @objc private func handleQuit() {
        NSApp.terminate(nil)
    }
}
