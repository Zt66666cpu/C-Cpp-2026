import AppKit
import SwiftUI
import WebKit

enum WebViewKind {
    case summary
    case full
}

@MainActor
protocol WebBridgeDelegate: AnyObject {
    var currentSurfaceRawValue: String { get }
    var currentSurfaceStyleRawValue: String { get }
    func webBridgeDidRequestSurface(_ surface: String)
    func webBridgeDidRequestCollapse()
    func webBridgeDidUpdateInputFocus(_ focused: Bool)
}

@MainActor
final class WebBridge: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    let summaryWebView: WKWebView
    let fullWebView: WKWebView
    let poller: MonitorPoller
    weak var delegate: WebBridgeDelegate?

    private let keepAliveWindow: NSWindow
    private var summaryReady = false
    private var fullLoadedOnce = false
    private var lastSurfaceStyle = "notch"
    private var lastSnapshotJSON: String?

    init(poller: MonitorPoller) {
        self.poller = poller

        let summaryConfig = WKWebViewConfiguration()
        summaryConfig.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        let cc = WKUserContentController()
        summaryConfig.userContentController = cc
        let summaryWV = WKWebView(frame: .zero, configuration: summaryConfig)
        summaryWV.setValue(false, forKey: "drawsBackground")
        summaryWV.autoresizingMask = [.width, .height]
        self.summaryWebView = summaryWV

        let fullConfig = WKWebViewConfiguration()
        let fullWV = WKWebView(frame: .zero, configuration: fullConfig)
        fullWV.setValue(false, forKey: "drawsBackground")
        fullWV.autoresizingMask = [.width, .height]
        self.fullWebView = fullWV

        let keepAliveWindow = NSWindow(
            contentRect: NSRect(x: -10000, y: -10000, width: 480, height: 540),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        keepAliveWindow.isReleasedWhenClosed = false
        keepAliveWindow.alphaValue = 0
        keepAliveWindow.ignoresMouseEvents = true
        keepAliveWindow.isOpaque = false
        keepAliveWindow.backgroundColor = .clear
        keepAliveWindow.hasShadow = false
        keepAliveWindow.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        keepAliveWindow.contentView = NSView(frame: NSRect(x: 0, y: 0, width: 480, height: 540))
        self.keepAliveWindow = keepAliveWindow

        super.init()

        summaryWV.navigationDelegate = self
        cc.add(self, name: "native")
        attachToKeepAliveWindow()
    }

    func webView(for kind: WebViewKind) -> WKWebView {
        switch kind {
        case .summary: return summaryWebView
        case .full: return fullWebView
        }
    }

    func loadUI() {
        if let bundleURL = Bundle.main.resourceURL?.appendingPathComponent("ui/island.html"),
           FileManager.default.fileExists(atPath: bundleURL.path) {
            summaryWebView.loadFileURL(bundleURL, allowingReadAccessTo: bundleURL.deletingLastPathComponent())
            return
        }
        let workingDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("ui/island.html")
        summaryWebView.loadFileURL(
            workingDirectoryURL,
            allowingReadAccessTo: workingDirectoryURL.deletingLastPathComponent()
        )
    }

    func ensureFullMonitorLoaded(forceReload: Bool = false) {
        if fullLoadedOnce && !forceReload {
            return
        }
        var comps = URLComponents(string: poller.monitorBaseURL)
        if let key = poller.selectedRunKey, !key.isEmpty {
            comps?.queryItems = [URLQueryItem(name: "run", value: key)]
        }
        guard let url = comps?.url else { return }
        fullLoadedOnce = true
        fullWebView.load(URLRequest(url: url))
    }

    func reloadFullMonitor() {
        if !fullLoadedOnce {
            ensureFullMonitorLoaded(forceReload: true)
        } else {
            fullWebView.reload()
        }
    }

    func attachToKeepAliveWindow() {
        guard let contentView = keepAliveWindow.contentView else { return }
        for wv in [summaryWebView, fullWebView] {
            if wv.superview !== contentView {
                wv.removeFromSuperview()
                contentView.addSubview(wv)
            }
            wv.frame = contentView.bounds
        }
        keepAliveWindow.orderBack(nil)
    }

    func attach(_ kind: WebViewKind, to hostView: NSView) {
        let wv = webView(for: kind)
        if wv.superview !== hostView {
            wv.removeFromSuperview()
            hostView.addSubview(wv)
        }
        wv.frame = hostView.bounds
        wv.autoresizingMask = [.width, .height]
    }

    func fadeWebViews(to alpha: CGFloat, duration: TimeInterval) {
        if duration <= 0 {
            summaryWebView.alphaValue = alpha
            fullWebView.alphaValue = alpha
            return
        }
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = duration
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            summaryWebView.animator().alphaValue = alpha
            fullWebView.animator().alphaValue = alpha
        }
    }

    func setSurfaceStyle(_ style: String) {
        lastSurfaceStyle = style
        summaryWebView.evaluateJavaScript(
            "window.__setSurfaceStyle && window.__setSurfaceStyle(\(jsString(style)));",
            completionHandler: nil
        )
    }

    func pushSnapshot(_ rawJSON: String) {
        lastSnapshotJSON = rawJSON
        guard summaryReady else { return }
        let meta: [String: Any] = [
            "surfaceStyle": lastSurfaceStyle,
            "selectedRunKey": poller.selectedRunKey ?? "",
            "monitorUrl": poller.monitorBaseURL,
            "webGuiUrl": "http://127.0.0.1:8768"
        ]
        let metaStr = jsonLiteral(meta)
        summaryWebView.evaluateJavaScript(
            "window.__updateMonitorData && window.__updateMonitorData(\(jsString(rawJSON)), \(metaStr));",
            completionHandler: nil
        )
    }

    // MARK: - WKNavigationDelegate

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard webView === summaryWebView else { return }
        summaryReady = true
        setSurfaceStyle(delegate?.currentSurfaceStyleRawValue ?? lastSurfaceStyle)
        if let cached = lastSnapshotJSON ?? poller.latestSnapshotJSON {
            pushSnapshot(cached)
        }
    }

    // MARK: - WKScriptMessageHandler

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard let body = message.body as? [String: Any],
              let action = body["action"] as? String else { return }

        switch action {
        case "setSurface":
            if let surface = body["surface"] as? String {
                delegate?.webBridgeDidRequestSurface(surface)
            }
        case "selectRun":
            let key = body["run"] as? String
            poller.selectRun(key)
        case "refresh":
            poller.pollNow()
        case "openExternal":
            if let raw = body["url"] as? String, let url = URL(string: raw) {
                NSWorkspace.shared.open(url)
            }
        case "collapse":
            delegate?.webBridgeDidRequestCollapse()
        case "inputFocusChanged":
            if let focused = body["focused"] as? Bool {
                delegate?.webBridgeDidUpdateInputFocus(focused)
            }
        default:
            break
        }
    }

    private func jsString(_ value: String) -> String {
        guard let data = try? JSONEncoder().encode(value),
              let encoded = String(data: data, encoding: .utf8) else {
            return "\"\""
        }
        return encoded
    }

    private func jsonLiteral(_ value: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }
}

final class WebViewHostView: NSView {
    weak var bridge: WebBridge?
    var kind: WebViewKind = .summary

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil {
            bridge?.attach(kind, to: self)
        }
    }

    override func layout() {
        super.layout()
        if window != nil {
            bridge?.attach(kind, to: self)
        }
    }
}

struct WebViewHostRepresentable: NSViewRepresentable {
    let bridge: WebBridge
    let kind: WebViewKind

    func makeNSView(context: Context) -> WebViewHostView {
        let view = WebViewHostView()
        view.kind = kind
        view.bridge = bridge
        return view
    }

    func updateNSView(_ nsView: WebViewHostView, context: Context) {
        nsView.kind = kind
        nsView.bridge = bridge
        if nsView.window != nil {
            bridge.attach(kind, to: nsView)
        }
    }
}
