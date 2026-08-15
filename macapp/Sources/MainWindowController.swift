// MainWindowController.swift
//
// The app window: the project's own web UI in a WKWebView, wired into macOS
// conventions — real Open panels, real downloads, external links in the
// browser, native zoom.

import AppKit
import UniformTypeIdentifiers
import WebKit

final class MainWindowController: NSObject, NSWindowDelegate, WKNavigationDelegate,
                                  WKUIDelegate, WKDownloadDelegate {
    let window: NSWindow
    private(set) var webView: WKWebView!
    private var downloadDestinations: [WKDownload: URL] = [:]
    private var baseURL: URL?

    /// Called when the web view's content process dies and we need the app to
    /// re-check that the server is still alive.
    var onContentProcessDied: (() -> Void)?

    override init() {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 840),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false)
        window.title = "Video Understanding"
        window.minSize = NSSize(width: 960, height: 640)
        window.isReleasedWhenClosed = false
        window.setFrameAutosaveName("VideoUnderstandingMainWindow")
        super.init()

        let configuration = WKWebViewConfiguration()
        // The UI starts playback from script when you click a scene or a
        // highlight; without this every seek would need a second click.
        configuration.mediaTypesRequiringUserActionForPlayback = []
        configuration.suppressesIncrementalRendering = false
        if #available(macOS 12.3, *) {
            configuration.preferences.isElementFullscreenEnabled = true
        }

        let webView = WKWebView(frame: window.contentLayoutRect, configuration: configuration)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = false
        self.webView = webView

        window.contentView = webView
        window.delegate = self
        window.center()
    }

    // MARK: Loading

    func load(_ url: URL) {
        baseURL = url
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    func show() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func reload() {
        if let baseURL = baseURL, webView.url == nil {
            load(baseURL)
        } else {
            webView.reloadFromOrigin()
        }
    }

    /// Opens a video in the UI after a native upload, using the page's own
    /// routing so the library and the detail view stay in step.
    func revealVideo(id: String) {
        let safe = id.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
        let script = """
        (function () {
          try { if (typeof loadLibrary === 'function') { loadLibrary(); } } catch (e) {}
          if (typeof openVideo === 'function') { openVideo('\(safe)'); }
          else { location.hash = '#/v/\(safe)'; }
        })();
        """
        webView.evaluateJavaScript(script, completionHandler: nil)
    }

    // MARK: Zoom

    func zoomIn() { webView.pageZoom = min(webView.pageZoom + 0.1, 2.5) }
    func zoomOut() { webView.pageZoom = max(webView.pageZoom - 0.1, 0.6) }
    func zoomReset() { webView.pageZoom = 1.0 }

    // MARK: WKNavigationDelegate

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.allow)
            return
        }
        // Anything that is not our own loopback server belongs in the browser.
        if isLocal(url) {
            decisionHandler(.allow)
        } else if url.scheme == "http" || url.scheme == "https" || url.scheme == "mailto" {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
        } else {
            decisionHandler(.cancel)
        }
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        let http = navigationResponse.response as? HTTPURLResponse
        let disposition = http?.value(forHTTPHeaderField: "Content-Disposition")?.lowercased() ?? ""
        if disposition.contains("attachment") || !navigationResponse.canShowMIMEType {
            decisionHandler(.download)
        } else {
            decisionHandler(.allow)
        }
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        LogStore.server.append("web view navigation failed: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        LogStore.server.append("web view could not load: \(error.localizedDescription)")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        LogStore.server.append("web content process terminated — reloading")
        onContentProcessDied?()
        reload()
    }

    private func isLocal(_ url: URL) -> Bool {
        guard let host = url.host else { return url.isFileURL == false }
        return host == "127.0.0.1" || host == "localhost"
    }

    // MARK: WKUIDelegate

    /// `target="_blank"` links (the API docs link in the header) open in the
    /// user's browser rather than a chromeless second window.
    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        if let url = navigationAction.request.url {
            NSWorkspace.shared.open(url)
        }
        return nil
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.allowedContentTypes = MainWindowController.videoContentTypes
        panel.beginSheetModal(for: window) { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping () -> Void
    ) {
        let alert = UI.alert("Video Understanding", informative: message)
        alert.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = UI.alert("Video Understanding", informative: message,
                             buttons: ["OK", "Cancel"])
        alert.beginSheetModal(for: window) { response in
            completionHandler(response == .alertFirstButtonReturn)
        }
    }

    // MARK: WKDownloadDelegate

    func download(
        _ download: WKDownload,
        decideDestinationUsing response: URLResponse,
        suggestedFilename: String,
        completionHandler: @escaping (URL?) -> Void
    ) {
        let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Downloads")
        let destination = uniqueURL(in: downloads, filename: suggestedFilename)
        downloadDestinations[download] = destination
        completionHandler(destination)
    }

    func downloadDidFinish(_ download: WKDownload) {
        guard let destination = downloadDestinations.removeValue(forKey: download) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([destination])
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        downloadDestinations.removeValue(forKey: download)
        let alert = UI.alert("The download failed",
                             informative: error.localizedDescription,
                             style: .warning)
        alert.beginSheetModal(for: window, completionHandler: nil)
    }

    private func uniqueURL(in directory: URL, filename: String) -> URL {
        let name = filename.isEmpty ? "download" : filename
        var candidate = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: candidate.path) else { return candidate }

        let base = candidate.deletingPathExtension().lastPathComponent
        let ext = candidate.pathExtension
        var index = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            let next = ext.isEmpty ? "\(base) \(index)" : "\(base) \(index).\(ext)"
            candidate = directory.appendingPathComponent(next)
            index += 1
        }
        return candidate
    }

    /// Matches `VIDEO_ALLOWED_EXTENSIONS` in the Python settings.
    static let videoExtensions = ["mp4", "mov", "m4v", "avi", "mkv", "webm"]

    static var videoContentTypes: [UTType] {
        var types: [UTType] = [.movie, .video]
        for ext in videoExtensions {
            if let type = UTType(filenameExtension: ext) { types.append(type) }
        }
        return types
    }
}
