// AppDelegate.swift
//
// Orchestration: set up the runtime, start the server, show the UI, and keep
// the two in step for the rest of the session.

import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate, SetupWindowDelegate {
    private let server = ServerController()
    private let setupWindow = SetupWindowController()
    private let mainWindow = MainWindowController()
    private lazy var modelsWindow: ModelsWindowController = {
        let controller = ModelsWindowController()
        controller.onFinished = { [weak self] in self?.restartServer(nil) }
        return controller
    }()
    private lazy var diagnosticsWindow: DiagnosticsWindowController = {
        let controller = DiagnosticsWindowController()
        controller.healthProvider = { [weak self] in self?.server.health() ?? nil }
        controller.portProvider = { [weak self] in self?.server.port ?? 0 }
        return controller
    }()
    private lazy var serverLogWindow = LogWindowController(
        title: "Server Log", store: LogStore.server, fileURL: Paths.serverLog)
    private lazy var setupLogWindow = LogWindowController(
        title: "Setup Log", store: LogStore.setup, fileURL: Paths.setupLog)

    /// Videos dropped on the dock icon before the server was ready.
    private var pendingFiles: [URL] = []
    private var isReady = false
    private var isWorking = false

    private let suppressModelPromptKey = "SuppressModelPrompt"

    // MARK: Lifecycle

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = MainMenu.build(target: self)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Paths.ensureDirectories()
        setupWindow.delegate = self
        mainWindow.onContentProcessDied = { [weak self] in self?.verifyServerStillUp() }
        server.onUnexpectedExit = { [weak self] status in self?.serverDied(status: status) }
        startBootstrap()
    }

    func applicationWillTerminate(_ notification: Notification) {
        server.stop()
        OllamaManager.shared.stopManagedServer()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if isReady { mainWindow.show() } else if !isWorking { setupWindow.show() }
        return true
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        true
    }

    /// Videos dropped on the dock icon, or opened with "Open With".
    func application(_ application: NSApplication, open urls: [URL]) {
        let videos = urls.filter { $0.isFileURL }
        guard !videos.isEmpty else { return }
        if isReady {
            analyze(videos)
        } else {
            pendingFiles.append(contentsOf: videos)
        }
    }

    // MARK: Setup

    private func startBootstrap() {
        guard !isWorking else { return }
        isWorking = true
        isReady = false
        LogStore.setup.rotate(header: "— setup started —")
        setupWindow.beginWork()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let bootstrapper = Bootstrapper(
                log: { LogStore.setup.append($0) },
                progress: { progress in
                    DispatchQueue.main.async { self.setupWindow.update(progress) }
                })

            switch bootstrapper.run() {
            case .failure(let failure):
                DispatchQueue.main.async {
                    self.isWorking = false
                    self.setupWindow.showFailure(title: failure.title,
                                                 message: failure.message,
                                                 remedy: failure.remedy)
                }
            case .success(let python):
                switch self.server.start(python: python) {
                case .failure(let error):
                    DispatchQueue.main.async {
                        self.isWorking = false
                        self.setupWindow.showFailure(title: error.title,
                                                     message: error.message,
                                                     remedy: .retry)
                    }
                case .success(let url):
                    DispatchQueue.main.async { self.serverBecameReady(at: url) }
                }
            }
        }
    }

    private func serverBecameReady(at url: URL) {
        isWorking = false
        isReady = true
        LogStore.setup.append("server ready at \(url.absoluteString)")

        mainWindow.load(url)
        mainWindow.show()
        setupWindow.close()

        let queued = pendingFiles
        pendingFiles.removeAll()
        if !queued.isEmpty { analyze(queued) }

        promptForModelsIfNeeded()
    }

    /// A one-time nudge: the app works without local models, but not well.
    private func promptForModelsIfNeeded() {
        guard !UserDefaults.standard.bool(forKey: suppressModelPromptKey) else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard ModelsWindowController.modelsAreMissing() else { return }
            DispatchQueue.main.async {
                guard let self = self else { return }
                let alert = UI.alert(
                    "Set up the local models?",
                    informative: "Without them the app still analyses video — scenes, frames, "
                        + "timings — but descriptions and highlights fall back to heuristics. "
                        + "The download is a few gigabytes and runs in the background.",
                    buttons: ["Set Up…", "Later", "Don't Ask Again"])
                alert.beginSheetModal(for: self.mainWindow.window) { response in
                    switch response {
                    case .alertFirstButtonReturn:
                        self.modelsWindow.show()
                    case .alertThirdButtonReturn:
                        UserDefaults.standard.set(true, forKey: self.suppressModelPromptKey)
                    default:
                        break
                    }
                }
            }
        }
    }

    // MARK: SetupWindowDelegate

    func setupWindowDidRequestRetry() {
        startBootstrap()
    }

    func setupWindowDidRequestRemedy(_ remedy: Remedy) {
        switch remedy {
        case .installFFmpegWithHomebrew:
            runHomebrewInstall(formula: "ffmpeg", headline: "Installing FFmpeg")
        case .installPythonWithHomebrew:
            runHomebrewInstall(formula: "python@3.12", headline: "Installing Python 3.12")
        case .openPythonDownloadPage:
            open("https://www.python.org/downloads/macos/")
            setupWindow.showFailure(
                title: "Waiting for Python",
                message: "Install Python 3.12 or newer, then press Try Again.",
                remedy: .retry)
        case .openHomebrewPage:
            open("https://brew.sh")
            setupWindow.showFailure(
                title: "Waiting for Homebrew",
                message: "Install Homebrew and then `brew install ffmpeg`, or install FFmpeg "
                    + "however you prefer, then press Try Again.",
                remedy: .retry)
        case .retry, .none:
            startBootstrap()
        }
    }

    private func runHomebrewInstall(formula: String, headline: String) {
        isWorking = true
        setupWindow.update(SetupProgress(headline: headline,
                                         detail: "Homebrew is doing the work; this can take a few minutes.",
                                         fraction: nil))
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let ok = Bootstrapper.homebrewInstall(formula, log: { LogStore.setup.append($0) })
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.isWorking = false
                if ok {
                    self.startBootstrap()
                } else {
                    self.setupWindow.showFailure(
                        title: "Homebrew could not install \(formula)",
                        message: "See the log below. You can install it yourself with "
                            + "`brew install \(formula)` and then press Try Again.",
                        remedy: .retry)
                }
            }
        }
    }

    // MARK: Server health

    private func serverDied(status: Int32) {
        guard isReady else { return }
        isReady = false
        let alert = UI.alert(
            "The analysis server stopped",
            informative: "It exited with status \(status). Restarting usually fixes it; the log "
                + "says why it stopped.",
            style: .warning,
            buttons: ["Restart", "Show Log", "Quit"])
        let handle: (NSApplication.ModalResponse) -> Void = { [weak self] response in
            guard let self = self else { return }
            switch response {
            case .alertFirstButtonReturn: self.restartServer(nil)
            case .alertSecondButtonReturn: self.serverLogWindow.show()
            default: NSApp.terminate(nil)
            }
        }
        // The window may be hidden — in that case the sheet would never be seen.
        if mainWindow.window.isVisible {
            alert.beginSheetModal(for: mainWindow.window, completionHandler: handle)
        } else {
            handle(alert.runModal())
        }
    }

    /// Cheap liveness probe used when the web view misbehaves.
    private func verifyServerStillUp() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self, self.server.health() == nil else { return }
            DispatchQueue.main.async { self.restartServer(nil) }
        }
    }

    // MARK: Menu actions

    @objc func openVideo(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = MainWindowController.videoContentTypes
        panel.message = "Choose one or more videos to analyse"
        panel.prompt = "Analyse"

        let complete: (NSApplication.ModalResponse) -> Void = { [weak self] response in
            guard response == .OK else { return }
            self?.analyze(panel.urls)
        }
        if isReady {
            panel.beginSheetModal(for: mainWindow.window, completionHandler: complete)
        } else {
            complete(panel.runModal())
        }
    }

    /// Uploads through the same endpoint the web UI uses, then asks the page to
    /// open the first one.
    private func analyze(_ urls: [URL]) {
        guard let baseURL = server.baseURL, isReady else {
            pendingFiles.append(contentsOf: urls)
            return
        }
        mainWindow.show()

        var first: String?
        let group = DispatchGroup()
        var failures: [String] = []

        for url in urls {
            group.enter()
            Uploader.upload(url, to: baseURL) { result in
                DispatchQueue.main.async {
                    switch result {
                    case .success(let videoID):
                        if first == nil { first = videoID }
                    case .failure(let error):
                        failures.append("\(url.lastPathComponent): \(error.localizedDescription)")
                    }
                    group.leave()
                }
            }
        }

        group.notify(queue: .main) { [weak self] in
            guard let self = self else { return }
            if let videoID = first { self.mainWindow.revealVideo(id: videoID) }
            guard !failures.isEmpty else { return }
            let alert = UI.alert(
                failures.count == 1 ? "That video could not be analysed" : "Some videos could not be analysed",
                informative: failures.joined(separator: "\n"),
                style: .warning)
            alert.beginSheetModal(for: self.mainWindow.window, completionHandler: nil)
        }
    }

    @objc func reloadUI(_ sender: Any?) { mainWindow.reload() }
    @objc func zoomIn(_ sender: Any?) { mainWindow.zoomIn() }
    @objc func zoomOut(_ sender: Any?) { mainWindow.zoomOut() }
    @objc func zoomReset(_ sender: Any?) { mainWindow.zoomReset() }

    @objc func showModels(_ sender: Any?) { modelsWindow.show() }
    @objc func showDiagnostics(_ sender: Any?) { diagnosticsWindow.show() }
    @objc func showServerLog(_ sender: Any?) { serverLogWindow.show() }
    @objc func showSetupLog(_ sender: Any?) { setupLogWindow.show() }

    @objc func openLibraryFolder(_ sender: Any?) {
        NSWorkspace.shared.open(Paths.dataDir)
    }

    @objc func openExportsFolder(_ sender: Any?) {
        NSWorkspace.shared.open(Paths.outputDir)
    }

    @objc func editConfiguration(_ sender: Any?) {
        if !FileManager.default.fileExists(atPath: Paths.configFile.path) {
            try? "".write(to: Paths.configFile, atomically: true, encoding: .utf8)
        }
        NSWorkspace.shared.open(Paths.configFile)
        let alert = UI.alert(
            "Configuration opened",
            informative: "Save your changes, then choose Server ▸ Restart Server for them to "
                + "take effect.")
        alert.runModal()
    }

    @objc func restartServer(_ sender: Any?) {
        guard !isWorking else { return }
        isWorking = true
        isReady = false
        setupWindow.beginWork()
        setupWindow.update(SetupProgress(headline: "Restarting the server",
                                         detail: "Reloading configuration and models.",
                                         fraction: nil))
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            self.server.stop()
            switch self.server.start(python: Paths.venvPython) {
            case .success(let url):
                DispatchQueue.main.async { self.serverBecameReady(at: url) }
            case .failure(let error):
                DispatchQueue.main.async {
                    self.isWorking = false
                    self.setupWindow.showFailure(title: error.title,
                                                 message: error.message,
                                                 remedy: .retry)
                }
            }
        }
    }

    @objc func reinstallRuntime(_ sender: Any?) {
        let alert = UI.alert(
            "Reinstall the runtime?",
            informative: "The private Python environment is deleted and rebuilt. Your library, "
                + "results and configuration are untouched. This needs an internet connection.",
            style: .warning,
            buttons: ["Reinstall", "Cancel"])
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        server.stop()
        try? FileManager.default.removeItem(at: Paths.runtime)
        startBootstrap()
    }

    @objc func resetLibrary(_ sender: Any?) {
        let alert = UI.alert(
            "Delete every video and result?",
            informative: "This permanently removes the uploads, thumbnails, renders and the "
                + "database at \(Paths.dataDir.path). Exports in \(Paths.outputDir.lastPathComponent) "
                + "are kept. This cannot be undone.",
            style: .critical,
            buttons: ["Delete Everything", "Cancel"])
        guard alert.runModal() == .alertFirstButtonReturn else { return }

        server.stop()
        try? FileManager.default.removeItem(at: Paths.dataDir)
        Paths.ensureDirectories()
        restartServer(nil)
    }

    @objc func openAPIDocs(_ sender: Any?) {
        guard let baseURL = server.baseURL else { return }
        NSWorkspace.shared.open(baseURL.appendingPathComponent("docs"))
    }

    @objc func openReadme(_ sender: Any?) {
        open("https://github.com/vikramchand/video-editing-tool#readme")
    }

    private func open(_ string: String) {
        guard let url = URL(string: string) else { return }
        NSWorkspace.shared.open(url)
    }
}
