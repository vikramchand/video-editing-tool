// InfoWindows.swift
//
// The three secondary windows: local model setup, diagnostics, and a live log
// tail. None of them are required to use the app; all of them exist so that
// "it isn't working" has an answer inside the app rather than in a terminal.

import AppKit

// MARK: - Models

/// Installs and downloads the local models.
///
/// Deliberately not part of first-run setup: the models are several gigabytes,
/// and the app is usable (with heuristic fallbacks, and a warning on every
/// result) before they arrive.
final class ModelsWindowController: NSObject {
    private let window: NSWindow
    private let headline = UI.label("Local models", size: 17, weight: .semibold)
    private let detail = UI.label("", size: 12, color: .secondaryLabelColor)
    private let statusLabel = UI.label("", size: 12)
    private let progressBar = NSProgressIndicator()
    private let logScroll: NSScrollView
    private let logText: NSTextView
    private let installButton: NSButton
    private let closeButton: NSButton

    private let plan = ModelPlan.recommended()
    private var working = false
    private var logToken: UUID?

    /// Called on the main queue after models are installed, so the server can
    /// be restarted and pick them up.
    var onFinished: (() -> Void)?

    override init() {
        window = UI.window(title: "Local Models", width: 620, height: 470)
        let views = UI.logView()
        logScroll = views.scroll
        logText = views.text
        installButton = NSButton()
        closeButton = NSButton()
        super.init()

        installButton.bezelStyle = .rounded
        installButton.translatesAutoresizingMaskIntoConstraints = false
        installButton.title = "Install Models"
        installButton.keyEquivalent = "\r"
        installButton.target = self
        installButton.action = #selector(installTapped)

        closeButton.bezelStyle = .rounded
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.title = "Not Now"
        closeButton.target = self
        closeButton.action = #selector(closeTapped)

        buildLayout()
    }

    private func buildLayout() {
        guard let content = window.contentView else { return }

        progressBar.translatesAutoresizingMaskIntoConstraints = false
        progressBar.style = .bar
        progressBar.isIndeterminate = true
        progressBar.isHidden = true

        detail.stringValue = """
        Vision and reasoning run through Ollama on this machine. \
        For \(Toolchain.memoryGB) GB of RAM the recommended set is \
        \(plan.vision) and \(plan.llm) — roughly 9 GB to download, once.
        """

        for view in [headline, detail, statusLabel, progressBar, logScroll, installButton, closeButton] {
            content.addSubview(view)
        }

        NSLayoutConstraint.activate([
            headline.topAnchor.constraint(equalTo: content.topAnchor, constant: 22),
            headline.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            headline.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            detail.topAnchor.constraint(equalTo: headline.bottomAnchor, constant: 8),
            detail.leadingAnchor.constraint(equalTo: headline.leadingAnchor),
            detail.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            statusLabel.topAnchor.constraint(equalTo: detail.bottomAnchor, constant: 12),
            statusLabel.leadingAnchor.constraint(equalTo: headline.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            progressBar.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 12),
            progressBar.leadingAnchor.constraint(equalTo: headline.leadingAnchor),
            progressBar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            logScroll.topAnchor.constraint(equalTo: progressBar.bottomAnchor, constant: 12),
            logScroll.leadingAnchor.constraint(equalTo: headline.leadingAnchor),
            logScroll.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            logScroll.bottomAnchor.constraint(equalTo: installButton.topAnchor, constant: -16),

            installButton.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            installButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -20),
            installButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 120),

            closeButton.trailingAnchor.constraint(equalTo: installButton.leadingAnchor, constant: -10),
            closeButton.bottomAnchor.constraint(equalTo: installButton.bottomAnchor),
        ])
    }

    func show() {
        if logToken == nil {
            logToken = LogStore.setup.addObserver { [weak self] line in
                guard let self = self else { return }
                UI.append(line, to: self.logText)
            }
        }
        refreshStatus()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// True when either recommended model is missing — the trigger for the
    /// gentle nudge on the main window.
    static func modelsAreMissing() -> Bool {
        let plan = ModelPlan.recommended()
        let ollama = OllamaManager.shared
        guard ollama.binary != nil, ollama.isRunning(timeout: 1.5) else { return true }
        return !ollama.isInstalled(plan.vision) || !ollama.isInstalled(plan.llm)
    }

    private func refreshStatus() {
        statusLabel.stringValue = "Checking…"
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let ollama = OllamaManager.shared
            let installed = ollama.binary != nil
            let running = installed && ollama.isRunning(timeout: 2)
            let models = running ? ollama.installedModels() : []
            let hasVision = models.contains(self.plan.vision)
            let hasLLM = models.contains(self.plan.llm)

            let lines = [
                Self.statusLine("Ollama", ok: running,
                                okText: "running", badText: installed ? "installed, not running" : "not installed"),
                Self.statusLine("Vision · \(self.plan.vision)", ok: hasVision,
                                okText: "ready", badText: "not downloaded"),
                Self.statusLine("Reasoning · \(self.plan.llm)", ok: hasLLM,
                                okText: "ready", badText: "not downloaded"),
                Self.statusLine("Transcription · whisper \(self.plan.whisper)", ok: true,
                                okText: "downloads itself on first use", badText: ""),
            ]

            DispatchQueue.main.async {
                self.statusLabel.stringValue = lines.joined(separator: "\n")
                let complete = running && hasVision && hasLLM
                self.installButton.isEnabled = !complete && !self.working
                self.installButton.title = complete ? "All Set" : "Install Models"
                self.closeButton.title = complete ? "Close" : "Not Now"
            }
        }
    }

    private static func statusLine(_ name: String, ok: Bool, okText: String, badText: String) -> String {
        "\(ok ? "✓" : "•")  \(name) — \(ok ? okText : badText)"
    }

    @objc private func installTapped() {
        guard !working else { return }
        working = true
        installButton.isEnabled = false
        progressBar.isHidden = false
        progressBar.startAnimation(nil)
        statusLabel.stringValue = "Working…"

        let plan = self.plan
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let log: (String) -> Void = { LogStore.setup.append($0) }
            let ollama = OllamaManager.shared

            if ollama.binary == nil {
                log("installing ollama with Homebrew…")
                if !Bootstrapper.homebrewInstall("ollama", log: log) {
                    log("could not install ollama automatically — see https://ollama.com/download")
                }
            }
            if ollama.binary != nil {
                ollama.startIfNeeded(log: log)
                if ollama.isRunning() {
                    if !ollama.isInstalled(plan.vision) { _ = ollama.pull(plan.vision, log: log) }
                    if !ollama.isInstalled(plan.llm) { _ = ollama.pull(plan.llm, log: log) }
                }
            }

            DispatchQueue.main.async {
                guard let self = self else { return }
                self.working = false
                self.progressBar.stopAnimation(nil)
                self.progressBar.isHidden = true
                self.refreshStatus()
                self.onFinished?()
            }
        }
    }

    @objc private func closeTapped() {
        window.orderOut(nil)
    }
}

// MARK: - Diagnostics

/// The in-app equivalent of `video-understand check`.
final class DiagnosticsWindowController: NSObject {
    private let window: NSWindow
    private let textView: NSTextView
    private let refreshButton: NSButton

    /// Supplies the live `/health` payload; set by the app delegate.
    var healthProvider: (() -> [String: Any]?)?
    var portProvider: (() -> Int)?

    override init() {
        window = UI.window(title: "Diagnostics", width: 660, height: 520, resizable: true)
        let views = UI.logView()
        textView = views.text
        refreshButton = NSButton()
        super.init()

        refreshButton.bezelStyle = .rounded
        refreshButton.translatesAutoresizingMaskIntoConstraints = false
        refreshButton.title = "Refresh"
        refreshButton.target = self
        refreshButton.action = #selector(refresh)

        let copyButton = UI.button("Copy", target: self, action: #selector(copyReport))

        guard let content = window.contentView else { return }
        content.addSubview(views.scroll)
        content.addSubview(refreshButton)
        content.addSubview(copyButton)

        NSLayoutConstraint.activate([
            views.scroll.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            views.scroll.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 16),
            views.scroll.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
            views.scroll.bottomAnchor.constraint(equalTo: refreshButton.topAnchor, constant: -12),

            refreshButton.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
            refreshButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -16),
            copyButton.trailingAnchor.constraint(equalTo: refreshButton.leadingAnchor, constant: -10),
            copyButton.bottomAnchor.constraint(equalTo: refreshButton.bottomAnchor),
        ])
    }

    func show() {
        refresh()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func refresh() {
        UI.setText("Collecting…\n", in: textView)
        let health = healthProvider
        let port = portProvider?() ?? 0
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let payload: [String: Any]? = health.flatMap { $0() }
            let report = DiagnosticsWindowController.buildReport(port: port, health: payload)
            DispatchQueue.main.async {
                guard let self = self else { return }
                UI.setText(report, in: self.textView)
            }
        }
    }

    @objc private func copyReport() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(textView.string, forType: .string)
    }

    private static func buildReport(port: Int, health: [String: Any]?) -> String {
        var lines: [String] = []
        lines.append("Video Understanding \(Paths.appVersion) (build \(Paths.buildID))")
        lines.append("Hardware        \(Toolchain.hardwareDescription)")
        lines.append("macOS           \(ProcessInfo.processInfo.operatingSystemVersionString)")
        lines.append("")

        if let python = Toolchain.findPython() {
            lines.append("Python          \(python.versionString) — \(python.url.path)")
        } else {
            lines.append("Python          not found (3.12+ required)")
        }
        lines.append("Runtime         \(Paths.runtime.path)")
        lines.append("FFmpeg          \(Toolchain.ffmpegVersion() ?? "not found on PATH")")

        let ollama = OllamaManager.shared
        if ollama.binary != nil {
            let running = ollama.isRunning(timeout: 2)
            lines.append("Ollama          \(running ? "running at \(ollama.host)" : "installed, not running")")
            if running {
                let models = ollama.installedModels()
                lines.append("Models          \(models.isEmpty ? "none installed" : models.joined(separator: ", "))")
            }
        } else {
            lines.append("Ollama          not installed")
        }

        lines.append("")
        let address = port > 0 ? "http://127.0.0.1:\(port)" : "not running"
        lines.append("Server          \(address)")
        lines.append("Library         \(Paths.dataDir.path)")
        lines.append("Exports         \(Paths.outputDir.path)")
        lines.append("Configuration   \(Paths.configFile.path)")
        lines.append("Logs            \(Paths.logsDir.path)")
        lines.append("")

        if let health = health {
            lines.append("— /health —")
            if let data = try? JSONSerialization.data(withJSONObject: health,
                                                      options: [.prettyPrinted, .sortedKeys]),
               let text = String(data: data, encoding: .utf8) {
                lines.append(text)
            }
        } else {
            lines.append("— /health — unreachable")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Logs

/// A live tail of one of the log stores.
final class LogWindowController: NSObject {
    private let window: NSWindow
    private let textView: NSTextView
    private let store: LogStore
    private let fileURL: URL
    private var token: UUID?

    init(title: String, store: LogStore, fileURL: URL) {
        window = UI.window(title: title, width: 760, height: 520, resizable: true)
        let views = UI.logView()
        textView = views.text
        self.store = store
        self.fileURL = fileURL
        super.init()

        let revealButton = UI.button("Reveal in Finder", target: self, action: #selector(reveal))
        let copyButton = UI.button("Copy", target: self, action: #selector(copyAll))

        guard let content = window.contentView else { return }
        content.addSubview(views.scroll)
        content.addSubview(revealButton)
        content.addSubview(copyButton)

        NSLayoutConstraint.activate([
            views.scroll.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            views.scroll.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 16),
            views.scroll.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
            views.scroll.bottomAnchor.constraint(equalTo: revealButton.topAnchor, constant: -12),

            revealButton.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
            revealButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -16),
            copyButton.trailingAnchor.constraint(equalTo: revealButton.leadingAnchor, constant: -10),
            copyButton.bottomAnchor.constraint(equalTo: revealButton.bottomAnchor),
        ])
    }

    func show() {
        UI.setText(store.text() + "\n", in: textView)
        if token == nil {
            token = store.addObserver { [weak self] line in
                guard let self = self else { return }
                UI.append(line, to: self.textView)
            }
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func reveal() {
        NSWorkspace.shared.activateFileViewerSelecting([fileURL])
    }

    @objc private func copyAll() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(textView.string, forType: .string)
    }
}
