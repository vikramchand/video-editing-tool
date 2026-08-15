// SetupWindowController.swift
//
// The window the user sees on first launch, and any time setup fails: a
// headline, a progress bar, a live log, and — when something is missing — a
// button that fixes it.

import AppKit

protocol SetupWindowDelegate: AnyObject {
    func setupWindowDidRequestRetry()
    func setupWindowDidRequestRemedy(_ remedy: Remedy)
}

final class SetupWindowController: NSObject, NSWindowDelegate {
    weak var delegate: SetupWindowDelegate?

    private let window: NSWindow
    private let headline = UI.label("Starting Video Understanding", size: 17, weight: .semibold)
    private let detail = UI.label("", size: 12, color: .secondaryLabelColor)
    private let progressBar = NSProgressIndicator()
    private let logScroll: NSScrollView
    private let logText: NSTextView
    private let primaryButton: NSButton
    private let secondaryButton: NSButton

    private var pendingRemedy: Remedy = .none

    override init() {
        window = UI.window(title: "Video Understanding", width: 620, height: 470)
        let views = UI.logView()
        logScroll = views.scroll
        logText = views.text
        primaryButton = NSButton()
        secondaryButton = NSButton()
        super.init()

        primaryButton.bezelStyle = .rounded
        primaryButton.translatesAutoresizingMaskIntoConstraints = false
        primaryButton.target = self
        primaryButton.action = #selector(primaryTapped)
        primaryButton.keyEquivalent = "\r"
        primaryButton.isHidden = true

        secondaryButton.bezelStyle = .rounded
        secondaryButton.translatesAutoresizingMaskIntoConstraints = false
        secondaryButton.target = self
        secondaryButton.action = #selector(secondaryTapped)
        secondaryButton.title = "Quit"

        buildLayout()
        window.delegate = self

        LogStore.setup.addObserver { [weak self] line in
            guard let self = self else { return }
            UI.append(line, to: self.logText)
        }
    }

    // MARK: Layout

    private func buildLayout() {
        guard let content = window.contentView else { return }

        let icon = NSImageView()
        icon.image = NSApp.applicationIconImage
        icon.imageScaling = .scaleProportionallyUpOrDown
        icon.translatesAutoresizingMaskIntoConstraints = false

        progressBar.translatesAutoresizingMaskIntoConstraints = false
        progressBar.style = .bar
        progressBar.isIndeterminate = true
        progressBar.minValue = 0
        progressBar.maxValue = 1
        progressBar.startAnimation(nil)

        for view in [icon, headline, detail, progressBar, logScroll, primaryButton, secondaryButton] {
            content.addSubview(view)
        }

        NSLayoutConstraint.activate([
            icon.topAnchor.constraint(equalTo: content.topAnchor, constant: 24),
            icon.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            icon.widthAnchor.constraint(equalToConstant: 56),
            icon.heightAnchor.constraint(equalToConstant: 56),

            headline.topAnchor.constraint(equalTo: content.topAnchor, constant: 26),
            headline.leadingAnchor.constraint(equalTo: icon.trailingAnchor, constant: 16),
            headline.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            detail.topAnchor.constraint(equalTo: headline.bottomAnchor, constant: 6),
            detail.leadingAnchor.constraint(equalTo: headline.leadingAnchor),
            detail.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            progressBar.topAnchor.constraint(equalTo: icon.bottomAnchor, constant: 20),
            progressBar.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            progressBar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),

            logScroll.topAnchor.constraint(equalTo: progressBar.bottomAnchor, constant: 16),
            logScroll.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            logScroll.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            logScroll.bottomAnchor.constraint(equalTo: primaryButton.topAnchor, constant: -16),

            primaryButton.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            primaryButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -20),
            primaryButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 110),

            secondaryButton.trailingAnchor.constraint(equalTo: primaryButton.leadingAnchor, constant: -10),
            secondaryButton.bottomAnchor.constraint(equalTo: primaryButton.bottomAnchor),
        ])
    }

    // MARK: Presentation

    func show() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func close() {
        window.orderOut(nil)
    }

    var isVisible: Bool { window.isVisible }

    /// Back to the "working on it" state.
    func beginWork() {
        pendingRemedy = .none
        primaryButton.isHidden = true
        secondaryButton.title = "Quit"
        UI.setText("", in: logText)
        progressBar.isIndeterminate = true
        progressBar.startAnimation(nil)
        headline.stringValue = "Setting up Video Understanding"
        detail.stringValue = "First run only — the app installs its own private runtime."
        show()
    }

    func update(_ progress: SetupProgress) {
        headline.stringValue = progress.headline
        detail.stringValue = progress.detail
        if let fraction = progress.fraction {
            progressBar.isIndeterminate = false
            progressBar.doubleValue = fraction
        } else {
            progressBar.isIndeterminate = true
            progressBar.startAnimation(nil)
        }
    }

    func showFailure(title: String, message: String, remedy: Remedy) {
        pendingRemedy = remedy
        progressBar.stopAnimation(nil)
        progressBar.isIndeterminate = false
        progressBar.doubleValue = 0
        headline.stringValue = title
        detail.stringValue = message
        secondaryButton.title = "Quit"

        switch remedy {
        case .installFFmpegWithHomebrew:
            primaryButton.title = "Install FFmpeg"
        case .installPythonWithHomebrew:
            primaryButton.title = "Install Python"
        case .openPythonDownloadPage:
            primaryButton.title = "Download Python"
        case .openHomebrewPage:
            primaryButton.title = "Get Homebrew"
        case .retry, .none:
            primaryButton.title = "Try Again"
        }
        primaryButton.isHidden = false
        show()
    }

    // MARK: Actions

    @objc private func primaryTapped() {
        let remedy = pendingRemedy
        pendingRemedy = .none
        primaryButton.isHidden = true
        if case .none = remedy {
            delegate?.setupWindowDidRequestRetry()
        } else if case .retry = remedy {
            delegate?.setupWindowDidRequestRetry()
        } else {
            delegate?.setupWindowDidRequestRemedy(remedy)
        }
    }

    @objc private func secondaryTapped() {
        NSApp.terminate(nil)
    }
}
