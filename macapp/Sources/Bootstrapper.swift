// Bootstrapper.swift
//
// First-run setup: find a Python, make sure FFmpeg exists, build a private
// virtualenv, install the bundled package into it, and write a config sized
// for this machine. Runs on a background queue; reports progress upward.

import Foundation

/// What the user can do about a failure, expressed as intent rather than UI.
enum Remedy {
    case installFFmpegWithHomebrew
    case installPythonWithHomebrew
    case openPythonDownloadPage
    case openHomebrewPage
    case retry
    case none
}

struct SetupFailure: Error {
    let title: String
    let message: String
    let remedy: Remedy
}

/// Progress reported to the setup window.
struct SetupProgress {
    let headline: String
    let detail: String
    /// nil means "indeterminate".
    let fraction: Double?
}

final class Bootstrapper {
    /// Optional dependency groups installed by default. Both are wanted for the
    /// full experience: local Whisper transcription and content-aware scenes.
    private let extras = ProcessInfo.processInfo.environment["VU_EXTRAS"] ?? "whisper,scenes"

    private let log: (String) -> Void
    private let progress: (SetupProgress) -> Void

    init(log: @escaping (String) -> Void, progress: @escaping (SetupProgress) -> Void) {
        self.log = log
        self.progress = progress
    }

    /// Runs every setup step. Returns the interpreter the server should use.
    func run() -> Result<URL, SetupFailure> {
        Paths.ensureDirectories()

        guard FileManager.default.fileExists(atPath: Paths.bundledSource.path) else {
            return .failure(SetupFailure(
                title: "The app bundle is incomplete",
                message: "No Python source was found inside the application. "
                    + "Rebuild the app with `make mac`.",
                remedy: .none))
        }

        if let failure = checkFFmpeg() { return .failure(failure) }

        let interpreter: PythonInterpreter
        switch findPython() {
        case .success(let found): interpreter = found
        case .failure(let failure): return .failure(failure)
        }

        if let failure = prepareRuntime(using: interpreter) { return .failure(failure) }

        writeConfigurationIfMissing()

        progress(SetupProgress(headline: "Ready", detail: "Starting the server…", fraction: 1.0))
        return .success(Paths.venvPython)
    }

    // MARK: Steps

    private func checkFFmpeg() -> SetupFailure? {
        progress(SetupProgress(headline: "Checking FFmpeg",
                               detail: "FFmpeg does the decoding, sampling and rendering.",
                               fraction: 0.05))
        if let version = Toolchain.ffmpegVersion(), Toolchain.ffprobe() != nil {
            log("found \(version)")
            return nil
        }
        log("ffmpeg was not found")
        if Toolchain.brew() != nil {
            return SetupFailure(
                title: "FFmpeg is required",
                message: "FFmpeg does all the decoding, frame sampling and rendering. "
                    + "Homebrew is installed, so the app can install it for you.",
                remedy: .installFFmpegWithHomebrew)
        }
        return SetupFailure(
            title: "FFmpeg is required",
            message: "FFmpeg does all the decoding, frame sampling and rendering. "
                + "Install Homebrew, then run: brew install ffmpeg",
            remedy: .openHomebrewPage)
    }

    private func findPython() -> Result<PythonInterpreter, SetupFailure> {
        progress(SetupProgress(headline: "Checking Python",
                               detail: "Looking for Python 3.12 or newer.",
                               fraction: 0.12))
        if let interpreter = Toolchain.findPython() {
            log("using Python \(interpreter.versionString) at \(interpreter.url.path)")
            return .success(interpreter)
        }

        let found = Toolchain.findAnyPython()
        let detail = found.map { "The newest one found is \($0.versionString) at \($0.url.path)." }
            ?? "No Python 3 installation was found."
        log("no suitable Python: \(detail)")

        if Toolchain.brew() != nil {
            return .failure(SetupFailure(
                title: "Python 3.12 or newer is required",
                message: "\(detail) Homebrew is installed, so the app can install Python for you.",
                remedy: .installPythonWithHomebrew))
        }
        return .failure(SetupFailure(
            title: "Python 3.12 or newer is required",
            message: "\(detail) Install Python from python.org, then try again.",
            remedy: .openPythonDownloadPage))
    }

    /// Creates the virtualenv and installs the bundled package into it.
    ///
    /// A stamp file records the app version, the source revision and the extras
    /// that were installed; anything different means the runtime is rebuilt.
    private func prepareRuntime(using interpreter: PythonInterpreter) -> SetupFailure? {
        let stamp = "\(Paths.appVersion)|\(Paths.buildID)|\(extras)|\(interpreter.versionString)"
        let existing = (try? String(contentsOf: Paths.stampFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if existing == stamp, FileManager.default.isExecutableFile(atPath: Paths.venvPython.path) {
            log("runtime is up to date (\(Paths.buildID))")
            progress(SetupProgress(headline: "Runtime ready",
                                   detail: "Everything is already installed.",
                                   fraction: 0.95))
            return nil
        }

        if existing != nil {
            log("runtime is out of date — reinstalling")
        }

        // A virtualenv is bound to the interpreter that created it. If Python
        // was upgraded or removed underneath us, the environment is rubble —
        // rebuild it rather than pip into it.
        if FileManager.default.isExecutableFile(atPath: Paths.venvPython.path) {
            let boundTo = existing?.split(separator: "|").last.map(String.init)
            let interpreterChanged = boundTo != nil && boundTo != interpreter.versionString
            if interpreterChanged || !CommandRunner.run(Paths.venvPython, ["-c", "import sys"]).succeeded {
                log("the existing runtime does not match Python \(interpreter.versionString) — rebuilding it")
                try? FileManager.default.removeItem(at: Paths.runtime)
            }
        }

        if !FileManager.default.isExecutableFile(atPath: Paths.venvPython.path) {
            progress(SetupProgress(headline: "Creating the runtime",
                                   detail: "A private Python environment, kept out of your system.",
                                   fraction: 0.2))
            log("creating virtualenv at \(Paths.runtime.path)")
            try? FileManager.default.removeItem(at: Paths.runtime)
            let result = CommandRunner.run(interpreter.url,
                                           ["-m", "venv", Paths.runtime.path],
                                           onLine: log)
            guard result.succeeded else {
                return SetupFailure(
                    title: "Could not create the Python environment",
                    message: result.tail(6),
                    remedy: .retry)
            }
        }

        progress(SetupProgress(headline: "Installing dependencies",
                               detail: "First run only. This downloads a few hundred MB.",
                               fraction: 0.3))
        let upgrade = CommandRunner.run(Paths.venvPython,
                                        ["-m", "pip", "install", "--upgrade", "pip", "--quiet"],
                                        onLine: log)
        if !upgrade.succeeded { log("pip self-upgrade failed; continuing anyway") }

        // `path[extras]` is a single argv entry — no shell quoting involved, so
        // an app installed under "/Applications/Video Understanding.app" works.
        let target = extras.isEmpty
            ? Paths.bundledSource.path
            : "\(Paths.bundledSource.path)[\(extras)]"

        let install = CommandRunner.run(
            Paths.venvPython,
            ["-m", "pip", "install", "--no-input", "--upgrade", target],
            onLine: { [weak self] line in
                guard let self = self else { return }
                self.log(line)
                // pip's own phases make a decent progress signal.
                if line.hasPrefix("Collecting") || line.hasPrefix("Downloading")
                    || line.hasPrefix("Installing") || line.hasPrefix("Building") {
                    self.progress(SetupProgress(
                        headline: "Installing dependencies",
                        detail: String(line.prefix(110)),
                        fraction: nil))
                }
            })

        guard install.succeeded else {
            // A failed *re*install is survivable: if the previous runtime still
            // imports, run on that and try again next launch — the stamp is
            // deliberately not written. Being offline should not cost you the
            // app you already had working.
            if existing != nil, runtimeIsUsable() {
                log("install failed, but the existing runtime still works — continuing with it")
                return nil
            }
            let output = install.output.lowercased()
            let offline = output.contains("network")
                || output.contains("temporary failure")
                || output.contains("could not find a version")
                || output.contains("failed to establish a new connection")
            return SetupFailure(
                title: "Could not install the dependencies",
                message: offline
                    ? "The download failed. Check your internet connection and try again."
                    : install.tail(8),
                remedy: .retry)
        }

        try? stamp.write(to: Paths.stampFile, atomically: true, encoding: .utf8)
        log("runtime installed (\(Paths.buildID))")
        return nil
    }

    /// Writes a config sized for this Mac, once. After that it belongs to the
    /// user: the app reads it, opens it on request, and never rewrites it.
    private func writeConfigurationIfMissing() {
        guard !FileManager.default.fileExists(atPath: Paths.configFile.path) else { return }
        let plan = ModelPlan.recommended()
        let contents = """
        # Written by the Video Understanding Mac app on first launch, sized for
        # this machine (\(Toolchain.memoryGB) GB RAM). Edit freely — the app
        # never overwrites it. Menu: Video Understanding ▸ Edit Configuration.

        vision_model: \(plan.vision)
        llm_model: \(plan.llm)
        whisper_model: \(plan.whisper)

        video:
          frames_per_scene: \(plan.framesPerScene)
          max_frames: \(plan.maxFrames)

        """
        try? contents.write(to: Paths.configFile, atomically: true, encoding: .utf8)
        log("wrote \(Paths.configFile.path)")
    }

    /// Can the existing virtualenv actually serve the API?
    private func runtimeIsUsable() -> Bool {
        guard FileManager.default.isExecutableFile(atPath: Paths.venvPython.path) else { return false }
        return CommandRunner.run(Paths.venvPython,
                                 ["-c", "import video_understanding, fastapi, uvicorn"]).succeeded
    }

    // MARK: Remedies

    /// Runs `brew install <formula>`, streaming output into the setup log.
    static func homebrewInstall(_ formula: String, log: @escaping (String) -> Void) -> Bool {
        guard let brew = Toolchain.brew() else {
            log("Homebrew is not installed")
            return false
        }
        log("running: brew install \(formula)")
        let result = CommandRunner.run(brew, ["install", formula], onLine: log)
        return result.succeeded
    }
}
