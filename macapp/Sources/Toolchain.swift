// Toolchain.swift
//
// Finding the things the pipeline depends on: a usable Python, FFmpeg,
// Homebrew, and Ollama. A GUI app launched from Finder inherits almost no
// PATH, so every lookup here searches known install locations explicitly
// rather than trusting the environment.

import Darwin
import Foundation

struct PythonInterpreter {
    let url: URL
    let major: Int
    let minor: Int
    let patch: Int

    var versionString: String { "\(major).\(minor).\(patch)" }
    var isSupported: Bool { major == 3 && minor >= 12 }
}

enum Toolchain {
    /// Directories a Finder-launched app would otherwise never see.
    static let searchPaths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/homebrew/sbin",
        "/usr/local/sbin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]

    /// PATH for every subprocess we spawn, including the API server — this is
    /// how the Python code finds `ffmpeg` and `ffprobe`.
    static func augmentedPath() -> String {
        var parts = searchPaths
        if let inherited = ProcessInfo.processInfo.environment["PATH"] {
            for component in inherited.split(separator: ":").map(String.init)
            where !parts.contains(component) {
                parts.append(component)
            }
        }
        return parts.joined(separator: ":")
    }

    /// A clean environment: inherited virtualenv/Python variables are stripped
    /// so that launching from a terminal with some other venv active cannot
    /// poison the app's runtime.
    static func baseEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        for key in ["VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "CONDA_PREFIX"] {
            environment.removeValue(forKey: key)
        }
        environment["PATH"] = augmentedPath()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return environment
    }

    /// Locates an executable by name across the known directories.
    static func find(_ name: String) -> URL? {
        for directory in searchPaths {
            let candidate = URL(fileURLWithPath: directory).appendingPathComponent(name)
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        if let inherited = ProcessInfo.processInfo.environment["PATH"] {
            for directory in inherited.split(separator: ":").map(String.init) {
                let candidate = URL(fileURLWithPath: directory).appendingPathComponent(name)
                if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
            }
        }
        return nil
    }

    // MARK: Python

    /// Interpreters to try, best first.
    ///
    /// 3.12 leads deliberately: it is what the project targets and what has
    /// wheels for every optional dependency, including CTranslate2 (Whisper)
    /// and OpenCV (scene detection). Newer versions are tried after it.
    private static var pythonCandidates: [URL] {
        var candidates: [URL] = []
        let versions = ["3.12", "3.13", "3.14"]
        for prefix in ["/opt/homebrew/bin", "/usr/local/bin"] {
            for version in versions {
                candidates.append(URL(fileURLWithPath: "\(prefix)/python\(version)"))
            }
        }
        for version in versions {
            candidates.append(URL(fileURLWithPath:
                "/Library/Frameworks/Python.framework/Versions/\(version)/bin/python3"))
        }
        for prefix in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
            candidates.append(URL(fileURLWithPath: "\(prefix)/python3"))
        }
        if let onPath = find("python3") { candidates.append(onPath) }
        return candidates
    }

    /// The first interpreter that is 3.12+ *and* can build a virtualenv.
    static func findPython() -> PythonInterpreter? {
        var seen = Set<String>()
        for candidate in pythonCandidates {
            let path = (try? FileManager.default.destinationOfSymbolicLink(atPath: candidate.path))
                .map { $0.hasPrefix("/") ? $0 : candidate.deletingLastPathComponent().appendingPathComponent($0).path }
                ?? candidate.path
            guard !seen.contains(path) else { continue }
            seen.insert(path)
            guard FileManager.default.isExecutableFile(atPath: candidate.path) else { continue }
            guard let interpreter = probe(candidate), interpreter.isSupported else { continue }
            guard canCreateVirtualEnvironments(interpreter.url) else { continue }
            return interpreter
        }
        return nil
    }

    /// Any 3.x interpreter we could find, supported or not — used to explain
    /// *why* setup failed ("found 3.9, need 3.12+") instead of just "no Python".
    static func findAnyPython() -> PythonInterpreter? {
        for candidate in pythonCandidates {
            guard FileManager.default.isExecutableFile(atPath: candidate.path) else { continue }
            if let interpreter = probe(candidate) { return interpreter }
        }
        return nil
    }

    private static func probe(_ url: URL) -> PythonInterpreter? {
        let script = "import sys; print('%d %d %d' % sys.version_info[:3])"
        let result = CommandRunner.run(url, ["-c", script])
        guard result.succeeded else { return nil }
        let numbers = result.output
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: " ")
            .compactMap { Int($0) }
        guard numbers.count == 3 else { return nil }
        return PythonInterpreter(url: url, major: numbers[0], minor: numbers[1], patch: numbers[2])
    }

    private static func canCreateVirtualEnvironments(_ url: URL) -> Bool {
        CommandRunner.run(url, ["-c", "import venv, ensurepip"]).succeeded
    }

    // MARK: Other tools

    static func ffmpeg() -> URL? { find("ffmpeg") }
    static func ffprobe() -> URL? { find("ffprobe") }
    static func brew() -> URL? { find("brew") }

    static func ffmpegVersion() -> String? {
        guard let binary = ffmpeg() else { return nil }
        let result = CommandRunner.run(binary, ["-version"])
        guard result.succeeded else { return nil }
        return result.output.split(separator: "\n").first.map(String.init)
    }

    /// Physical RAM in gigabytes, used to size the model recommendation.
    static var memoryGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    static var hardwareDescription: String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let name = String(cString: model)
        #if arch(arm64)
        let architecture = "Apple Silicon"
        #else
        let architecture = "Intel"
        #endif
        return name.isEmpty ? architecture : "\(architecture) · \(name) · \(memoryGB) GB RAM"
    }
}

// MARK: - Models

/// The model set recommended for this machine, following the sizing table in
/// the project README.
struct ModelPlan {
    let vision: String
    let llm: String
    let whisper: String

    static func recommended(forGB gigabytes: Int = Toolchain.memoryGB) -> ModelPlan {
        if gigabytes >= 60 {
            return ModelPlan(vision: "qwen2.5vl:32b", llm: "qwen3:14b", whisper: "medium")
        }
        if gigabytes >= 28 {
            return ModelPlan(vision: "qwen2.5vl:7b", llm: "qwen3:8b", whisper: "small")
        }
        return ModelPlan(vision: "qwen2.5vl:3b", llm: "qwen3:4b", whisper: "base")
    }

    /// Frame budget matters more than model size on small machines; mirror the
    /// README's 16 GB advice in the generated config.
    var framesPerScene: Int { vision.hasSuffix(":3b") ? 2 : 3 }
    var maxFrames: Int { vision.hasSuffix(":3b") ? 40 : 100 }
}

// MARK: - Ollama

/// Ollama is a separate process the app can install, start and pull models
/// with — but never requires. Without it the pipeline degrades to heuristics
/// and says so in `warnings`.
final class OllamaManager {
    static let shared = OllamaManager()

    let host = "http://localhost:11434"
    private var managedProcess: Process?

    private init() {}

    var binary: URL? { Toolchain.find("ollama") }

    /// True when something is answering on the Ollama port.
    func isRunning(timeout: TimeInterval = 2) -> Bool {
        guard let url = URL(string: "\(host)/api/tags") else { return false }
        guard let response = HTTP.get(url, timeout: timeout) else { return false }
        return response.status == 200
    }

    /// Model names currently installed, e.g. ["qwen3:8b", "qwen2.5vl:7b"].
    func installedModels() -> [String] {
        guard let url = URL(string: "\(host)/api/tags"),
              let response = HTTP.get(url, timeout: 4),
              response.status == 200,
              let json = try? JSONSerialization.jsonObject(with: response.data) as? [String: Any],
              let models = json["models"] as? [[String: Any]] else { return [] }
        return models.compactMap { $0["name"] as? String }
    }

    func isInstalled(_ model: String) -> Bool {
        let installed = installedModels()
        if installed.contains(model) { return true }
        // `ollama list` reports "qwen3:8b"; a request for "qwen3" should match.
        return installed.contains { $0.split(separator: ":").first.map(String.init) == model }
    }

    /// Starts Ollama if it is installed but not listening.
    ///
    /// Prefers the Ollama.app when present (it manages its own menu-bar agent);
    /// otherwise runs `ollama serve` as a child of this app.
    @discardableResult
    func startIfNeeded(log: @escaping (String) -> Void) -> Bool {
        if isRunning() { return true }
        guard let binary = binary else { return false }

        log("starting ollama…")
        let process = Process()
        process.executableURL = binary
        process.arguments = ["serve"]
        process.environment = Toolchain.baseEnvironment()
        process.standardInput = FileHandle.nullDevice

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let accumulator = LineAccumulator { line in log("ollama: \(line)") }
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil } else { accumulator.feed(data) }
        }

        do {
            try process.run()
        } catch {
            log("could not start ollama: \(error.localizedDescription)")
            return false
        }
        managedProcess = process

        for _ in 0..<40 {
            if isRunning(timeout: 1) {
                log("ollama is ready")
                return true
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        log("ollama did not come up in time")
        return false
    }

    /// Downloads a model, streaming progress lines.
    func pull(_ model: String, log: @escaping (String) -> Void) -> Bool {
        guard let binary = binary else {
            log("ollama is not installed")
            return false
        }
        log("pulling \(model) — this is the slow step, several GB")
        let result = CommandRunner.run(binary, ["pull", model], onLine: log)
        if result.succeeded {
            log("\(model) is ready")
        } else {
            log("failed to pull \(model) (exit \(result.status))")
        }
        return result.succeeded
    }

    /// Only stops an Ollama we started ourselves — never one the user runs.
    func stopManagedServer() {
        guard let process = managedProcess, process.isRunning else { return }
        process.terminate()
        managedProcess = nil
    }
}
