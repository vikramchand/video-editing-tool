// Support.swift
//
// Filesystem layout, subprocess plumbing and logging — everything the rest of
// the app needs before it can think about windows.

import Darwin
import Foundation

// MARK: - Paths

/// Where the app keeps the things it owns.
///
/// The bundle itself is treated as read-only: the Python source ships inside
/// `Contents/Resources/app-source`, and every mutable thing (the virtualenv,
/// the library, logs, config) lives under Application Support so that
/// replacing the app never destroys a user's data.
enum Paths {
    static let folderName = "VideoUnderstanding"

    static var support: URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent(folderName, isDirectory: true)
    }

    static var runtime: URL { support.appendingPathComponent("runtime", isDirectory: true) }
    static var venvBin: URL { runtime.appendingPathComponent("bin", isDirectory: true) }
    static var venvPython: URL { venvBin.appendingPathComponent("python3") }
    static var dataDir: URL { support.appendingPathComponent("data", isDirectory: true) }
    static var outputDir: URL { support.appendingPathComponent("output", isDirectory: true) }
    static var logsDir: URL { support.appendingPathComponent("logs", isDirectory: true) }
    static var serverLog: URL { logsDir.appendingPathComponent("server.log") }
    static var setupLog: URL { logsDir.appendingPathComponent("setup.log") }
    static var stampFile: URL { runtime.appendingPathComponent(".installed") }
    static var configFile: URL { support.appendingPathComponent("config.yaml") }

    /// The Python package, staged into the bundle at build time.
    static var bundledSource: URL {
        (Bundle.main.resourceURL ?? Bundle.main.bundleURL)
            .appendingPathComponent("app-source", isDirectory: true)
    }

    /// Identifies the exact source revision inside the bundle, so upgrading the
    /// app reinstalls the runtime instead of silently running last week's code.
    static var buildID: String {
        let file = bundledSource.appendingPathComponent("BUILD_ID")
        let raw = (try? String(contentsOf: file, encoding: .utf8)) ?? "unknown"
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static var appVersion: String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0"
    }

    static func ensureDirectories() {
        for directory in [support, dataDir, outputDir, logsDir] {
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
}

// MARK: - Line buffering

/// Turns an arbitrary byte stream into whole lines.
///
/// `\r` terminates a line as well as `\n`: `ollama pull` and `pip` both draw
/// progress by rewriting the current line, and without this they would arrive
/// as one enormous string at the end.
final class LineAccumulator {
    private var partial = ""
    private let emit: (String) -> Void

    init(emit: @escaping (String) -> Void) {
        self.emit = emit
    }

    func feed(_ data: Data) {
        guard let text = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .isoLatin1) else { return }
        partial += text
        while let index = partial.firstIndex(where: { $0 == "\n" || $0 == "\r" }) {
            let line = String(partial[partial.startIndex..<index])
            partial = String(partial[partial.index(after: index)...])
            if !line.isEmpty { emit(line) }
        }
    }

    func flush() {
        let remainder = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        partial = ""
        if !remainder.isEmpty { emit(remainder) }
    }
}

// MARK: - Running commands

struct CommandResult {
    let status: Int32
    let output: String

    var succeeded: Bool { status == 0 }

    /// The tail of the output, for error messages that should stay readable.
    func tail(_ lines: Int = 12) -> String {
        let all = output.split(separator: "\n", omittingEmptySubsequences: false)
        return all.suffix(lines).joined(separator: "\n")
    }
}

enum CommandRunner {
    /// Runs a command to completion, streaming its output line by line.
    ///
    /// `onLine` is invoked on a background queue — callers that touch the UI
    /// must hop to the main queue themselves.
    @discardableResult
    static func run(
        _ executable: URL,
        _ arguments: [String],
        environment: [String: String]? = nil,
        currentDirectory: URL? = nil,
        onLine: ((String) -> Void)? = nil
    ) -> CommandResult {
        // A Process whose working directory does not exist fails to launch.
        Paths.ensureDirectories()

        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment ?? Toolchain.baseEnvironment()
        process.currentDirectoryURL = currentDirectory ?? Paths.support
        process.standardInput = FileHandle.nullDevice

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        let lock = NSLock()
        var collected = ""
        let accumulator = LineAccumulator { line in
            lock.lock()
            collected += line + "\n"
            lock.unlock()
            onLine?(line)
        }

        let handle = pipe.fileHandleForReading
        let drained = DispatchSemaphore(value: 0)
        handle.readabilityHandler = { fileHandle in
            let data = fileHandle.availableData
            if data.isEmpty {
                fileHandle.readabilityHandler = nil
                drained.signal()
            } else {
                accumulator.feed(data)
            }
        }

        do {
            try process.run()
        } catch {
            handle.readabilityHandler = nil
            let message = "could not launch \(executable.path): \(error.localizedDescription)"
            onLine?(message)
            return CommandResult(status: 127, output: message)
        }

        process.waitUntilExit()
        _ = drained.wait(timeout: .now() + 5)
        handle.readabilityHandler = nil
        accumulator.flush()

        lock.lock()
        let output = collected
        lock.unlock()
        return CommandResult(status: process.terminationStatus, output: output)
    }
}

// MARK: - HTTP

enum HTTP {
    /// A blocking GET. Only ever called from background queues.
    static func get(_ url: URL, timeout: TimeInterval = 3) -> (status: Int, data: Data)? {
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        var result: (status: Int, data: Data)?
        let semaphore = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: request) { data, response, _ in
            if let http = response as? HTTPURLResponse {
                result = (http.statusCode, data ?? Data())
            }
            semaphore.signal()
        }
        task.resume()
        if semaphore.wait(timeout: .now() + timeout + 2) == .timedOut {
            task.cancel()
            return nil
        }
        return result
    }
}

// MARK: - Ports

enum Ports {
    /// Preferred range, deliberately away from 8000 so a hand-run `make run`
    /// and the app can coexist.
    static let preferred = Array(8756...8790)

    static func pick() -> Int {
        for port in preferred where isFree(port) { return port }
        return 0 // let uvicorn fail loudly rather than guess
    }

    static func isFree(_ port: Int) -> Bool {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }

        var reuse: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = UInt16(port).bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                bind(descriptor, raw, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bound == 0
    }
}

// MARK: - Logging

/// An append-only log with a bounded in-memory tail, mirrored to disk.
final class LogStore {
    static let setup = LogStore(fileURL: Paths.setupLog)
    static let server = LogStore(fileURL: Paths.serverLog)

    private var observers: [UUID: (String) -> Void] = [:]
    private let fileURL: URL
    private let lock = NSLock()
    private var buffer: [String] = []
    private let limit = 1200
    private var handle: FileHandle?

    init(fileURL: URL) {
        self.fileURL = fileURL
    }

    /// Registers a live tail; the returned token removes it again.
    /// Observers are always called on the main queue.
    @discardableResult
    func addObserver(_ observer: @escaping (String) -> Void) -> UUID {
        let token = UUID()
        lock.lock()
        observers[token] = observer
        lock.unlock()
        return token
    }

    func removeObserver(_ token: UUID) {
        lock.lock()
        observers.removeValue(forKey: token)
        lock.unlock()
    }

    func append(_ line: String) {
        lock.lock()
        buffer.append(line)
        if buffer.count > limit { buffer.removeFirst(buffer.count - limit) }
        let current = Array(observers.values)
        lock.unlock()

        write(line + "\n")
        if current.isEmpty { return }
        DispatchQueue.main.async {
            for observer in current { observer(line) }
        }
    }

    func text() -> String {
        lock.lock()
        defer { lock.unlock() }
        return buffer.joined(separator: "\n")
    }

    func tail(_ count: Int) -> String {
        lock.lock()
        defer { lock.unlock() }
        return buffer.suffix(count).joined(separator: "\n")
    }

    /// Starts a new session in the file, so a log is about one run.
    func rotate(header: String) {
        Paths.ensureDirectories()
        lock.lock()
        buffer.removeAll()
        handle?.closeFile()
        handle = nil
        lock.unlock()

        // Atomic write replaces the inode, which is exactly why the handle is
        // closed first — the next append reopens the new file.
        try? "".write(to: fileURL, atomically: true, encoding: .utf8)
        append(header)
    }

    private func write(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        lock.lock()
        defer { lock.unlock() }
        if handle == nil {
            Paths.ensureDirectories()
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                FileManager.default.createFile(atPath: fileURL.path, contents: nil)
            }
            handle = try? FileHandle(forWritingTo: fileURL)
            handle?.seekToEndOfFile()
        }
        handle?.write(data)
    }
}
