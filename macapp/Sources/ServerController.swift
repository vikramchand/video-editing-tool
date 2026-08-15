// ServerController.swift
//
// Owns the uvicorn child process: starting it on a free loopback port,
// waiting for /health, restarting it, and making sure it dies with the app.

import Darwin
import Foundation

struct ServerError: Error {
    let title: String
    let message: String
}

final class ServerController {
    private(set) var port: Int = 0
    private var process: Process?
    private let stateLock = NSLock()

    /// Called on the main queue when the server exits without being asked to.
    var onUnexpectedExit: ((Int32) -> Void)?

    private var stopping = false

    var isRunning: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return process?.isRunning ?? false
    }

    var baseURL: URL? {
        guard port > 0 else { return nil }
        return URL(string: "http://127.0.0.1:\(port)")
    }

    var webURL: URL? { baseURL }

    // MARK: Lifecycle

    /// Launches the API and blocks until `/health` answers. Background queue only.
    func start(python: URL) -> Result<URL, ServerError> {
        stop()

        stateLock.lock()
        stopping = false
        stateLock.unlock()

        port = Ports.pick()
        guard port > 0, let baseURL = baseURL else {
            return .failure(ServerError(
                title: "No free port",
                message: "Every port in the app's range is in use. Quit whatever is using them and try again."))
        }

        LogStore.server.rotate(header: "— starting server on 127.0.0.1:\(port) —")

        var environment = Toolchain.baseEnvironment()
        environment["HOST"] = "127.0.0.1"
        environment["PORT"] = String(port)
        environment["DATA_DIR"] = Paths.dataDir.path
        environment["OUTPUT_DIR"] = Paths.outputDir.path
        environment["CONFIG_FILE"] = Paths.configFile.path
        environment["LOG_LEVEL"] = ProcessInfo.processInfo.environment["VU_LOG_LEVEL"] ?? "INFO"
        // The web UI is same-origin; no other origin should reach this server.
        environment["CORS_ORIGINS"] = "http://127.0.0.1:\(port),http://localhost:\(port)"

        let process = Process()
        process.executableURL = Paths.venvPython
        process.arguments = [
            "-m", "uvicorn",
            "video_understanding.api.app:app",
            "--host", "127.0.0.1",
            "--port", String(port),
        ]
        process.environment = environment
        // Run from Application Support so a stray .env in some other directory
        // can never change how the app behaves.
        process.currentDirectoryURL = Paths.support
        process.standardInput = FileHandle.nullDevice

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let accumulator = LineAccumulator { LogStore.server.append($0) }
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
            } else {
                accumulator.feed(data)
            }
        }

        process.terminationHandler = { [weak self] finished in
            guard let self = self else { return }
            self.stateLock.lock()
            let deliberate = self.stopping
            self.stateLock.unlock()
            guard !deliberate else { return }
            let status = finished.terminationStatus
            DispatchQueue.main.async { self.onUnexpectedExit?(status) }
        }

        do {
            try process.run()
        } catch {
            return .failure(ServerError(
                title: "Could not start the server",
                message: error.localizedDescription))
        }

        stateLock.lock()
        self.process = process
        stateLock.unlock()

        // Importing FastAPI, Whisper and OpenCV takes a few seconds on a cold
        // filesystem cache, so the window here is generous.
        let deadline = Date().addingTimeInterval(120)
        while Date() < deadline {
            if !process.isRunning {
                return .failure(ServerError(
                    title: "The server stopped during start-up",
                    message: LogStore.server.tail(12)))
            }
            if let health = HTTP.get(baseURL.appendingPathComponent("health"), timeout: 2),
               health.status == 200 {
                LogStore.server.append("— health check passed —")
                return .success(baseURL)
            }
            Thread.sleep(forTimeInterval: 0.4)
        }

        return .failure(ServerError(
            title: "The server did not respond",
            message: "It never answered on 127.0.0.1:\(port).\n\n" + LogStore.server.tail(10)))
    }

    /// SIGTERM, then SIGKILL if uvicorn is being stubborn.
    func stop() {
        stateLock.lock()
        let running = process
        stopping = true
        process = nil
        stateLock.unlock()

        guard let running = running, running.isRunning else { return }
        running.terminate()

        let deadline = Date().addingTimeInterval(5)
        while running.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if running.isRunning {
            kill(running.processIdentifier, SIGKILL)
        }
    }

    // MARK: Introspection

    /// The parsed `/health` payload, or nil if the server is unreachable.
    func health() -> [String: Any]? {
        guard let baseURL = baseURL,
              let response = HTTP.get(baseURL.appendingPathComponent("health"), timeout: 4),
              response.status == 200,
              let json = try? JSONSerialization.jsonObject(with: response.data) as? [String: Any]
        else { return nil }
        return json
    }
}

// MARK: - Uploading

enum Uploader {
    /// Posts a file to `/v1/videos` exactly as the web UI does.
    ///
    /// The multipart body is assembled on disk rather than in memory: videos
    /// are large, and `uploadTask(with:fromFile:)` streams it.
    static func upload(
        _ fileURL: URL,
        to baseURL: URL,
        completion: @escaping (Result<String, Error>) -> Void
    ) {
        let boundary = "Boundary-\(UUID().uuidString)"
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("vu-upload-\(UUID().uuidString)")

        do {
            try assembleBody(fileURL: fileURL, boundary: boundary, at: temporary)
        } catch {
            completion(.failure(error))
            return
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("v1/videos"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 600

        let task = URLSession.shared.uploadTask(with: request, fromFile: temporary) { data, response, error in
            try? FileManager.default.removeItem(at: temporary)

            if let error = error {
                completion(.failure(error))
                return
            }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            let payload = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }

            guard (200..<300).contains(status) else {
                let detail = (payload?["detail"] as? String) ?? "the server returned \(status)"
                completion(.failure(UploadError(message: detail)))
                return
            }
            guard let videoID = payload?["video_id"] as? String else {
                completion(.failure(UploadError(message: "the server did not return a video id")))
                return
            }
            completion(.success(videoID))
        }
        task.resume()
    }

    private static func assembleBody(fileURL: URL, boundary: String, at destination: URL) throws {
        let filename = fileURL.lastPathComponent
        let header = "--\(boundary)\r\n"
            + "Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n"
            + "Content-Type: application/octet-stream\r\n\r\n"
        let footer = "\r\n--\(boundary)--\r\n"

        FileManager.default.createFile(atPath: destination.path, contents: nil)
        guard let output = try? FileHandle(forWritingTo: destination) else {
            throw UploadError(message: "could not stage the upload")
        }
        defer { output.closeFile() }

        output.write(Data(header.utf8))

        let input = try FileHandle(forReadingFrom: fileURL)
        defer { input.closeFile() }
        while true {
            let chunk = input.readData(ofLength: 4 * 1024 * 1024)
            if chunk.isEmpty { break }
            output.write(chunk)
        }
        output.write(Data(footer.utf8))
    }
}

struct UploadError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
