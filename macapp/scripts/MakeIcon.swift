// MakeIcon.swift
//
// Draws the app icon at every size macOS asks for and writes an .iconset
// directory. Run by build.sh:
//
//     swift MakeIcon.swift /path/to/AppIcon.iconset
//
// Core Graphics does the anti-aliasing, so the app has no binary image assets
// checked into the repository.

import AppKit
import CoreGraphics
import Foundation

/// One image, drawn at an arbitrary pixel size.
func drawIcon(pixels: Int) -> CGImage? {
    let size = CGFloat(pixels)
    guard let context = CGContext(
        data: nil,
        width: pixels,
        height: pixels,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    context.setAllowsAntialiasing(true)
    context.interpolationQuality = .high

    // macOS app icons sit inside a rounded square with generous margins.
    let inset = size * 0.055
    let plate = CGRect(x: inset, y: inset, width: size - inset * 2, height: size - inset * 2)
    let radius = plate.width * 0.2237 // the standard "squircle" approximation

    let platePath = CGPath(roundedRect: plate, cornerWidth: radius, cornerHeight: radius, transform: nil)
    context.saveGState()
    context.addPath(platePath)
    context.clip()

    let colors = [
        CGColor(red: 0.36, green: 0.58, blue: 1.00, alpha: 1.0),
        CGColor(red: 0.15, green: 0.28, blue: 0.78, alpha: 1.0),
    ] as CFArray
    if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                 colors: colors,
                                 locations: [0.0, 1.0]) {
        context.drawLinearGradient(
            gradient,
            start: CGPoint(x: plate.minX, y: plate.maxY),
            end: CGPoint(x: plate.maxX, y: plate.minY),
            options: [])
    }
    context.restoreGState()

    // Play triangle, sitting a little above centre to leave room for the
    // timeline ticks — the two things the app is actually about.
    let triangleHeight = plate.height * 0.38
    let triangleWidth = triangleHeight * 0.88
    let centre = CGPoint(x: plate.midX, y: plate.midY + plate.height * 0.06)
    let corner = triangleHeight * 0.10

    context.saveGState()
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.97))
    context.setLineWidth(corner * 2)
    context.setLineJoin(.round)
    context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.97))
    context.beginPath()
    context.move(to: CGPoint(x: centre.x - triangleWidth * 0.34, y: centre.y + triangleHeight / 2))
    context.addLine(to: CGPoint(x: centre.x + triangleWidth * 0.66, y: centre.y))
    context.addLine(to: CGPoint(x: centre.x - triangleWidth * 0.34, y: centre.y - triangleHeight / 2))
    context.closePath()
    context.drawPath(using: .fillStroke)
    context.restoreGState()

    // Timeline: three segments of different weight, the way a scene strip looks.
    let barHeight = plate.height * 0.055
    let barY = plate.minY + plate.height * 0.145
    let barInset = plate.width * 0.18
    let available = plate.width - barInset * 2
    let gap = available * 0.06
    let widths = [available * 0.44, available * 0.24, available * 0.20]
    let alphas: [CGFloat] = [0.95, 0.6, 0.85]

    var x = plate.minX + barInset
    for (index, width) in widths.enumerated() {
        let rect = CGRect(x: x, y: barY, width: width, height: barHeight)
        let path = CGPath(roundedRect: rect,
                          cornerWidth: barHeight / 2,
                          cornerHeight: barHeight / 2,
                          transform: nil)
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: alphas[index]))
        context.addPath(path)
        context.fillPath()
        x += width + gap
    }

    return context.makeImage()
}

func writePNG(_ image: CGImage, to url: URL) throws {
    let rep = NSBitmapImageRep(cgImage: image)
    rep.size = NSSize(width: image.width, height: image.height)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "MakeIcon", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "could not encode PNG"])
    }
    try data.write(to: url)
}

// MARK: - Entry point

let arguments = CommandLine.arguments
guard arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: MakeIcon.swift <output.iconset>\n".utf8))
    exit(2)
}

let outputDirectory = URL(fileURLWithPath: arguments[1])
try? FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

/// (logical size, scale) pairs, exactly what iconutil expects.
let variants: [(Int, Int)] = [
    (16, 1), (16, 2),
    (32, 1), (32, 2),
    (128, 1), (128, 2),
    (256, 1), (256, 2),
    (512, 1), (512, 2),
]

for (points, scale) in variants {
    let pixels = points * scale
    guard let image = drawIcon(pixels: pixels) else {
        FileHandle.standardError.write(Data("failed to draw \(pixels)px icon\n".utf8))
        exit(1)
    }
    let suffix = scale == 1 ? "" : "@\(scale)x"
    let name = "icon_\(points)x\(points)\(suffix).png"
    do {
        try writePNG(image, to: outputDirectory.appendingPathComponent(name))
    } catch {
        FileHandle.standardError.write(Data("failed to write \(name): \(error)\n".utf8))
        exit(1)
    }
}
