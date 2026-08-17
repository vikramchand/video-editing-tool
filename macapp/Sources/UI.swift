// UI.swift
//
// Small AppKit helpers. Everything is built in code — the app has no nib, so
// `swiftc` alone can produce the whole bundle.

import AppKit

enum UI {
    static func label(
        _ text: String,
        size: CGFloat = 13,
        weight: NSFont.Weight = .regular,
        color: NSColor = .labelColor
    ) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = NSFont.systemFont(ofSize: size, weight: weight)
        field.textColor = color
        field.lineBreakMode = .byWordWrapping
        field.maximumNumberOfLines = 0
        field.translatesAutoresizingMaskIntoConstraints = false
        field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return field
    }

    static func button(
        _ title: String,
        target: AnyObject?,
        action: Selector,
        isDefault: Bool = false
    ) -> NSButton {
        let button = NSButton(title: title, target: target, action: action)
        button.bezelStyle = .rounded
        button.translatesAutoresizingMaskIntoConstraints = false
        if isDefault {
            button.keyEquivalent = "\r"
        }
        return button
    }

    /// A read-only, monospaced, scrolling text view for logs.
    static func logView() -> (scroll: NSScrollView, text: NSTextView) {
        let scroll = NSScrollView()
        scroll.translatesAutoresizingMaskIntoConstraints = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .lineBorder
        scroll.drawsBackground = true

        let text = NSTextView()
        text.isEditable = false
        text.isSelectable = true
        text.drawsBackground = true
        text.backgroundColor = .textBackgroundColor
        text.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        text.textColor = .secondaryLabelColor
        text.isVerticallyResizable = true
        text.isHorizontallyResizable = false
        text.autoresizingMask = [.width]
        text.textContainerInset = NSSize(width: 6, height: 6)
        text.minSize = NSSize(width: 0, height: 0)
        text.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                              height: CGFloat.greatestFiniteMagnitude)
        text.textContainer?.widthTracksTextView = true
        text.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)

        scroll.documentView = text
        return (scroll, text)
    }

    static func append(_ line: String, to textView: NSTextView, follow: Bool = true) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
            .foregroundColor: NSColor.secondaryLabelColor,
        ]
        textView.textStorage?.append(NSAttributedString(string: line + "\n", attributes: attributes))
        if follow { textView.scrollToEndOfDocument(nil) }
    }

    static func setText(_ text: String, in textView: NSTextView) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
            .foregroundColor: NSColor.secondaryLabelColor,
        ]
        textView.textStorage?.setAttributedString(
            NSAttributedString(string: text, attributes: attributes))
        textView.scrollToEndOfDocument(nil)
    }

    static func window(
        title: String,
        width: CGFloat,
        height: CGFloat,
        resizable: Bool = false
    ) -> NSWindow {
        var style: NSWindow.StyleMask = [.titled, .closable, .miniaturizable]
        if resizable { style.insert(.resizable) }
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: width, height: height),
            styleMask: style,
            backing: .buffered,
            defer: false)
        window.title = title
        window.isReleasedWhenClosed = false
        window.center()
        return window
    }

    static func alert(
        _ messageText: String,
        informative: String,
        style: NSAlert.Style = .informational,
        buttons: [String] = ["OK"]
    ) -> NSAlert {
        let alert = NSAlert()
        alert.messageText = messageText
        alert.informativeText = informative
        alert.alertStyle = style
        for title in buttons { alert.addButton(withTitle: title) }
        return alert
    }
}
