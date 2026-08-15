// main.swift
//
// Entry point. Top-level code lives here and nowhere else, which is what lets
// the whole app compile with a plain `swiftc` invocation.

import AppKit

let application = NSApplication.shared
let appDelegate = AppDelegate()
application.delegate = appDelegate
application.setActivationPolicy(.regular)
application.run()
