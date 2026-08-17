// MainMenu.swift
//
// The menu bar, built in code. Everything the app can do that isn't in the web
// UI lives here: opening videos, folders, logs, diagnostics, and the two
// destructive resets.

import AppKit

enum MainMenu {
    static func build(target: AppDelegate) -> NSMenu {
        let mainMenu = NSMenu()
        mainMenu.addItem(appMenu(target: target))
        mainMenu.addItem(fileMenu(target: target))
        mainMenu.addItem(editMenu())
        mainMenu.addItem(viewMenu(target: target))
        mainMenu.addItem(serverMenu(target: target))
        mainMenu.addItem(windowMenu())
        mainMenu.addItem(helpMenu(target: target))
        return mainMenu
    }

    // MARK: Menus

    private static func appMenu(target: AppDelegate) -> NSMenuItem {
        let name = "Video Understanding"
        let menu = NSMenu(title: name)

        menu.addItem(item("About \(name)",
                          action: #selector(NSApplication.orderFrontStandardAboutPanel(_:))))
        menu.addItem(.separator())
        menu.addItem(item("Local Models…", action: #selector(AppDelegate.showModels(_:)), target: target))
        menu.addItem(item("Edit Configuration…",
                          action: #selector(AppDelegate.editConfiguration(_:)), target: target))
        menu.addItem(.separator())

        let services = NSMenu(title: "Services")
        let servicesItem = NSMenuItem(title: "Services", action: nil, keyEquivalent: "")
        servicesItem.submenu = services
        NSApp.servicesMenu = services
        menu.addItem(servicesItem)
        menu.addItem(.separator())

        menu.addItem(item("Hide \(name)", action: #selector(NSApplication.hide(_:)), key: "h"))
        let hideOthers = item("Hide Others",
                              action: #selector(NSApplication.hideOtherApplications(_:)), key: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        menu.addItem(hideOthers)
        menu.addItem(item("Show All", action: #selector(NSApplication.unhideAllApplications(_:))))
        menu.addItem(.separator())
        menu.addItem(item("Quit \(name)", action: #selector(NSApplication.terminate(_:)), key: "q"))

        let container = NSMenuItem()
        container.submenu = menu
        return container
    }

    private static func fileMenu(target: AppDelegate) -> NSMenuItem {
        let menu = NSMenu(title: "File")
        menu.addItem(item("Analyse Video…", action: #selector(AppDelegate.openVideo(_:)),
                          key: "o", target: target))
        menu.addItem(.separator())
        menu.addItem(item("Open Library Folder",
                          action: #selector(AppDelegate.openLibraryFolder(_:)), target: target))
        menu.addItem(item("Open Exports Folder",
                          action: #selector(AppDelegate.openExportsFolder(_:)), target: target))
        menu.addItem(.separator())
        menu.addItem(item("Close Window", action: #selector(NSWindow.performClose(_:)), key: "w"))

        let container = NSMenuItem(title: "File", action: nil, keyEquivalent: "")
        container.submenu = menu
        return container
    }

    private static func editMenu() -> NSMenuItem {
        let menu = NSMenu(title: "Edit")
        menu.addItem(item("Undo", action: Selector(("undo:")), key: "z"))
        let redo = item("Redo", action: Selector(("redo:")), key: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(redo)
        menu.addItem(.separator())
        menu.addItem(item("Cut", action: #selector(NSText.cut(_:)), key: "x"))
        menu.addItem(item("Copy", action: #selector(NSText.copy(_:)), key: "c"))
        menu.addItem(item("Paste", action: #selector(NSText.paste(_:)), key: "v"))
        menu.addItem(item("Select All", action: #selector(NSText.selectAll(_:)), key: "a"))

        let container = NSMenuItem(title: "Edit", action: nil, keyEquivalent: "")
        container.submenu = menu
        return container
    }

    private static func viewMenu(target: AppDelegate) -> NSMenuItem {
        let menu = NSMenu(title: "View")
        menu.addItem(item("Reload", action: #selector(AppDelegate.reloadUI(_:)), key: "r", target: target))
        menu.addItem(.separator())
        menu.addItem(item("Actual Size", action: #selector(AppDelegate.zoomReset(_:)), key: "0", target: target))
        menu.addItem(item("Zoom In", action: #selector(AppDelegate.zoomIn(_:)), key: "+", target: target))
        menu.addItem(item("Zoom Out", action: #selector(AppDelegate.zoomOut(_:)), key: "-", target: target))
        menu.addItem(.separator())
        let fullScreen = item("Enter Full Screen",
                              action: #selector(NSWindow.toggleFullScreen(_:)), key: "f")
        fullScreen.keyEquivalentModifierMask = [.command, .control]
        menu.addItem(fullScreen)

        let container = NSMenuItem(title: "View", action: nil, keyEquivalent: "")
        container.submenu = menu
        return container
    }

    private static func serverMenu(target: AppDelegate) -> NSMenuItem {
        let menu = NSMenu(title: "Server")
        menu.addItem(item("Diagnostics…", action: #selector(AppDelegate.showDiagnostics(_:)),
                          key: "d", target: target))
        menu.addItem(item("Server Log", action: #selector(AppDelegate.showServerLog(_:)), target: target))
        menu.addItem(item("Setup Log", action: #selector(AppDelegate.showSetupLog(_:)), target: target))
        menu.addItem(.separator())
        menu.addItem(item("Restart Server", action: #selector(AppDelegate.restartServer(_:)),
                          key: "R", target: target))
        menu.addItem(item("Reinstall Runtime…", action: #selector(AppDelegate.reinstallRuntime(_:)),
                          target: target))
        menu.addItem(.separator())
        menu.addItem(item("Reset Library…", action: #selector(AppDelegate.resetLibrary(_:)),
                          target: target))

        let container = NSMenuItem(title: "Server", action: nil, keyEquivalent: "")
        container.submenu = menu
        return container
    }

    private static func windowMenu() -> NSMenuItem {
        let menu = NSMenu(title: "Window")
        menu.addItem(item("Minimise", action: #selector(NSWindow.performMiniaturize(_:)), key: "m"))
        menu.addItem(item("Zoom", action: #selector(NSWindow.performZoom(_:))))
        menu.addItem(.separator())
        menu.addItem(item("Bring All to Front",
                          action: #selector(NSApplication.arrangeInFront(_:))))
        NSApp.windowsMenu = menu

        let container = NSMenuItem(title: "Window", action: nil, keyEquivalent: "")
        container.submenu = menu
        return container
    }

    private static func helpMenu(target: AppDelegate) -> NSMenuItem {
        let menu = NSMenu(title: "Help")
        menu.addItem(item("API Documentation", action: #selector(AppDelegate.openAPIDocs(_:)),
                          target: target))
        menu.addItem(item("Project README", action: #selector(AppDelegate.openReadme(_:)),
                          target: target))
        NSApp.helpMenu = menu

        let container = NSMenuItem(title: "Help", action: nil, keyEquivalent: "")
        container.submenu = menu
        return container
    }

    // MARK: Helper

    private static func item(
        _ title: String,
        action: Selector?,
        key: String = "",
        target: AnyObject? = nil
    ) -> NSMenuItem {
        let menuItem = NSMenuItem(title: title, action: action, keyEquivalent: key)
        if let target = target { menuItem.target = target }
        return menuItem
    }
}
