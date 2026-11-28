//
// DynamicNotchPanel.swift
// DynamicNotchKit
//
// Created by <Huy D.> on 2024-11-01.
//

import AppKit

final class DynamicNotchPanel: NSPanel {
    override init(
        contentRect: NSRect,
        styleMask style: NSWindow.StyleMask,
        backing backingStoreType: NSWindow.BackingStoreType,
        defer flag: Bool
    ) {
        super.init(
            contentRect: contentRect,
            styleMask: style,
            backing: backingStoreType,
            defer: flag
        )
        self.hasShadow = false
        self.backgroundColor = .clear
        // 本地修改（CalendarBoard）：原为 .screenSaver，会盖住输入法候选框和系统弹出菜单。
        // 只需高于菜单栏即可覆盖刘海区域。
        self.level = NSWindow.Level(rawValue: NSWindow.Level.mainMenu.rawValue + 3)
        self.collectionBehavior = [.canJoinAllSpaces, .stationary]
    }

    override var canBecomeKey: Bool {
        true
    }
}
