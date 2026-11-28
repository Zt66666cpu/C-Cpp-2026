//
//  EnvironmentValues+Extensions.swift
//  DynamicNotchKit
//
//  Created by Kai Azim on 2025-03-26.
//
//  CLT compatibility: the upstream `@Entry` macro requires Xcode's
//  SwiftUIMacros plugin, which is unavailable with Command Line Tools only.
//  These environment values are therefore declared with the classic
//  `EnvironmentKey` form; behavior is identical.
//

import SwiftUI

private struct NotchStyleEnvironmentKey: EnvironmentKey {
    static let defaultValue: DynamicNotchStyle = .auto
}

private struct NotchSectionEnvironmentKey: EnvironmentKey {
    static let defaultValue: DynamicNotchSection = .expanded
}

extension EnvironmentValues {
    var notchStyle: DynamicNotchStyle {
        get { self[NotchStyleEnvironmentKey.self] }
        set { self[NotchStyleEnvironmentKey.self] = newValue }
    }

    var notchSection: DynamicNotchSection {
        get { self[NotchSectionEnvironmentKey.self] }
        set { self[NotchSectionEnvironmentKey.self] = newValue }
    }
}

enum DynamicNotchSection {
    case expanded
    case compactLeading
    case compactTrailing
}
