// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GomokuIsland",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "Vendor/DynamicNotchKit")
    ],
    targets: [
        .executableTarget(
            name: "GomokuIsland",
            dependencies: [
                "DynamicNotchKit"
            ],
            path: "Sources/GomokuIsland",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
