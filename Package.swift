// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package: Package = .init(
    name: "ONNXRuntime",
    products: [
        .library(name: "CONNX", targets: ["CONNX"]),
        .library(name: "ONNXRuntime", targets: ["ONNXRuntime"])
    ],
    dependencies: [],
    targets: [
        .systemLibrary(name: "CONNX"),

        .target(
            name: "ONNXRuntime",
            dependencies: [
                "CONNX"
            ],
            linkerSettings: [
                .unsafeFlags(["-L/home/rameshravone/Works/Libs/onnxruntime/lib"])
            ]
        )
    ]
)
