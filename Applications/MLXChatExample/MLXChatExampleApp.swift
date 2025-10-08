//
//  MLXChatExampleApp.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import SwiftUI
import FoundationModels

@main
struct MLXChatExampleApp: App {
    var body: some Scene {
        WindowGroup {
            if #available(iOS 26.0, macOS 26.0, *) {
                StructuredComparisonView(
                    viewModel: StructuredComparisonViewModel(
                        mlxService: MLXService(),
                        foundationService: FoundationModelsService()
                    )
                )
            } else {
                ComparisonChatView(
                    viewModel: ComparisonChatViewModel(
                        mlxService: MLXService(),
                        foundationService: FoundationModelsService()
                    )
                )
            }
        }
    }
}