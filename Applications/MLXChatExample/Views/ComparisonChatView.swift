//
//  ComparisonChatView.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import SwiftUI

/// Main comparison view showing MLX and Foundation Models side by side
struct ComparisonChatView: View {
    @Bindable private var vm: ComparisonChatViewModel
    
    init(viewModel: ComparisonChatViewModel) {
        self.vm = viewModel
    }
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Side-by-side comparison area
                HStack(spacing: 0) {
                    // MLX Side
                    MLXResponseView(
                        response: vm.mlxResponse,
                        isGenerating: vm.mlxIsGenerating,
                        isPreloading: vm.mlxIsPreloading,
                        modelName: vm.selectedMLXModel.name,
                        error: vm.mlxError,
                        downloadProgress: vm.modelDownloadProgress
                    )
                    
                    Divider()
                    
                    // Foundation Models Side
                    FoundationResponseView(
                        response: vm.foundationResponse,
                        isGenerating: vm.foundationIsGenerating,
                        availability: vm.foundationAvailability,
                        error: vm.foundationError
                    )
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                
                Divider()
                
                // Input field
                ComparisonPromptField(
                    prompt: $vm.prompt,
                    isGenerating: vm.isGenerating,
                    sendAction: vm.generateBoth
                )
                .padding()
            }
            .navigationTitle("MLX vs FMF")
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    Button("Clear") {
                        vm.clearAll()
                    }
                }
            }
            .task {
                // Preload both models when view appears
                await vm.preloadModels()
            }
        }
    }
}

/// MLX response display panel
struct MLXResponseView: View {
    let response: String
    let isGenerating: Bool
    let isPreloading: Bool
    let modelName: String
    let error: String?
    let downloadProgress: Progress?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: "cpu")
                    .foregroundColor(.orange)
                Text("MLX")
                    .font(.headline)
                    .foregroundColor(.orange)
                Spacer()
                if isGenerating || isPreloading {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            
            // Model info
            Text(modelName)
                .font(.caption)
                .foregroundColor(.secondary)
            
            // Download progress if applicable
            if let progress = downloadProgress {
                ProgressView(value: progress.fractionCompleted) {
                    Text("Downloading model...")
                        .font(.caption2)
                }
            }
            
            // Response area
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if let error = error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundColor(.red)
                            .font(.caption)
                    } else if isPreloading {
                        Text("Loading model...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else if response.isEmpty && !isGenerating {
                        Text("Ready to generate...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else {
                        Text(LocalizedStringKey(response.isEmpty ? "Generating..." : response))
                            .textSelection(.enabled)
                            .font(.caption)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

/// Foundation Models response display panel
struct FoundationResponseView: View {
    let response: String
    let isGenerating: Bool
    let availability: String
    let error: String?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.blue)
                Text("FMF")
                    .font(.headline)
                    .foregroundColor(.blue)
                Spacer()
                if isGenerating {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            
            // Availability status
            HStack {
                Circle()
                    .fill(availability == "Available" ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(availability)
            }
            .font(.caption)
            .foregroundColor(.secondary)
            
            // Response area
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if let error = error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundColor(.red)
                            .font(.caption)
                    } else if availability != "Available" {
                        Text("Foundation Models not available")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else if response.isEmpty && !isGenerating {
                        Text("Ready to generate...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else {
                        Text(LocalizedStringKey(response.isEmpty ? "Generating..." : response))
                            .textSelection(.enabled)
                            .font(.caption)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

/// Input field for comparison view
struct ComparisonPromptField: View {
    @Binding var prompt: String
    let isGenerating: Bool
    let sendAction: () async -> Void
    
    @State private var task: Task<Void, Never>?
    
    var body: some View {
        HStack {
            TextField("What is the meaning of life?", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)
            
            Button {
                if isRunning {
                    task?.cancel()
                    removeTask()
                } else {
                    task = Task {
                        await sendAction()
                        removeTask()
                    }
                }
            } label: {
                Image(systemName: isRunning ? "stop.circle.fill" : "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundColor(prompt.isEmpty ? .gray : .blue)
            }
            .disabled(prompt.isEmpty && !isRunning)
            .keyboardShortcut(isRunning ? .cancelAction : .defaultAction)
        }
    }
    
    private var isRunning: Bool {
        task != nil && !(task!.isCancelled)
    }
    
    private func removeTask() {
        task = nil
    }
}

#Preview {
    ComparisonChatView(
        viewModel: ComparisonChatViewModel(
            mlxService: MLXService(),
            foundationService: FoundationModelsService()
        )
    )
}
