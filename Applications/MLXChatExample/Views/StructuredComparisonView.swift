//
//  StructuredComparisonView.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import SwiftUI

/// Main structured comparison view showing MLX and Foundation Models side by side
@available(iOS 26.0, macOS 26.0, *)
struct StructuredComparisonView: View {
    @Bindable private var vm: StructuredComparisonViewModel
    
    init(viewModel: StructuredComparisonViewModel) {
        self.vm = viewModel
    }
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Side-by-side structured comparison
                HStack(spacing: 0) {
                    // MLX Side
                    StructuredMLXView(
                        response: vm.mlxStructuredResponse,
                        partialResponse: vm.mlxPartialResponse,
                        rawJSON: vm.mlxRawJSON,
                        isGenerating: vm.mlxIsGenerating,
                        isPreloading: vm.mlxIsPreloading,
                        modelName: vm.selectedMLXModel.name,
                        error: vm.mlxError,
                        downloadProgress: vm.modelDownloadProgress
                    )
                    
                    Divider()
                    
                    // Foundation Models Side
                    StructuredFoundationView(
                        response: vm.foundationStructuredResponse,
                        isGenerating: vm.foundationIsGenerating,
                        availability: vm.foundationAvailability,
                        error: vm.foundationError
                    )
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                
                Divider()
                
                // Input field
                StructuredPromptField(
                    prompt: $vm.prompt,
                    isGenerating: vm.isGenerating,
                    sendAction: vm.generateBothStructured
                )
                .padding()
            }
            .navigationTitle("MLX vs FMF - Structured")
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    Button("Clear") {
                        vm.clearAll()
                    }
                }
                
                ToolbarItem(placement: .navigation) {
                    Button("Schema") {
                        showJSONSchema()
                    }
                }
            }
            .task {
                // Preload both models when view appears
                await vm.preloadModels()
            }
        }
    }
    
    private func showJSONSchema() {
        // TODO: Show JSON schema in a sheet or alert
        print("JSON Schema:", vm.jsonSchema)
    }
}

/// MLX structured response display panel
@available(iOS 26.0, macOS 26.0, *)
struct StructuredMLXView: View {
    let response: PhilosophicalResponse?
    let partialResponse: PhilosophicalResponse.PartiallyGenerated?
    let rawJSON: String
    let isGenerating: Bool
    let isPreloading: Bool
    let modelName: String
    let error: String?
    let downloadProgress: Progress?
    
    @State private var showRawJSON = false
    
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
                
                Button(action: { showRawJSON.toggle() }) {
                    Image(systemName: "curlybraces")
                        .foregroundColor(.secondary)
                }
                
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
                VStack(alignment: .leading, spacing: 12) {
                    if let error = error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundColor(.red)
                            .font(.caption)
                    } else if isPreloading {
                        Text("Loading model...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else if isGenerating {
                        VStack(alignment: .leading, spacing: 12) {
                            // Show partial response if available during streaming
                            if let partialResponse = partialResponse {
                                PartialPhilosophicalResponseView(
                                    partialResponse: partialResponse,
                                    title: "Streaming Response",
                                    isStreaming: true
                                )
                            } else {
                                Text("Generating structured response...")
                                    .foregroundColor(.secondary)
                                    .font(.subheadline)
                            }
                            
                            if !rawJSON.isEmpty && showRawJSON {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Raw JSON:")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text(rawJSON)
                                        .font(.system(size: 10, design: .monospaced))
                                        .padding(8)
                                        .background(Color.gray.opacity(0.1))
                                        .cornerRadius(4)
                                }
                            }
                        }
                    } else if let response = response {
                        PartialPhilosophicalResponseView(
                            partialResponse: response.asPartiallyGenerated(),
                            title: "Complete Response",
                            isStreaming: false
                        )
                    } else if showRawJSON && !rawJSON.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Raw JSON:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(rawJSON)
                                .font(.system(size: 10, design: .monospaced))
                                .padding(8)
                                .background(Color.gray.opacity(0.1))
                                .cornerRadius(4)
                                .textSelection(.enabled)
                        }
                    } else {
                        Text("Ready to generate...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
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

/// Foundation Models structured response display panel
@available(iOS 26.0, macOS 26.0, *)
struct StructuredFoundationView: View {
    let response: PhilosophicalResponse?
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
                VStack(alignment: .leading, spacing: 12) {
                    if let error = error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundColor(.red)
                            .font(.caption)
                    } else if availability != "Available" {
                        Text("Foundation Models not available")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else if isGenerating {
                        Text("Generating structured response...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else if let response = response {
                        PartialPhilosophicalResponseView(
                            partialResponse: response.asPartiallyGenerated(),
                            title: "Foundation Models",
                            isStreaming: false
                        )
                    } else {
                        Text("Ready to generate...")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
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

/// Beautiful card displaying structured philosophical response
@available(iOS 26.0, macOS 26.0, *)
struct StructuredResponseCard: View {
    let response: PhilosophicalResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Main Answer
            if let answer = response.answer {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "quote.bubble")
                            .foregroundColor(.primary)
                        Text("Answer")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                    }
                    Text(answer)
                        .padding(.leading, 4)
                }
            }
            
            // Confidence & Philosophy Row
            HStack {
                if let confidence = response.confidence {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "gauge.medium")
                                .foregroundColor(.blue)
                            Text("Confidence")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        HStack {
                            ForEach(1...10, id: \.self) { index in
                                Circle()
                                    .fill(index <= confidence ? Color.blue : Color.gray.opacity(0.3))
                                    .frame(width: 6, height: 6)
                            }
                            Text("\(confidence)/10")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                
                Spacer()
                
                if let philosophy = response.philosophy {
                    VStack(alignment: .trailing, spacing: 4) {
                        HStack {
                            Image(systemName: "brain")
                                .foregroundColor(.purple)
                            Text("Philosophy")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Text(philosophy)
                            .font(.caption)
                            .fontWeight(.medium)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }
            
            // Follow-up Question
            if let followUpQuestion = response.followUpQuestion {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "questionmark.circle")
                            .foregroundColor(.green)
                        Text("Follow-up")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(followUpQuestion)
                        .font(.caption)
                        .italic()
                        .padding(.leading, 4)
                }
            }
            
            // Practical Application
            if let practicalApplication = response.practicalApplication {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "lightbulb")
                            .foregroundColor(.orange)
                        Text("Practical Application")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(practicalApplication)
                        .font(.caption)
                        .padding(.leading, 4)
                }
            }
        }
        .padding(12)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
        .textSelection(.enabled)
    }
}

/// Input field for structured comparison view
@available(iOS 26.0, macOS 26.0, *)
struct StructuredPromptField: View {
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
    if #available(iOS 26.0, macOS 26.0, *) {
        StructuredComparisonView(
            viewModel: StructuredComparisonViewModel(
                mlxService: MLXService(),
                foundationService: FoundationModelsService()
            )
        )
    } else {
        Text("Requires iOS 26.0+")
    }
}