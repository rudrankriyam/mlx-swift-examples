//
//  ComparisonChatViewModel.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import Foundation
import MLXLMCommon

/// ViewModel for side-by-side comparison of MLX and Foundation Models
/// Manages concurrent generation from both services
@Observable
@MainActor
class ComparisonChatViewModel {
    /// MLX service for running local models
    private let mlxService: MLXService
    
    /// Foundation Models service for Apple Intelligence
    private let foundationService: FoundationModelsService
    
    /// Current user input prompt
    var prompt: String = "What is the meaning of life?"
    
    /// Chat history for context
    var messages: [Message] = [
        .system("You are a helpful assistant!")
    ]
    
    // MARK: - MLX State
    /// MLX model to use (defaulting to Llama 1B)
    var selectedMLXModel: LMModel = MLXService.availableModels.first { $0.name.contains("llama3.2:1b") } ?? MLXService.availableModels.first!
    
    /// Current MLX response being generated
    var mlxResponse: String = ""
    
    /// Whether MLX is currently generating
    var mlxIsGenerating = false
    
    /// MLX generation error, if any
    var mlxError: String?
    
    /// MLX generation task for cancellation
    private var mlxGenerateTask: Task<Void, any Error>?
    
    /// Whether MLX model is being preloaded
    var mlxIsPreloading = false
    
    // MARK: - Foundation Models State
    /// Current Foundation Models response being generated
    var foundationResponse: String = ""
    
    /// Whether Foundation Models is currently generating
    var foundationIsGenerating = false
    
    /// Foundation Models error, if any
    var foundationError: String?
    
    /// Foundation Models generation task for cancellation
    private var foundationGenerateTask: Task<Void, any Error>?
    
    // MARK: - Computed Properties
    /// Whether any generation is in progress
    var isGenerating: Bool {
        mlxIsGenerating || foundationIsGenerating
    }
    
    /// MLX model download progress
    var modelDownloadProgress: Progress? {
        mlxService.modelDownloadProgress
    }
    
    /// Foundation Models availability status
    var foundationAvailability: String {
        foundationService.statusDescription
    }
    
    init(mlxService: MLXService, foundationService: FoundationModelsService) {
        self.mlxService = mlxService
        self.foundationService = foundationService
    }
    
    /// Preload both MLX model and Foundation Models session when the view appears
    func preloadModels() async {
        mlxIsPreloading = true
        mlxError = nil
        
        // Prewarm Foundation Models session (non-async)
        foundationService.prewarmSession()
        
        do {
            // Load MLX model into cache
            _ = try await mlxService.load(model: selectedMLXModel)
        } catch {
            mlxError = "Failed to preload MLX model: \(error.localizedDescription)"
        }
        
        mlxIsPreloading = false
    }
    
    /// Generate responses from both models simultaneously
    @available(iOS 26.0, macOS 26.0, *)
    func generateBoth() async {
        guard !prompt.isEmpty else { return }
        
        // Cancel any existing tasks
        cancelGeneration()
        
        // Add user message to history
        messages.append(.user(prompt))
        
        // Clear previous responses and errors
        clearResponses()
        
        // Store the prompt and clear input
        let currentPrompt = prompt
        prompt = ""
        
        // Start both generations concurrently
        async let mlxTask = generateMLX(prompt: currentPrompt)
        async let foundationTask = generateFoundation(prompt: currentPrompt)
        
        // Wait for both to complete
        await (mlxTask, foundationTask)
        
        // Add responses to message history once both are done
        if !mlxResponse.isEmpty {
            messages.append(.assistant("**MLX (Llama 1B):** \(mlxResponse)"))
        }
        if !foundationResponse.isEmpty {
            messages.append(.assistant("**Foundation Models:** \(foundationResponse)"))
        }
    }
    
    /// Generate response using MLX service
    private func generateMLX(prompt: String) async {
        mlxIsGenerating = true
        mlxError = nil
        
        mlxGenerateTask = Task {
            do {
                // Create temporary messages for this generation
                let tempMessages = messages + [.user(prompt), .assistant("")]
                
                for await generation in try await mlxService.generate(
                    messages: tempMessages, model: selectedMLXModel
                ) {
                    switch generation {
                    case .chunk(let chunk):
                        mlxResponse += chunk
                    case .info:
                        break // Ignore info/metrics
                    case .toolCall:
                        break
                    }
                }
            } catch {
                if !Task.isCancelled {
                    mlxError = error.localizedDescription
                }
            }
        }
        
        do {
            try await mlxGenerateTask?.value
        } catch {
            if !Task.isCancelled {
                mlxError = error.localizedDescription
            }
        }
        
        mlxIsGenerating = false
        mlxGenerateTask = nil
    }
    
    /// Generate response using Foundation Models service
    @available(iOS 26.0, macOS 26.0, *)
    private func generateFoundation(prompt: String) async {
        foundationIsGenerating = true
        foundationError = nil
        
        foundationGenerateTask = Task {
            do {
                for await generation in try await foundationService.generate(prompt: prompt) {
                    switch generation {
                    case .chunk(let chunk):
                        foundationResponse += chunk
                    case .error(let error):
                        if !Task.isCancelled {
                            foundationError = error.localizedDescription
                        }
                    }
                }
            } catch {
                if !Task.isCancelled {
                    foundationError = error.localizedDescription
                }
            }
        }
        
        do {
            try await foundationGenerateTask?.value
        } catch {
            if !Task.isCancelled {
                foundationError = error.localizedDescription
            }
        }
        
        foundationIsGenerating = false
        foundationGenerateTask = nil
    }
    
    /// Cancel all ongoing generation tasks
    func cancelGeneration() {
        mlxGenerateTask?.cancel()
        foundationGenerateTask?.cancel()
        mlxGenerateTask = nil
        foundationGenerateTask = nil
        
        mlxIsGenerating = false
        foundationIsGenerating = false
    }
    
    /// Clear responses and errors
    private func clearResponses() {
        mlxResponse = ""
        foundationResponse = ""
        mlxError = nil
        foundationError = nil
    }
    
    /// Clear all chat history and responses
    func clearAll() {
        cancelGeneration()
        clearResponses()
        messages = [.system("You are a helpful assistant!")]
        foundationService.clearSession()
    }
}