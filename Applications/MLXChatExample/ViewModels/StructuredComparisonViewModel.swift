//
//  StructuredComparisonViewModel.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import Foundation
import MLXLMCommon
import FoundationModels

/// ViewModel for structured side-by-side comparison of MLX and Foundation Models
/// Manages concurrent structured generation from both services
@Observable
@MainActor
class StructuredComparisonViewModel {
    /// MLX service for running local models
    private let mlxService: MLXService
    
    /// Foundation Models service for Apple Intelligence
    private let foundationService: FoundationModelsService
    
    /// Current user input prompt
    var prompt: String = "What is the meaning of life?"
    
    // MARK: - MLX State
    /// MLX model to use (defaulting to Llama 1B)
    var selectedMLXModel: LMModel = MLXService.availableModels.first { $0.name.contains("qwen3:1.7b") } ?? MLXService.availableModels.first!
    
    /// Current MLX structured response
    var mlxStructuredResponse: PhilosophicalResponse?
    
    /// MLX partial response during streaming
    var mlxPartialResponse: PhilosophicalResponse.PartiallyGenerated?
    
    /// MLX raw JSON for debugging
    var mlxRawJSON: String = ""
    
    /// Whether MLX is currently generating
    var mlxIsGenerating = false
    
    /// MLX generation error, if any
    var mlxError: String?
    
    /// MLX generation task for cancellation
    private var mlxGenerateTask: Task<Void, any Error>?
    
    /// Whether MLX model is being preloaded
    var mlxIsPreloading = false
    
    // MARK: - Foundation Models State
    /// Current Foundation Models structured response
    var foundationStructuredResponse: PhilosophicalResponse?
    
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
    
    /// JSON Schema for display/debugging
    @available(iOS 26.0, macOS 26.0, *)
    var jsonSchema: String {
        do {
            return try foundationService.getJSONSchema()
        } catch {
            return "Failed to get schema: \(error.localizedDescription)"
        }
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
    
    /// Generate structured responses from both models simultaneously
    @available(iOS 26.0, macOS 26.0, *)
    func generateBothStructured() async {
        guard !prompt.isEmpty else { return }
        
        // Cancel any existing tasks
        cancelGeneration()
        
        // Clear previous responses and errors
        clearResponses()
        
        // Store the prompt and clear input
        let currentPrompt = prompt
        prompt = ""
        
        // Start both generations concurrently
        async let mlxTask = generateMLXStructuredStreaming(prompt: currentPrompt)
        //  async let foundationTask = generateFoundationStructured(prompt: currentPrompt)
        
        // Wait for both to complete
        await (mlxTask)
    }
    
    /// Generate structured response using MLX service with streaming partial updates
    @available(iOS 26.0, macOS 26.0, *)
    private func generateMLXStructuredStreaming(prompt: String) async {
        mlxIsGenerating = true
        mlxError = nil
        
        print("🚀 MLX Streaming: Starting structured generation for prompt: \(prompt)")
        
        mlxGenerateTask = Task {
            do {
                // Get JSON schema from Foundation Models
                let schema = try foundationService.getJSONSchema()
                print("📋 MLX Streaming: Using JSON Schema:")
                print(schema)
                
                var rawResponse = ""
                
                // Generate with schema constraint
                for await generation in try await mlxService.generateStructured(
                    prompt: prompt,
                    jsonSchema: schema,
                    model: selectedMLXModel
                ) {
                    switch generation {
                    case .chunk(let chunk):
                        rawResponse += chunk
                        mlxRawJSON = rawResponse
                        
                        // Try to parse partial response on every chunk
                        await parsePartialResponse(rawResponse)
                        
                    case .info, .toolCall:
                        break
                    }
                }
                
//                print("✅ MLX Streaming: Final raw response:")
//                print("'" + rawResponse + "'")
                
                // Final parsing for complete response
                if !rawResponse.isEmpty {
                    let cleanedJSON = extractJSONFromResponse(rawResponse)
                 //   print("🧹 MLX Streaming: Final cleaned JSON:")
                   // print("'" + cleanedJSON + "'")
                    mlxRawJSON = cleanedJSON
                    
                    if !cleanedJSON.isEmpty {
                        do {
                            print("🔄 MLX Streaming: Parsing final complete response...")
                            let structuredResponse = try PhilosophicalResponse(fromJSON: cleanedJSON)
                            print("🎉 MLX Streaming: Successfully parsed final structured response")
                          //  mlxStructuredResponse = structuredResponse
                        } catch {
                            print("❌ MLX Streaming: Failed to parse final JSON: \(error)")
                            mlxError = "Final JSON parsing failed: \(error.localizedDescription)"
                        }
                    } else {
                        print("❌ MLX Streaming: Final cleaned JSON is empty")
                        mlxError = "No valid JSON found in final response"
                    }
                } else {
                    print("❌ MLX Streaming: Final raw response is empty")
                    mlxError = "Empty response from model"
                }
                
            } catch {
                if !Task.isCancelled {
                    print("❌ MLX Streaming: Generation failed: \(error)")
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
        
     //   mlxIsGenerating = false
      //  mlxGenerateTask = nil
    }
    
    /// Parse partial response from accumulated JSON chunks
    @available(iOS 26.0, macOS 26.0, *)
    private func parsePartialResponse(_ rawResponse: String) async {
        // Try to extract and clean the JSON
        let cleanedJSON = extractJSONFromResponse(rawResponse)
        
        guard !cleanedJSON.isEmpty else {
            print("🔍 MLX Streaming: No JSON found in chunk, skipping parse attempt")
            return
        }
        
        do {
            print("🔄 MLX Streaming: Attempting to parse partial JSON:")
            print("'" + cleanedJSON + "'")
            
            // Create GeneratedContent from partial JSON
            let generatedContent = try GeneratedContent(json: cleanedJSON)
            
            // Convert to PartiallyGenerated
            let partialResponse = try PhilosophicalResponse.PartiallyGenerated(generatedContent)
            
            print("✨ MLX Streaming: Successfully parsed partial response")
            mlxPartialResponse = partialResponse
            
        } catch {
            // It's normal for partial JSON to fail parsing, so we don't treat this as an error
            print("🔍 MLX Streaming: Partial JSON parsing failed (normal): \(error.localizedDescription)")
            // Continue without updating mlxPartialResponse
        }
    }
    
    /// Generate structured response using Foundation Models service
    @available(iOS 26.0, macOS 26.0, *)
    private func generateFoundationStructured(prompt: String) async {
        foundationIsGenerating = true
        foundationError = nil
        
        foundationGenerateTask = Task {
            do {
                for await generation in try await foundationService.generateStructured(prompt: prompt) {
                    switch generation {
                    case .structuredResponse(let response):
                        foundationStructuredResponse = response
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
    
    /// Extract JSON from MLX response that might contain extra text
    private func extractJSONFromResponse(_ response: String) -> String {
        print("🔍 Extracting JSON from response: '\(response)'")
        
        // Remove any whitespace/newlines at start and end
        var trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        print("🔍 Trimmed: '\(trimmed)'")
        
        // Process thinking content similar to the ConversationView approach
        let cleanedContent = processThinkingContent(trimmed)
        trimmed = cleanedContent
        
        // Clean up any remaining whitespace after removing think tags
        trimmed = trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
        print("🔍 After removing think tags: '\(trimmed)'")
        
        // If it already looks like clean JSON, return it
        if trimmed.hasPrefix("{") && trimmed.hasSuffix("}") {
            print("🔍 Response already looks like clean JSON")
            return trimmed
        }
        
        // Look for JSON object boundaries
        if let startIndex = trimmed.firstIndex(of: "{"),
           let endIndex = trimmed.lastIndex(of: "}") {
            let extracted = String(trimmed[startIndex...endIndex])
            print("🔍 Extracted JSON: '\(extracted)'")
            return extracted
        }
        
        // Try to find JSON in the middle of text
        let lines = trimmed.components(separatedBy: .newlines)
        for line in lines {
            let cleanLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if cleanLine.hasPrefix("{") && cleanLine.hasSuffix("}") {
                print("🔍 Found JSON in line: '\(cleanLine)'")
                return cleanLine
            }
        }
        
        print("🔍 No valid JSON found, returning original trimmed response")
        return trimmed
    }
    
    /// Process thinking content to extract only the non-thinking parts
    /// Adapted from ConversationView's processThinkingContent method
    private func processThinkingContent(_ content: String) -> String {
        // Remove <think> tags and their content
        let pattern = "<think>.*?(</think>|$)"
        do {
            let regex = try NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
            let range = NSRange(location: 0, length: content.utf16.count)
            let cleanedContent = regex.stringByReplacingMatches(in: content, options: [], range: range, withTemplate: "")
            return cleanedContent.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            print("❌ Error processing thinking content: \(error)")
            // Fallback to original content if regex fails
            return content
        }
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
        mlxStructuredResponse = nil
        mlxPartialResponse = nil
        foundationStructuredResponse = nil
        mlxRawJSON = ""
        mlxError = nil
        foundationError = nil
    }
    
    /// Clear all responses
    func clearAll() {
        cancelGeneration()
        clearResponses()
        foundationService.clearSession()
    }
}
