//
//  FoundationModelsService.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import Foundation
import FoundationModels

/// Service for Apple's Foundation Models framework integration
/// Provides access to on-device Apple Intelligence models
@Observable
class FoundationModelsService {
    /// The system language model instance
    private let model = SystemLanguageModel.default
    
    /// Current language model session for conversation
    private var session: LanguageModelSession?
    
    /// Current availability status of the model
    var availability: SystemLanguageModel.Availability {
        model.availability
    }
    
    /// Whether the model is currently available for use
    var isAvailable: Bool {
        switch availability {
        case .available:
            return true
        case .unavailable:
            return false
        }
    }
    
    /// Human-readable status description
    var statusDescription: String {
        switch availability {
        case .available:
            return "Available"
        case .unavailable(.deviceNotEligible):
            return "Device Not Eligible"
        case .unavailable(.appleIntelligenceNotEnabled):
            return "Apple Intelligence Disabled"
        case .unavailable(.modelNotReady):
            return "Model Downloading..."
        case .unavailable(let other):
            return "Unavailable"
        }
    }
    
    /// Initialize or reset the session with system instructions
    private func initializeSession() {
        let instructions = Instructions("You are a helpful assistant!")
        
        session = LanguageModelSession(instructions: instructions)
    }
    
    /// Prewarm the Foundation Models session for faster response
    func prewarmSession() {
        guard isAvailable else { return }
        
        // Initialize session if needed
        if session == nil {
            initializeSession()
        }
        
        // Prewarm the session to load resources into memory
        session?.prewarm()
    }
    
    /// Generate structured response for a given prompt
    /// - Parameter prompt: The user's input prompt
    /// - Returns: AsyncStream of structured response updates
    @available(iOS 26.0, macOS 26.0, *)
    func generateStructured(prompt: String) async throws -> AsyncStream<StructuredFoundationGeneration> {
        // Ensure model is available
        guard isAvailable else {
            throw FoundationModelsError.modelUnavailable
        }
        
        // Initialize session if needed
        if session == nil {
            initializeSession()
        }
        
        guard let session = session else {
            throw FoundationModelsError.sessionInitializationFailed
        }
        
        // Check if session is busy
        if session.isResponding {
            throw FoundationModelsError.sessionBusy
        }
        
        return AsyncStream { continuation in
            Task {
                do {
                    let promptObject = Prompt(prompt)
                    
                    // Use structured generation with PhilosophicalResponse
                    let response = try await session.respond(
                        to: promptObject,
                        generating: PhilosophicalResponse.self
                    )
                    
                    // Send the structured response
                    continuation.yield(.structuredResponse(response.content))
                    
                } catch {
                    continuation.yield(.error(error))
                }
                continuation.finish()
            }
        }
    }
    
    /// Get the JSON schema for MLX integration
    @available(iOS 26.0, macOS 26.0, *)
    func getJSONSchema() throws -> String {
        let schema = PhilosophicalResponse.generationSchema
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let schemaData = try encoder.encode(schema)
        let schemaString = String(data: schemaData, encoding: .utf8) ?? "{}"
        return schemaString
    }
    
    /// Generate streaming response for a given prompt (legacy text mode)
    /// - Parameter prompt: The user's input prompt
    /// - Returns: AsyncStream of response chunks
    @available(iOS 26.0, macOS 26.0, *)
    func generate(prompt: String) async throws -> AsyncStream<FoundationGeneration> {
        // Ensure model is available
        guard isAvailable else {
            throw FoundationModelsError.modelUnavailable
        }
        
        // Initialize session if needed
        if session == nil {
            initializeSession()
        }
        
        guard let session = session else {
            throw FoundationModelsError.sessionInitializationFailed
        }
        
        // Check if session is busy
        if session.isResponding {
            throw FoundationModelsError.sessionBusy
        }
        
        return AsyncStream { continuation in
            Task {
                do {
                    let promptObject = Prompt(prompt)
                    var lastContent = ""
                    
                    // Use streaming response
                    for try await snapshot in session.streamResponse(to: promptObject) {
                        // Extract string content from GeneratedContent
                        let currentContent: String
                        switch snapshot.rawContent.kind {
                        case .string(let stringValue):
                            currentContent = stringValue
                        case .null:
                            currentContent = ""
                        default:
                            // For non-string content, convert to string representation
                            currentContent = "\(snapshot.rawContent)"
                        }
                        
                        // Send incremental chunks
                        if currentContent != lastContent {
                            if lastContent.isEmpty {
                                // First chunk - send entire content
                                continuation.yield(.chunk(currentContent))
                            } else {
                                // Send only the new part
                                if currentContent.hasPrefix(lastContent) {
                                    let newChunk = String(currentContent.dropFirst(lastContent.count))
                                    if !newChunk.isEmpty {
                                        continuation.yield(.chunk(newChunk))
                                    }
                                } else {
                                    // Content changed completely, send new content
                                    continuation.yield(.chunk(currentContent))
                                }
                            }
                            lastContent = currentContent
                        }
                    }
                    
                } catch {
                    continuation.yield(.error(error))
                }
                continuation.finish()
            }
        }
    }
    
    /// Clear the current session to start fresh
    func clearSession() {
        session = nil
    }
}

/// Represents different types of structured generation output from Foundation Models
@available(iOS 26.0, macOS 26.0, *)
enum StructuredFoundationGeneration {
    case structuredResponse(PhilosophicalResponse)
    case error(Error)
}

/// Represents different types of generation output from Foundation Models
enum FoundationGeneration {
    case chunk(String)
    case error(Error)
}

/// Errors specific to Foundation Models service
enum FoundationModelsError: LocalizedError {
    case modelUnavailable
    case sessionInitializationFailed
    case sessionBusy
    
    var errorDescription: String? {
        switch self {
        case .modelUnavailable:
            return "Foundation model is not available"
        case .sessionInitializationFailed:
            return "Failed to initialize language model session"
        case .sessionBusy:
            return "Session is currently processing another request"
        }
    }
}
