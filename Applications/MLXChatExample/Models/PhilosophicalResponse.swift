//
//  PhilosophicalResponse.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import Foundation
import FoundationModels

/// A structured response for philosophical questions using Foundation Models guided generation
@available(iOS 26.0, macOS 26.0, *)
@Generable(description: "A thoughtful philosophical response about life's big questions")
struct PhilosophicalResponse {
    /// The main philosophical answer
    @Guide(description: "A clear, thoughtful answer in 2-3 sentences")
    var answer: String?
    
    /// Confidence level in the response
    @Guide(description: "Confidence level from 1-10, where 10 is most confident", .range(1...10))
    var confidence: Int?
    
    /// Related philosophical school or thinker
    @Guide(description: "Name of a relevant philosophical school or famous philosopher")
    var philosophy: String?
    
    /// A thought-provoking follow-up question
    @Guide(description: "An engaging follow-up question to continue the philosophical discussion")
    var followUpQuestion: String?
    
    /// A brief practical application
    @Guide(description: "How this philosophy might apply to daily life in one sentence")
    var practicalApplication: String?
}

/// Extension to handle creation from generated content
@available(iOS 26.0, macOS 26.0, *)
extension PhilosophicalResponse {
    /// Create from Foundation Models GeneratedContent
    init(from generatedContent: GeneratedContent) throws {
        self = try PhilosophicalResponse(generatedContent)
    }
    
    /// Create from JSON string (for MLX responses)
    init(fromJSON json: String) throws {
        let generatedContent = try GeneratedContent(json: json)
        self = try PhilosophicalResponse(generatedContent)
    }
}