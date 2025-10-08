//
//  PartialPhilosophicalResponseView.swift
//  MLXChatExample
//
//  Created by AI Assistant on 21.04.2025.
//

import SwiftUI
import FoundationModels

/// View for displaying streaming/partial philosophical responses
@available(iOS 26.0, macOS 26.0, *)
struct PartialPhilosophicalResponseView: View {
    let partialResponse: PhilosophicalResponse.PartiallyGenerated
    let title: String
    let isStreaming: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.primary)
                
                if isStreaming {
                    ProgressView()
                        .scaleEffect(0.8)
                        .padding(.leading, 8)
                }
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 12) {
                // Answer field
                PartialFieldView(
                    label: "Answer",
                    value: partialResponse.answer,
                    icon: "brain.head.profile"
                )
                
                // Confidence field
                if let confidence = partialResponse.confidence {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "gauge.with.dots.needle.33percent")
                            .foregroundStyle(.secondary)
                            .frame(width: 20, alignment: .leading)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Confidence")
                                .font(.caption)
                                .foregroundStyle(.secondary)

                            HStack {
                                ForEach(1...10, id: \.self) { index in
                                    Circle()
                                        .fill(index <= confidence ? Color.blue : Color.gray.opacity(0.3))
                                        .frame(width: 6, height: 6)
                                }
                                Text("\(confidence)/10")
                                    .font(.body)
                                    .foregroundStyle(.primary)
                            }
                        }

                        Spacer()
                    }
                    .animation(.easeInOut(duration: 0.3), value: confidence)
                } else {
                    PartialFieldPlaceholder(
                        label: "Confidence",
                        icon: "gauge.with.dots.needle.33percent"
                    )
                }
                
                // Philosophy field
                PartialFieldView(
                    label: "Related Philosophy",
                    value: partialResponse.philosophy,
                    icon: "book.closed"
                )
                
                // Follow-up question field
                PartialFieldView(
                    label: "Follow-up Question",
                    value: partialResponse.followUpQuestion,
                    icon: "questionmark.circle"
                )
                
                // Practical application field
                PartialFieldView(
                    label: "Practical Application",
                    value: partialResponse.practicalApplication,
                    icon: "lightbulb"
                )
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

/// View for displaying a field that has content
@available(iOS 26.0, macOS 26.0, *)
struct PartialFieldView: View {
    let label: String
    let value: String?
    let icon: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
                .frame(width: 20, alignment: .leading)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                if let value, !value.isEmpty {
                    Text(value)
                        .font(.body)
                        .foregroundStyle(.primary)
                        .transition(.opacity.combined(with: .move(edge: .leading)))
                } else {
                    Text("Generating...")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .italic()
                }
            }
            
            Spacer()
        }
        .animation(.easeInOut(duration: 0.3), value: value)
    }
}

/// Placeholder for fields that haven't been generated yet
@available(iOS 26.0, macOS 26.0, *)
struct PartialFieldPlaceholder: View {
    let label: String
    let icon: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(.quaternary)
                .frame(width: 20, alignment: .leading)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                
                Text("Waiting...")
                    .font(.body)
                    .foregroundStyle(.quaternary)
                    .italic()
            }
            
            Spacer()
        }
    }
}

#Preview {
    if #available(iOS 26.0, macOS 26.0, *) {
        // Create a mock complete response and use it as partial
        let mockResponse = PhilosophicalResponse(
            answer: "Life's meaning emerges through our connections and contributions to others.",
            confidence: 8,
            philosophy: "Existentialism",
            followUpQuestion: "How do you find meaning in your daily interactions?",
            practicalApplication: "Practice active listening and help others when possible."
        )
        
        // Convert to PartiallyGenerated (which is just the same type)
        let partialResponse = mockResponse.asPartiallyGenerated()
        
        PartialPhilosophicalResponseView(
            partialResponse: partialResponse,
            title: "MLX Streaming",
            isStreaming: true
        )
        .padding()
    } else {
        Text("Requires iOS 26.0+")
    }
}