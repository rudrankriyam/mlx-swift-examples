// Copyright © 2024 Apple Inc.
//
// Real-time streaming parallel LLM text display view

import SwiftUI

struct ParallelStreamingComparisonView: View {
    @ObservedObject var viewModel: ParallelStreamingViewModel

    init(viewModel: ParallelStreamingViewModel = ParallelStreamingViewModel()) {
        self.viewModel = viewModel
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header with controls
            HStack {
                Text("Live Parallel Text Streaming")
                    .font(.title2)
                    .fontWeight(.bold)

                Spacer()

                if #available(macOS 26.0, *) {
                    Button(action: {
                        if viewModel.isRunning {
                            viewModel.stopAllSessions()
                        } else {
                            Task {
                                await viewModel.startParallelEvaluation()
                            }
                        }
                    }) {
                        HStack(spacing: 8) {
                            Image(systemName: viewModel.isRunning ? "stop.circle.fill" : "play.circle.fill")
                            Text(viewModel.isRunning ? "Stop" : "Evaluate")
                        }
                        .font(.headline)
                       // .background(viewModel.isRunning ? Color.red.opacity(0.8) : Color.blue.opacity(0.8))
                        .foregroundColor(.white)
                    }
                    .buttonStyle(.glassProminent)
                    .buttonBorderShape(.roundedRectangle(radius: 16))
                } else {
                    // Fallback on earlier versions
                }

                if !viewModel.isRunning && viewModel.hasResults {
                    Button(action: {
                        viewModel.reset()
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.clockwise.circle.fill")
                            Text("Reset")
                        }
                        .font(.headline)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.gray.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
            }
            .padding()

            // Side-by-side streaming text display
            HStack(spacing: 0) {
                ForEach(0..<3, id: \.self) { sessionId in
                    StreamingTextView(
                        sessionId: sessionId,
                        text: viewModel.streamingTexts[sessionId] ?? "",
                        isActive: viewModel.activeSessions.contains(sessionId),
                        isComplete: viewModel.sessionResults[sessionId] != nil
                    )
                }
            }
        }
        .navigationTitle("Live Streaming Comparison")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

struct StreamingTextView: View {
    let sessionId: Int
    let text: String
    let isActive: Bool
    let isComplete: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Session header
            HStack {
                Text("Session \(sessionId)")
                    .font(.headline)
                    .foregroundColor(isActive ? .blue : isComplete ? .green : .gray)

                Spacer()

                if isActive {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color.blue)
                            .frame(width: 8, height: 8)
                        Text("Streaming")
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                } else if isComplete {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("Complete")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                } else {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color.gray)
                            .frame(width: 8, height: 8)
                        Text("Waiting")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color.secondary.opacity(0.1))

            // Streaming text area
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if text.isEmpty && !isActive && !isComplete {
                        VStack(spacing: 8) {
                            Image(systemName: "text.bubble")
                                .font(.largeTitle)
                                .foregroundColor(.gray.opacity(0.5))
                            Text("Waiting to start...")
                                .foregroundColor(.gray)
                                .font(.subheadline)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(.top, 60)
                    } else {
                        Text(text)
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(16)
                            .lineSpacing(4)

                        if isActive {
                            HStack(spacing: 4) {
                                Text("Generating")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                                ForEach(0..<3) { index in
                                    Circle()
                                        .fill(Color.blue)
                                        .frame(width: 4, height: 4)
                                }
                            }
                            .padding(.horizontal, 16)
                            .padding(.bottom, 16)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .border(Color.gray.opacity(0.2), width: 1)
    }
}


// MARK: - View Model

@MainActor
class ParallelStreamingViewModel: ObservableObject {
    @Published var sessionResults: [ParallelEvaluationExample.EvaluationResult?] = Array(repeating: nil, count: 3)
    @Published var streamingTexts: [Int: String] = [:]
    @Published var activeSessions: Set<Int> = []
    @Published var isRunning = false

    private var example = ParallelEvaluationExample()
    private var animationTimer: Timer?

    var hasResults: Bool {
        sessionResults.contains(where: { $0 != nil })
    }

    // Animation for the "Generating..." dots
    private var dotAnimationPhase = 0
    func dotOpacity(for index: Int) -> Double {
        let phase = (dotAnimationPhase + index) % 4
        return phase == 0 ? 0.3 : 1.0
    }

    private func startAnimation() {
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.dotAnimationPhase = (self?.dotAnimationPhase ?? 0 + 1) % 4
                self?.objectWillChange.send()
            }
        }
    }

    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }

    func startParallelEvaluation() async {
        guard !isRunning else { return }

        reset()
        isRunning = true
        startAnimation()

        let config = ParallelEvaluationExample.ParallelConfig(
            numConcurrentSessions: 3,
            prompts: [
                "Write a short poem about artificial intelligence.",
                "Explain neural networks in simple terms.",
                "What are the benefits of renewable energy?"
            ],
            maxTokens: 2000
        )

        do {
            // Use the real streaming implementation
            let (_, _) = try await example.runParallelEvaluationWithStreaming(
                config: config,
                onTokenGenerated: { [weak self] sessionId, token in
                    Task { @MainActor in
                        self?.handleTokenGenerated(sessionId: sessionId, token: token)
                    }
                },
                onSessionComplete: { [weak self] sessionId, result in
                    Task { @MainActor in
                        self?.handleSessionComplete(sessionId: sessionId, result: result)
                    }
                }
            )

        } catch {
            print("❌ Error in parallel evaluation: \(error)")
            await MainActor.run {
                self.isRunning = false
                self.stopAnimation()
                self.objectWillChange.send()
            }
        }
    }

    func stopAllSessions() {
        isRunning = false
        activeSessions.removeAll()
        stopAnimation()
        // Note: In a real implementation, you'd need to cancel the actual tasks
    }

    func reset() {
        sessionResults = Array(repeating: nil, count: 3)
        streamingTexts.removeAll()
        activeSessions.removeAll()
        isRunning = false
        stopAnimation()
    }

    // Handle real-time token generation
    private func handleTokenGenerated(sessionId: Int, token: String) {
        // Append the new token to the streaming text
        let currentText = streamingTexts[sessionId] ?? ""
        streamingTexts[sessionId] = currentText + token

        // Mark session as active if not already
        activeSessions.insert(sessionId)

        // Update UI
        objectWillChange.send()
    }

    // Handle session completion
    private func handleSessionComplete(sessionId: Int, result: ParallelEvaluationExample.EvaluationResult) {
        sessionResults[sessionId] = result
        activeSessions.remove(sessionId)

        // Make sure the final text is set
        streamingTexts[sessionId] = result.generatedText

        // Check if all sessions are complete
        if activeSessions.isEmpty && isRunning {
            isRunning = false
            stopAnimation()
        }

        // Update UI
        objectWillChange.send()
    }

}

#Preview {
    NavigationView {
        ParallelStreamingComparisonView()
    }
}
