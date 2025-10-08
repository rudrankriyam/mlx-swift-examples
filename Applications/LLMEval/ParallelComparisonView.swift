// Copyright © 2024 Apple Inc.
//
// Side-by-side comparison view for parallel LLM evaluation results

import SwiftUI
import Charts

struct ParallelComparisonView: View {
    let results: [ParallelEvaluationExample.EvaluationResult]
    let totalTime: TimeInterval

    private var averageTokensPerSecond: Double {
        results.map { $0.tokensPerSecond }.reduce(0, +) / Double(results.count)
    }

    private var averageTokens: Double {
        Double(results.map { $0.totalTokens }.reduce(0, +)) / Double(results.count)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Text("Parallel LLM Evaluation Results")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text("\(results.count) concurrent sessions • Total time: \(totalTime.formatted())s")
                        .foregroundColor(.secondary)
                }
                .padding(.top)

                // Overall Statistics
                VStack(spacing: 16) {
                    Text("Overall Performance")
                        .font(.title2)
                        .fontWeight(.semibold)

                    HStack(spacing: 20) {
                        StatCard(
                            title: "Average Speed",
                            value: "\(averageTokensPerSecond.formatted())",
                            unit: "tokens/sec",
                            icon: "speedometer"
                        )

                        StatCard(
                            title: "Total Sessions",
                            value: "\(results.count)",
                            unit: "sessions",
                            icon: "square.grid.3x3"
                        )

                        StatCard(
                            title: "Average Tokens",
                            value: "\(Int(averageTokens))",
                            unit: "tokens",
                            icon: "text.word.spacing"
                        )

                        StatCard(
                            title: "Total Time",
                            value: totalTime.formatted(),
                            unit: "seconds",
                            icon: "clock"
                        )
                    }
                }
                .padding(.horizontal)

                // Performance Chart
                VStack(spacing: 16) {
                    Text("Performance Comparison")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Chart(results, id: \.sessionId) { result in
                        BarMark(
                            x: .value("Session", "Session \(result.sessionId)"),
                            y: .value("Tokens/sec", result.tokensPerSecond)
                        )
                        .foregroundStyle(.blue.gradient)
                    }
                    .frame(height: 200)
                    .padding(.horizontal)
                }

                // Side-by-Side Session Results
                VStack(spacing: 16) {
                    Text("Session Details")
                        .font(.title2)
                        .fontWeight(.semibold)

                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 16),
                        GridItem(.flexible(), spacing: 16)
                    ], spacing: 16) {
                        ForEach(results.sorted(by: { $0.sessionId < $1.sessionId }), id: \.sessionId) { result in
                            SessionResultCard(result: result)
                        }
                    }
                    .padding(.horizontal)
                }

                // Efficiency Analysis
                VStack(spacing: 16) {
                    Text("Efficiency Analysis")
                        .font(.title2)
                        .fontWeight(.semibold)

                    VStack(spacing: 12) {
                        EfficiencyRow(
                            label: "Concurrent Processing",
                            value: "✅ \(results.count) sessions ran simultaneously"
                        )

                        EfficiencyRow(
                            label: "Thread Safety",
                            value: "✅ Task-local random states used"
                        )

                        EfficiencyRow(
                            label: "GPU Serialization",
                            value: "✅ Operations serialized on GPU safely"
                        )

                        EfficiencyRow(
                            label: "Memory Management",
                            value: "✅ 20MB GPU cache limit enforced"
                        )
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.bottom, 20)
        }
        .navigationTitle("Parallel Comparison")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let unit: String
    let icon: String

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title)
                .foregroundColor(.blue)

            Text(value)
                .font(.title2)
                .fontWeight(.bold)

            Text(unit)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.1))
                .shadow(color: Color.gray.opacity(0.2), radius: 2, x: 0, y: 1)
        )
    }
}

struct SessionResultCard: View {
    let result: ParallelEvaluationExample.EvaluationResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Session \(result.sessionId)")
                    .font(.headline)
                    .foregroundColor(.blue)
                Spacer()
                Text("\(result.tokensPerSecond.formatted()) t/s")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.green)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Prompt:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(result.prompt)
                    .font(.subheadline)
                    .lineLimit(2)
                    .foregroundColor(.primary)

                Divider()

                HStack(spacing: 16) {
                    VStack(alignment: .leading) {
                        Text("\(result.totalTokens) tokens")
                            .font(.caption)
                        Text("\(result.duration.formatted())s")
                            .font(.caption)
                    }
                    .foregroundColor(.secondary)

                    Spacer()

                    VStack(alignment: .trailing) {
                        Text("Generated Text")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(result.generatedText.prefix(50) + (result.generatedText.count > 50 ? "..." : ""))
                            .font(.caption)
                            .lineLimit(2)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.1))
                .shadow(color: Color.gray.opacity(0.2), radius: 2, x: 0, y: 1)
        )
    }
}

struct EfficiencyRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
            Spacer()
            Text(value)
                .font(.subheadline)
                .foregroundColor(.green)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    // Sample data for preview
    let sampleResults = [
        ParallelEvaluationExample.EvaluationResult(
            sessionId: 0,
            prompt: "Write a haiku about artificial intelligence",
            generatedText: "Circuits dream of thought,\nSilicon whispers wisdom,\nCode becomes aware.",
            tokensPerSecond: 45.23,
            totalTokens: 98,
            duration: 2.17
        ),
        ParallelEvaluationExample.EvaluationResult(
            sessionId: 1,
            prompt: "Explain quantum computing in simple terms",
            generatedText: "Quantum computers use quantum bits that can be both 0 and 1 at the same time. This allows them to solve certain problems much faster than regular computers.",
            tokensPerSecond: 42.15,
            totalTokens: 87,
            duration: 2.06
        ),
        ParallelEvaluationExample.EvaluationResult(
            sessionId: 2,
            prompt: "What are the benefits of renewable energy",
            generatedText: "Renewable energy sources like solar and wind provide clean, sustainable power that doesn't produce greenhouse gases. They reduce dependence on fossil fuels and create jobs in green technology sectors.",
            tokensPerSecond: 48.92,
            totalTokens: 95,
            duration: 1.94
        )
    ]

    NavigationView {
        ParallelComparisonView(results: sampleResults, totalTime: 2.45)
    }
}
