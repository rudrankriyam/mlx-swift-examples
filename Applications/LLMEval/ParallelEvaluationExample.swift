// Copyright © 2024 Apple Inc.
//
// Example demonstrating parallel LLM evaluation with proper thread safety handling
// This shows how to run multiple LLM sessions simultaneously while managing:
// - Unevaluated MLXArray thread safety
// - Task-local random state isolation
// - Session-specific KV cache management
// - GPU serialization

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Example class demonstrating parallel LLM evaluation
class ParallelEvaluationExample {

    /// Configuration for parallel evaluation
    struct ParallelConfig {
        let numConcurrentSessions: Int
        let modelConfiguration: ModelConfiguration
        let prompts: [String]
        let maxTokens: Int
        let temperature: Float

        init(
            numConcurrentSessions: Int = 2,
            modelConfiguration: ModelConfiguration = LLMRegistry.qwen3_1_7b_4bit,
            prompts: [String] = [
                "Write a haiku about artificial intelligence.",
                "Explain quantum computing in simple terms.",
                "What are the benefits of renewable energy?"
            ],
            maxTokens: Int = 2000,
            temperature: Float = 0.7
        ) {
            self.numConcurrentSessions = numConcurrentSessions
            self.modelConfiguration = modelConfiguration
            self.prompts = prompts
            self.maxTokens = maxTokens
            self.temperature = temperature
        }
    }

    /// Result of a single evaluation session
    struct EvaluationResult {
        let sessionId: Int
        let prompt: String
        let generatedText: String
        let tokensPerSecond: Double
        let totalTokens: Int
        let duration: TimeInterval
    }

    /// Load and prepare model for parallel evaluation
    /// - Important: Pre-evaluates all model weights to avoid thread safety issues
    func loadModel(for config: ParallelConfig) async throws -> ModelContainer {
        print("🔄 Loading model: \(config.modelConfiguration.id)")

        let modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: config.modelConfiguration
        ) { progress in
            Task { @MainActor in
                print("📥 Download progress: \(Int(progress.fractionCompleted * 100))%")
            }
        }

        // CRITICAL: Pre-evaluate all model weights to avoid thread safety issues
        // Unevaluated MLXArrays are not thread-safe and will cause crashes
        print("⚡ Pre-evaluating model weights for thread safety...")
        await modelContainer.perform { context in
            let numParams = context.model.numParameters()
            print("📊 Model loaded with \(numParams / (1024*1024))M parameters")

            // Force evaluation of all parameters
            // This ensures no unevaluated MLXArrays remain
            eval(context.model)
        }

        // Limit GPU memory usage
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024) // 20MB cache limit

        return modelContainer
    }

    /// Run a single evaluation session with isolated random state
    /// - Note: Uses task-local random state to avoid global state conflicts
    func evaluateSession(
        sessionId: Int,
        prompt: String,
        modelContainer: ModelContainer,
        config: ParallelConfig
    ) async throws -> EvaluationResult {

        let startTime = Date()

        print("🚀 Starting session \(sessionId) with prompt: \(prompt.prefix(50))...")

        let parameters = GenerateParameters(
            maxTokens: config.maxTokens,
            temperature: config.temperature
        )

        var generatedTokens: [Int] = []
        var totalTokens = 0

        // Each session gets its own isolated random state
        // This prevents interference between concurrent sessions
        let sessionRandomState = MLXRandom.RandomState(seed: UInt64(sessionId * 1000))

        try await modelContainer.perform { (context: ModelContext) -> Void in
            let userInput = UserInput(
                chat: [.system("You are a helpful assistant"), .user(prompt)],
                tools: nil
            )

            let lmInput = try await context.processor.prepare(input: userInput)

            // Use task-local random state for this session
            // This isolates the random state from other concurrent sessions
            try withRandomState(sessionRandomState) {
                let iterator = try TokenIterator(
                    input: lmInput,
                    model: context.model,
                    parameters: parameters
                )

                // Collect all generated tokens
                for token in iterator {
                    generatedTokens.append(token)
                    totalTokens += 1
                }
            }
        }

        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(totalTokens) / duration

        // Decode the generated tokens to text
        var decodedText = ""
        await modelContainer.perform { context in
            decodedText = context.tokenizer.decode(tokens: generatedTokens)
        }

        let result = EvaluationResult(
            sessionId: sessionId,
            prompt: prompt,
            generatedText: decodedText,
            tokensPerSecond: tokensPerSecond,
            totalTokens: totalTokens,
            duration: duration
        )

        print("✅ Session \(sessionId) completed: \(tokensPerSecond.formatted()) tokens/sec, \(totalTokens) tokens")

        return result
    }

    /// Run a single evaluation session with real-time token streaming
    func evaluateSessionWithStreaming(
        sessionId: Int,
        prompt: String,
        modelContainer: ModelContainer,
        config: ParallelConfig,
        onTokenGenerated: @escaping (String) -> Void
    ) async throws -> EvaluationResult {

        let startTime = Date()

        print("🚀 Starting streaming session \(sessionId) with prompt: \(prompt.prefix(50))...")

        let parameters = GenerateParameters(
            maxTokens: config.maxTokens,
            temperature: config.temperature
        )

        var generatedTokens: [Int] = []
        var totalTokens = 0
        var streamedText = ""

        // Each session gets its own isolated random state
        let sessionRandomState = MLXRandom.RandomState(seed: UInt64(sessionId * 1000))

        try await modelContainer.perform { (context: ModelContext) -> Void in
            let userInput = UserInput(
                chat: [.system("You are a helpful assistant"), .user(prompt)],
                tools: nil
            )

            let lmInput = try await context.processor.prepare(input: userInput)

            // Use task-local random state for this session
            try withRandomState(sessionRandomState) {
                let iterator = try TokenIterator(
                    input: lmInput,
                    model: context.model,
                    parameters: parameters
                )

                // Stream tokens in real-time
                for token in iterator {
                    generatedTokens.append(token)
                    totalTokens += 1

                    // Decode the new token and stream it
                    let tokenText = context.tokenizer.decode(tokens: [token])
                    streamedText += tokenText

                    // Call the streaming callback
                    onTokenGenerated(tokenText)
                }
            }
        }

        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(totalTokens) / duration

        let result = EvaluationResult(
            sessionId: sessionId,
            prompt: prompt,
            generatedText: streamedText,
            tokensPerSecond: tokensPerSecond,
            totalTokens: totalTokens,
            duration: duration
        )

        print("✅ Streaming session \(sessionId) completed: \(tokensPerSecond.formatted()) tokens/sec, \(totalTokens) tokens")

        return result
    }

    /// Run parallel evaluation sessions with streaming support
    /// Demonstrates how to safely run multiple LLM sessions concurrently
    func runParallelEvaluation(config: ParallelConfig = ParallelConfig()) async throws -> (results: [EvaluationResult], totalTime: TimeInterval) {

        let (results, totalTime) = try await runParallelEvaluationWithStreaming(config: config)
        return (results, totalTime)
    }

    /// Run parallel evaluation with real-time streaming callbacks
    func runParallelEvaluationWithStreaming(
        config: ParallelConfig = ParallelConfig(),
        onTokenGenerated: @escaping (Int, String) -> Void = { _, _ in },
        onSessionComplete: @escaping (Int, EvaluationResult) -> Void = { _, _ in }
    ) async throws -> (results: [EvaluationResult], totalTime: TimeInterval) {

        print("🎯 Starting parallel LLM evaluation with \(config.numConcurrentSessions) sessions")
        print("📋 Model: \(config.modelConfiguration.id)")
        print("🔥 Temperature: \(config.temperature)")
        print("📝 Max tokens: \(config.maxTokens)")

        // Step 1: Load model with pre-evaluation for thread safety
        let modelContainer = try await loadModel(for: config)

        // Step 2: Prepare evaluation tasks with streaming support
        let evaluationTasks = (0..<config.numConcurrentSessions).map { sessionId in
            Task {
                let prompt = config.prompts[sessionId % config.prompts.count]
                return try await evaluateSessionWithStreaming(
                    sessionId: sessionId,
                    prompt: prompt,
                    modelContainer: modelContainer,
                    config: config,
                    onTokenGenerated: { token in
                        onTokenGenerated(sessionId, token)
                    }
                )
            }
        }

        print("\n⚡ Running \(evaluationTasks.count) evaluation sessions concurrently...")
        print("💡 Note: GPU operations will serialize, but CPU preprocessing can run in parallel")

        let startTime = Date()

        // Step 3: Execute all evaluation tasks concurrently
        // Note: While CPU work happens in parallel, GPU operations serialize
        var results: [EvaluationResult] = []
        for task in evaluationTasks {
            let result = try await task.value
            results.append(result)
            onSessionComplete(result.sessionId, result)
        }

        let totalTime = Date().timeIntervalSince(startTime)

        // Step 4: Report results
        print("\n" + "=".repeat(count: 60))
        print("📊 PARALLEL EVALUATION RESULTS")
        print("=".repeat(count: 60))

        for result in results.sorted(by: { $0.sessionId < $1.sessionId }) {
            print("""
            🎯 Session \(result.sessionId):
               📝 Prompt: \(result.prompt.prefix(40))...
               📊 Performance: \(result.tokensPerSecond.formatted()) tokens/sec
               🔢 Tokens: \(result.totalTokens)
               ⏱️  Duration: \(result.duration.formatted())s
               💬 Generated: \(result.generatedText.prefix(80))...

            """)
        }

        let avgTokensPerSecond = results.map { $0.tokensPerSecond }.reduce(0, +) / Double(results.count)
        print("📈 Average performance: \(avgTokensPerSecond.formatted()) tokens/sec across \(results.count) sessions")
        print("⏱️  Total time: \(totalTime.formatted())s")
        print("💡 Note: GPU serialization ensures thread safety but may limit parallelism benefits")

        print("\n" + "=".repeat(count: 60))
        print("✅ Parallel evaluation completed successfully!")
        print("🔒 Thread safety maintained through:")
        print("   • Pre-evaluated model weights")
        print("   • Task-local random states")
        print("   • Session-isolated KV caches")
        print("   • GPU operation serialization")

        return (results, totalTime)
    }

    /// Convenience method to run with default configuration
    func runDefaultExample() async throws {
        let config = ParallelConfig(
            numConcurrentSessions: 3,
            prompts: [
                "Write a short poem about machine learning.",
                "Explain the concept of neural networks simply.",
                "What are the key differences between AI and machine learning?"
            ]
        )
        try await runParallelEvaluation(config: config)
    }
}

// MARK: - Helper Extensions

extension String {
    func `repeat`(count: Int) -> String {
        String(repeating: self, count: count)
    }
}

extension Double {
    func formatted() -> String {
        String(format: "%.2f", self)
    }
}

// MARK: - Usage Example

/// Example usage function that can be called from ContentView or other parts of the app
func demonstrateParallelEvaluation() async {
    print("🚀 Demonstrating Parallel LLM Evaluation")
    print("This example shows how to safely run multiple LLM sessions concurrently")
    print("while handling MLX thread safety concerns.\n")

    let example = ParallelEvaluationExample()

    do {
        try await example.runDefaultExample()
    } catch {
        print("❌ Error during parallel evaluation: \(error)")
    }
}

/// Advanced example with custom configuration
func runAdvancedParallelEvaluation() async {
    let config = ParallelEvaluationExample.ParallelConfig(
        numConcurrentSessions: 2,
        modelConfiguration: LLMRegistry.qwen3_1_7b_4bit,
        prompts: [
            "Analyze the environmental impact of electric vehicles.",
            "Describe the process of photosynthesis in detail."
        ],
        maxTokens: 150,
        temperature: 0.8
    )

    let example = ParallelEvaluationExample()
    do {
        try await example.runParallelEvaluation(config: config)
    } catch {
        print("❌ Error: \(error)")
    }
}
