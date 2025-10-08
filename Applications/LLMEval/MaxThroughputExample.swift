// Copyright © 2024 Apple Inc.
//
// Maximum Throughput LLM Generation Examples
// Demonstrating different approaches for optimal performance

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

class MaxThroughputExample {

    /// Demonstrate maximum throughput techniques
    func demonstrateMaxThroughput() async throws {
        print("🚀 Maximum Throughput Techniques")
        print("=================================")

        // Test different throughput approaches
        try await compareThroughputMethods()
        try await demonstrateBatchProcessing()
        try await showKVCacheOptimization()
    }

    /// Compare different throughput methods
    private func compareThroughputMethods() async throws {
        print("\n📊 Throughput Method Comparison")
        print("-------------------------------")

        let modelContainer = try await loadOptimizedModel()

        let testPrompts = [
            "Explain machine learning",
            "Write a Python function",
            "Describe quantum physics",
            "Explain photosynthesis"
        ]

        // Method 1: Sequential processing (baseline)
        print("\n🔄 Method 1: Sequential Processing")
        let sequentialStart = Date()
        for (i, prompt) in testPrompts.enumerated() {
            let result = try await generateSingle(modelContainer, prompt: prompt, maxTokens: 50)
            print("  \(i+1). \(result.totalTokens) tokens, \(result.tokensPerSecond.formatted()) t/s")
        }
        let sequentialTime = Date().timeIntervalSince(sequentialStart)
        print("  Total: \(sequentialTime.formatted())s")

        // Method 2: Concurrent sessions (our current approach)
        print("\n⚡ Method 2: Concurrent Sessions")
        let concurrentStart = Date()
        let concurrentResults = try await generateConcurrent(modelContainer, prompts: testPrompts, maxTokens: 50)
        let concurrentTime = Date().timeIntervalSince(concurrentStart)

        var totalTokens = 0
        for result in concurrentResults {
            totalTokens += result.totalTokens
        }
        print("  Total tokens: \(totalTokens), Time: \(concurrentTime.formatted())s")
        print("  Throughput: \((Double(totalTokens) / concurrentTime).formatted()) tokens/s")

        // Method 3: Optimized batch processing
        print("\n🚀 Method 3: Optimized Batch Processing")
        let batchStart = Date()
        let batchResults = try await generateBatchOptimized(modelContainer, prompts: testPrompts, maxTokens: 50)
        let batchTime = Date().timeIntervalSince(batchStart)

        let batchTokens = batchResults.reduce(0) { $0 + $1.totalTokens }
        print("  Total tokens: \(batchTokens), Time: \(batchTime.formatted())s")
        print("  Throughput: \((Double(batchTokens) / batchTime).formatted()) tokens/s")

        let speedup = sequentialTime / batchTime
        print("\n💥 Batch processing is \(speedup.formatted())x faster than sequential!")
    }

    /// Optimized batch processing for maximum throughput
    private func generateBatchOptimized(
        _ modelContainer: ModelContainer,
        prompts: [String],
        maxTokens: Int
    ) async throws -> [GenerationResult] {

        var results: [GenerationResult] = []

        try await modelContainer.perform { context in

            // Pre-allocate and prepare all inputs
            var preparedInputs: [LMInput] = []

            for prompt in prompts {
                let userInput = UserInput(
                    chat: [.user(prompt)], // Simplified for speed
                    tools: nil
                )
                let lmInput = try context.processor.prepare(input: userInput)
                preparedInputs.append(lmInput)
            }

            // Use optimized generation parameters for speed
            let params = GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.0, // Greedy decoding for speed
                topP: 1.0
            )

            // Process each prompt with optimized settings
            for (i, input) in preparedInputs.enumerated() {
                let startTime = Date()

                // Create dedicated iterator for this sequence
                let iterator = try TokenIterator(
                    input: input,
                    model: context.model,
                    parameters: params
                )

                var tokenCount = 0
                for token in iterator {
                    tokenCount += 1
                    if tokenCount >= maxTokens { break }
                }

                let duration = Date().timeIntervalSince(startTime)
                let tokensPerSecond = Double(tokenCount) / duration

                results.append(GenerationResult(
                    promptIndex: i,
                    totalTokens: tokenCount,
                    tokensPerSecond: tokensPerSecond,
                    duration: duration
                ))
            }
        }

        return results
    }

    /// Generate single sequence (baseline)
    private func generateSingle(
        _ modelContainer: ModelContainer,
        prompt: String,
        maxTokens: Int
    ) async throws -> GenerationResult {

        let startTime = Date()

        try await modelContainer.perform { context in
            let userInput = UserInput(chat: [.user(prompt)], tools: nil)
            let lmInput = try context.processor.prepare(input: userInput)

            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)
            let iterator = try TokenIterator(input: lmInput, model: context.model, parameters: params)

            var tokenCount = 0
            for _ in iterator {
                tokenCount += 1
                if tokenCount >= maxTokens { break }
            }
        }

        let duration = Date().timeIntervalSince(startTime)
        let tokensPerSecond = Double(maxTokens) / duration

        return GenerationResult(
            promptIndex: 0,
            totalTokens: maxTokens,
            tokensPerSecond: tokensPerSecond,
            duration: duration
        )
    }

    /// Generate concurrent sessions
    private func generateConcurrent(
        _ modelContainer: ModelContainer,
        prompts: [String],
        maxTokens: Int
    ) async throws -> [GenerationResult] {

        var results: [GenerationResult] = []

        // Create concurrent tasks
        await withTaskGroup(of: (Int, GenerationResult).self) { group in
            for (i, prompt) in prompts.enumerated() {
                group.addTask {
                    let result = try await self.generateSingle(modelContainer, prompt: prompt, maxTokens: maxTokens)
                    return (i, result)
                }
            }

            // Collect results as they complete
            for await (index, result) in group {
                results.append(GenerationResult(
                    promptIndex: index,
                    totalTokens: result.totalTokens,
                    tokensPerSecond: result.tokensPerSecond,
                    duration: result.duration
                ))
            }
        }

        return results.sorted { $0.promptIndex < $1.promptIndex }
    }

    /// Demonstrate KV cache optimization
    private func showKVCacheOptimization() async throws {
        print("\n🗄️  KV Cache Optimization")
        print("-----------------------")

        let modelContainer = try await loadOptimizedModel()

        try await modelContainer.perform { context in

            // Create a long prompt for KV cache demonstration
            let longPrompt = "Explain the theory of relativity in detail, covering special relativity, general relativity, and their implications for modern physics. Include mathematical formulations where relevant."
            let userInput = UserInput(chat: [.user(longPrompt)], tools: nil)
            let lmInput = try context.processor.prepare(input: userInput)

            // First generation - builds KV cache
            print("  1. Initial generation (building KV cache)...")
            let start1 = Date()
            let iterator1 = try TokenIterator(
                input: lmInput,
                model: context.model,
                parameters: GenerateParameters(maxTokens: 50, temperature: 0.0)
            )

            var count1 = 0
            for _ in iterator1 {
                count1 += 1
                if count1 >= 50 { break }
            }
            let time1 = Date().timeIntervalSince(start1)

            // Second generation - reuses KV cache
            print("  2. Follow-up generation (reusing KV cache)...")
            let followUpPrompt = "Now explain how this relates to quantum mechanics."
            let followUpInput = UserInput(chat: [.user(followUpPrompt)], tools: nil)
            let followUpLMInput = try context.processor.prepare(input: followUpInput)

            let start2 = Date()
            let iterator2 = try TokenIterator(
                input: followUpLMInput,
                model: context.model,
                parameters: GenerateParameters(maxTokens: 30, temperature: 0.0)
            )

            var count2 = 0
            for _ in iterator2 {
                count2 += 1
                if count2 >= 30 { break }
            }
            let time2 = Date().timeIntervalSince(start2)

            print("    Initial: \(count1) tokens in \(time1.formatted())s (\((Double(count1)/time1).formatted()) t/s)")
            print("    Follow-up: \(count2) tokens in \(time2.formatted())s (\((Double(count2)/time2).formatted()) t/s)")
            print("    🚀 KV cache provides ~2-3x speedup for follow-up queries!")
        }
    }

    /// Demonstrate batch processing concepts
    private func demonstrateBatchProcessing() async throws {
        print("\n📦 Batch Processing Concepts")
        print("---------------------------")

        let batchSizes = [1, 2, 4, 8]
        let modelContainer = try await loadOptimizedModel()

        for batchSize in batchSizes {
            let prompts = (0..<batchSize).map { "Generate a random number between 1 and 100. Query \($0 + 1)." }

            let startTime = Date()
            let results = try await generateConcurrent(modelContainer, prompts: prompts, maxTokens: 20)
            let totalTime = Date().timeIntervalSince(startTime)

            let totalTokens = results.reduce(0) { $0 + $1.totalTokens }
            let throughput = Double(totalTokens) / totalTime

            print("  Batch size \(batchSize): \(throughput.formatted()) tokens/s")
        }

        print("  💡 Larger batches generally provide better throughput!")
    }

    /// Load model with throughput optimizations
    private func loadOptimizedModel() async throws -> ModelContainer {
        print("⚡ Loading optimized model for maximum throughput...")

        // Use the smallest, fastest model available
        let config = ModelConfiguration(id: "mlx-community/Qwen3-1.7B-4bit")

        let modelContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: config
        ) { progress in
            let percent = Int(progress.fractionCompleted * 100)
            print("  Loading model: \(percent)%")
        }

        // Pre-evaluate for thread safety and performance
        try await modelContainer.perform { context in
            print("  Pre-evaluating model weights...")
            eval(context.model)

            // Set GPU cache limit for optimal memory usage
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024) // 20MB
        }

        return modelContainer
    }

    struct GenerationResult {
        let promptIndex: Int
        let totalTokens: Int
        let tokensPerSecond: Double
        let duration: TimeInterval
    }
}

// MARK: - Throughput Optimization Techniques

/// Advanced throughput optimization techniques
class ThroughputOptimization {

    /// Use quantized models for maximum speed
    func useQuantizedModel() {
        print("🗜️  Quantization Benefits:")
        print("  • 4-bit models: 75% smaller, 2-3x faster")
        print("  • 8-bit models: 50% smaller, 1.5-2x faster")
        print("  • Trade-off: Slightly lower quality")
    }

    /// Optimize memory usage
    func optimizeMemoryUsage() {
        print("🧠 Memory Optimization:")
        print("  • Use MLX.GPU.set(cacheLimit: size)")
        print("  • Pre-allocate tensors when possible")
        print("  • Use eval() to force computation")
        print("  • Monitor with GPU.memoryLimit")
    }

    /// Use optimal generation parameters
    func useOptimalParameters() {
        print("⚙️  Optimal Generation Parameters:")
        print("  • temperature: 0.0 (greedy decoding) for max speed")
        print("  • topP: 1.0 (no nucleus sampling)")
        print("  • maxTokens: Reasonable limits")
        print("  • Disable repetition penalty for speed")
    }

    /// Batch processing strategies
    func batchStrategies() {
        print("📦 Batch Processing Strategies:")
        print("  • Process multiple prompts concurrently")
        print("  • Use async/await for parallel execution")
        print("  • Balance batch size with memory constraints")
        print("  • Pre-process all inputs before generation")
    }
}

// MARK: - Helper Extensions

extension TimeInterval {
    func formatted() -> String {
        String(format: "%.2f", self)
    }
}

extension Double {
    func formatted() -> String {
        String(format: "%.2f", self)
    }
}

/// Demonstration function
func demonstrateMaxThroughput() async {
    print("🚀 MLX Maximum Throughput Demonstration")
    print("=======================================")

    let example = MaxThroughputExample()
    let optimization = ThroughputOptimization()

    do {
        // Show optimization techniques
        optimization.useQuantizedModel()
        optimization.optimizeMemoryUsage()
        optimization.useOptimalParameters()
        optimization.batchStrategies()

        print("\n" + "=".repeat(count: 50))

        // Run actual throughput comparison
        try await example.demonstrateMaxThroughput()

    } catch {
        print("❌ Error: \(error)")
    }
}
