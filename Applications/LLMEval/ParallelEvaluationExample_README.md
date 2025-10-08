# Parallel LLM Evaluation Example

This example demonstrates **safe concurrent execution** of multiple LLM evaluation sessions while properly handling MLX's thread safety requirements.

## 🎯 Purpose

This example addresses the key challenge of running multiple LLM sessions simultaneously (like a server responding to multiple requests) while ensuring thread safety. It demonstrates:

- ✅ **Concurrent session execution** - Multiple LLM evaluations running in parallel
- ✅ **Thread safety** - Proper handling of unevaluated MLXArrays
- ✅ **Isolated random states** - Task-local random state using `withRandomState`
- ✅ **Session isolation** - Each session gets its own KV cache
- ✅ **GPU serialization awareness** - Understanding MLX's GPU operation model

## 🚀 Key Features

### Thread Safety Solutions

1. **Pre-evaluated Model Weights**
   ```swift
   // CRITICAL: Force evaluation of all parameters
   // This ensures no unevaluated MLXArrays remain
   eval(context.model)
   ```

2. **Task-Local Random State**
   ```swift
   // Each session gets its own isolated random state
   let sessionRandomState = MLXRandom.RandomState(seed: UInt64(sessionId * 1000))

   try withRandomState(sessionRandomState) {
       // Safe concurrent random operations
   }
   ```

3. **Session-Isolated KV Caches**
   - Each evaluation session gets its own `TokenIterator`
   - No shared KV cache state between sessions
   - Automatic cache management per session

### Performance Characteristics

- **CPU Parallelism**: Preprocessing and tokenization can run concurrently
- **GPU Serialization**: MLX operations serialize on GPU but maintain thread safety
- **Memory Management**: Configurable GPU cache limits (`20MB` default)

## 📊 Usage

### Basic Usage

```swift
let example = ParallelEvaluationExample()
try await example.runDefaultExample()
```

### Advanced Configuration

```swift
let config = ParallelEvaluationExample.ParallelConfig(
    numConcurrentSessions: 3,
    modelConfiguration: LLMRegistry.qwen3_1_7b_4bit,
    prompts: [
        "Write a haiku about AI.",
        "Explain neural networks simply.",
        "What is machine learning?"
    ],
    maxTokens: 100,
    temperature: 0.7
)

try await example.runParallelEvaluation(config: config)
```

### Integration with LLMEval App

The example includes a **"Run Parallel Demo"** button in the main LLMEval app that demonstrates the parallel execution. This shows how to integrate concurrent evaluation into existing applications.

## 🔧 Technical Implementation

### Core Components

1. **`ParallelEvaluationExample`** - Main demonstration class
2. **`ParallelConfig`** - Configuration for concurrent sessions
3. **`EvaluationResult`** - Structured results from each session
4. **Task-based concurrency** - Swift's structured concurrency for safe parallel execution

### Thread Safety Mechanisms

| Concern | Solution | Implementation |
|---------|----------|----------------|
| Unevaluated MLXArrays | Pre-evaluation | `eval(context.model)` |
| Global random state | Task-local state | `withRandomState(randomState)` |
| Shared KV cache | Session isolation | Per-session `TokenIterator` |
| GPU operations | Serialization awareness | MLX's built-in GPU management |

### Performance Monitoring

Each session tracks:
- **Tokens per second** - Generation throughput
- **Total tokens** - Output length
- **Duration** - Session execution time
- **Session ID** - For result correlation

## 🎯 Example Output

```
🎯 Starting parallel LLM evaluation with 3 sessions
📋 Model: mlx-community/Qwen3-1.7B-4bit
🔥 Temperature: 0.7
📝 Max tokens: 100

⚡ Running 3 evaluation sessions concurrently...
💡 Note: GPU operations will serialize, but CPU preprocessing can run in parallel

🚀 Starting session 0 with prompt: Write a short poem about machine learning...
🚀 Starting session 1 with prompt: Explain the concept of neural networks simply...
🚀 Starting session 2 with prompt: What are the key differences between AI and machine learning?...

✅ Session 0 completed: 45.23 tokens/sec, 98 tokens
✅ Session 1 completed: 42.15 tokens/sec, 87 tokens
✅ Session 2 completed: 48.92 tokens/sec, 95 tokens

============================================================
📊 PARALLEL EVALUATION RESULTS
============================================================
🎯 Session 0:
   📝 Prompt: Write a short poem about machine learning...
   📊 Performance: 45.23 tokens/sec
   🔢 Tokens: 98
   ⏱️  Duration: 2.17s
   💬 Generated: In circuits deep, where data flows like streams...

📈 Average performance: 45.43 tokens/sec across 3 sessions
⏱️  Total time: 2.45s

🔒 Thread safety maintained through:
   • Pre-evaluated model weights
   • Task-local random states
   • Session-isolated KV caches
   • GPU operation serialization
```

## 🏃‍♂️ Running the Example

### In LLMEval App
1. Open the LLMEval app
2. Click **"Run Parallel Demo"** button
3. Watch the console output for parallel execution results

### Programmatic Usage
```swift
// Simple demo
await demonstrateParallelEvaluation()

// Advanced configuration
await runAdvancedParallelEvaluation()
```

## 🔒 Thread Safety Verification

The example includes comprehensive thread safety measures:

- **Model Weight Evaluation**: Forces evaluation of all model parameters before concurrent use
- **Random State Isolation**: Each session uses `withRandomState()` with unique seeds
- **KV Cache Separation**: Each session maintains its own KV cache state
- **GPU Memory Limits**: Configurable cache limits prevent memory issues

## 💡 Best Practices Demonstrated

1. **Pre-evaluate Models**: Always call `eval()` on models before concurrent use
2. **Use Task-Local Random State**: Employ `withRandomState()` for concurrent random operations
3. **Isolate Sessions**: Give each evaluation session its own resources
4. **Monitor Performance**: Track tokens/second and resource usage
5. **Handle Errors Gracefully**: Use proper error handling in concurrent code

This example serves as a reference implementation for safely running multiple LLM sessions concurrently in production applications.
