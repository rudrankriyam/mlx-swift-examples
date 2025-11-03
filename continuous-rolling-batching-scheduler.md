# Continuous & Rolling Batching Scheduler — Simplified Design
 Status: Implementation-ready design
 Target stack: Swift 6+, MLX (LanguageModel), macOS/iOS
 Primary reference: Libraries/MLXLMCommon/BatchGenerate.swift
 Strategy: Reuse existing code, add minimal scheduler layer (~300 lines)

 ──────────────────────────────────────────

## Goals


 1. Solo Fast Path: Use existing generate() for single requests (better TPS than batch=1)
 2. Batched Mode: Use existing BatchTokenIterator for 2+ concurrent requests
 3. Per-Request Params: Each request has its own sampling parameters (temperature, top-p, etc.) set
    by client
 4. Concurrency Limits: Configurable maxConcurrentRequests
 5. Statistics: Both per-request stats (for client) and aggregated stats (for logging)
 6. Cancellation: Remove cancelled streams from batch immediately
  . Swift 6 Safe: Full concurrency safety with actors and Sendable types
 ──────────────────────────────────────────

## What We're Reusing (No Need to Reimplement)

 ✅ From BatchGenerate.swift:

 •  BatchTokenIterator - rolling batch with attach/detach via insert()
 •  ActiveBatch - batch state with filter() and extend()
 •  processPrompts() - prefill with chunking
 •  leftPadPrompts(), makeBatchCache(), convertCache() - utilities
    Stats tracking (prompt/generation time, TPS)
 ✅ From Evaluate.swift:

 •  generate() returning AsyncStream<Generation> - perfect for solo mode
    TokenIterator - single-stream generation (fast path)
 ✅ From ModelContainer.swift:

 •  Actor patterns for model access
    ModelContext bundling model + tokenizer + processor
 ──────────────────────────────────────────

## Architecture (Simplified)


       ↓t Request (with params)
   ┌──────────────────────────┐
   │  InferenceScheduler      │ (actor, ~150 lines)
   │  - Queue management      │
   │  - Concurrency limits    │
   │  - Stats tracking        │
   └──────────────────────────┘
       ↓
       ├─→ Solo Mode (1 active request)
       │   └─→ Use existing generate() → AsyncStream<Generation>
       │       [Better TPS than batch=1]
       │
       └─→ Batch Mode (2+ active requests)
           └─→ Use BatchTokenIterator with per-UID samplers
               [Efficient concurrent processing]

 Key Insight: Don't promote from solo to batch - just switch code paths when count changes.

 ──────────────────────────────────────────

## Configuration


``` swift
       /// Maximum concurrent requests being processed
       public var maxConcurrentRequests: Int = 64

       /// Batch size for concurrent processing (when 2+ requests active)
       public var completionBatchSize: Int = 16

       /// Prefill batch size
       public var prefillBatchSize: Int = 8

       /// Prefill step size for chunking
       public var prefillStepSize: Int = 2_048

       /// Whether to collect per-token log probabilities (expensive)
       public var returnLogProbs: Bool = true
   }
```

 ──────────────────────────────────────────

## Core Types


``` swift
   public struct InferenceRequest: Sendable, Identifiable {
       public let id: UUID
       public let tokens: [Int]
       public let params: GenerateParameters  // Client sets temp, top-p, etc.
       public let maxTokens: Int
       public let createdAt: Date
   }

   /// Token emitted to client
   public struct TokenEvent: Sendable {
       public let requestID: UUID
       public let token: Int
       public let textDelta: String?
       public let finishReason: BatchGenerateResult.FinishReason?
       public let logProbs: MLXArray?  // If returnLogProbs enabled
   }

   /// Per-request statistics (for client)
   public struct RequestStats: Sendable {
       public let requestID: UUID
       public let promptTokens: Int
       public let generatedTokens: Int
       public let promptTime: TimeInterval
       public let generationTime: TimeInterval
       public var tokensPerSecond: Double {
           guard generationTime > 0 else { return 0 }
           return Double(generatedTokens) / generationTime
       }
   }

   /// Aggregated statistics (for logging)
   public struct AggregatedStats: Sendable {
       public let activeRequests: Int
       public let queuedRequests: Int
       public let totalRequestsProcessed: Int
       public let averageTPS: Double
       public let peakMemoryGB: Double
   }
```

---

## Internal Types

```swift
/// Internal tracking of active request with continuation and stats
private struct ActiveRequest {
    let request: InferenceRequest
    let continuation: AsyncStream<TokenEvent>.Continuation
    var stats: RequestStats
    var uid: Int?  // UID in BatchTokenIterator (nil if in solo mode)
    let startTime: Date
}

/// Scheduler mode state machine
private enum Mode {
    case idle
    case solo(requestID: UUID, task: Task<Void, Never>)
    case batch(task: Task<Void, Never>)
}

/// UID to RequestID mapping for batch mode
private typealias UIDMapping = [Int: UUID]
```

 ──────────────────────────────────────────

 Public API


``` swift
       /// Initialize with model context and configuration
       /// - Parameters:
       ///   - context: Pre-loaded ModelContext or load via modelID string
       ///   - config: Scheduler configuration
       public init(context: ModelContext, config: SchedulerConfig = .init())

       /// Submit a request for processing
       /// - Parameter request: Request with tokens and client-specified params
       /// - Returns: AsyncStream of tokens as they're generated
       public func submit(_ request: InferenceRequest) -> AsyncStream<TokenEvent>

       /// Get per-request statistics
       /// - Parameter requestID: UUID of the request
       /// - Returns: Stats for that specific request, or nil if not found
       public func stats(for requestID: UUID) -> RequestStats?

       /// Get aggregated statistics across all requests
       public func aggregatedStats() -> AggregatedStats

       /// Gracefully shutdown the scheduler
       public func shutdown() async
   }
```

 ──────────────────────────────────────────

## Key Implementation Details

 1. Solo vs Batch Decision (Simple!)


``` swift
       private var activeRequests: [UUID: ActiveRequest] = [:]
       private var queuedRequests: [InferenceRequest] = []

       private func startGeneration(_ request: InferenceRequest) async {
           if activeRequests.count == 1 {
               // SOLO MODE: Use existing generate() - better TPS
               await runSoloGeneration(request)
           } else if activeRequests.count >= 2 {
               // BATCH MODE: Use BatchTokenIterator
               await runBatchedGeneration()
           }
       }
   }
```

 No complex promotion! Just switch between existing code paths.

 2. Per-UID Samplers in BatchTokenIterator

 Only change needed to existing code: (~50 lines in BatchTokenIterator)


``` swift
   private let sampler: LogitSamplerfrom:

   // To:
   private var samplers: [Int: LogitSampler] = [:]
   private var processors: [Int: LogitProcessor] = [:]

   // Update insert() signature:
   public mutating func insert(
       prompts: [[Int]],
       maxTokens: [Int]?,
       samplers: [LogitSampler]  // NEW: per-request
   ) -> [Int]

   // In step(), change sampling logic:
   private mutating func step(inputTokens: MLXArray, cache: inout [KVCache])
       -> (MLXArray, MLXArray?)
   {
       let logits = model(inputTokens, cache: cache)
       let selected = logits[0..., -1, 0...]

       // Sample per-UID instead of batch-wide
       var sampledTokens = [Int32]()
       for (idx, uid) in currentBatch.uids.enumerated() {
           let sampler = samplers[uid]!
           let tokenLogits = selected[idx]
           sampledTokens.append(sampler.sample(logits: tokenLogits).item(Int32.self))
       }
       let sampled = MLXArray(sampledTokens)

       if returnLogProbs {
           return (sampled, selected)
       } else {
           return (sampled, nil)
       }
   }
```

 3. Cancellation Support

``` swift
       private var continuations: [UUID: AsyncStream<TokenEvent>.Continuation] = [:]

       public func submit(_ request: InferenceRequest) -> AsyncStream<TokenEvent> {
           AsyncStream { continuation in
               Task {
                   // Track continuation for cancellation
                   await track(continuation, for: request.id)

                   // Handle cancellation
                   continuation.onTermination = { @Sendable _ in
                       Task {
                           await self.handleCancellation(request.id)
                       }
                   }

                   await enqueueRequest(request)
               }
           }
       }

       private func handleCancellation(_ requestID: UUID) {
           // Remove from active batch immediately
           if let uid = requestToUID[requestID] {
               cancelledUIDs.insert(uid)
               // On next tick(), filter() will remove it from ActiveBatch
           }

           // Clean up tracking
           activeRequests.removeValue(forKey: requestID)
           requestStats.removeValue(forKey: requestID)
           continuations.removeValue(forKey: requestID)
       }
   }
```

 4. Statistics Tracking

``` swift
       // Per-request stats {
       private var requestStats: [UUID: RequestStats] = [:]

       // Aggregated stats
       private var totalRequests = 0
       private var completedRequests = 0
       private var totalTokensGenerated = 0
       private var totalGenerationTime: TimeInterval = 0

       private func updateStats(for requestID: UUID, token: Int, time: TimeInterval) {
           // Update per-request
           if var stats = requestStats[requestID] {
               stats.generatedTokens += 1
               stats.generationTime += time
               requestStats[requestID] = stats
           }

           // Update aggregated
           totalTokensGenerated += 1
           totalGenerationTime += time
       }

       public func stats(for requestID: UUID) -> RequestStats? {
           requestStats[requestID]
       }

       public func aggregatedStats() -> AggregatedStats {
           AggregatedStats(
               activeRequests: activeRequests.count,
               queuedRequests: queuedRequests.count,
               totalRequestsProcessed: completedRequests,
               averageTPS: totalGenerationTime > 0
                   ? Double(totalTokensGenerated) / totalGenerationTime
                   : 0,
               peakMemoryGB: Double(GPU.peakMemory) / 1_000_000_000.0
           )
       }
   }
```

---

## Complete Solo Mode Implementation

```swift
/// Run generation for a single request using the fast path
private func runSoloGeneration(_ request: InferenceRequest) async {
    let requestID = request.id
    let startTime = Date.timeIntervalSinceReferenceDate
    
    guard let activeReq = activeRequests[requestID] else { return }
    
    do {
        // Build LMInput from tokens
        let lmInput = LMInput(text: .init(tokens: request.tokens))
        
        // Get stop tokens from context
        var stopTokens = Set<Int>()
        if let eos = context.tokenizer.eosTokenId {
            stopTokens.insert(eos)
        }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                stopTokens.insert(id)
            }
        }
        
        // Create cache
        let cache = context.model.newCache(parameters: request.params)
        
        // Track timing
        var promptTokens = request.tokens.count
        var generatedTokens = 0
        var promptTime: TimeInterval = 0
        var firstToken = true
        
        // Use existing generate() function
        for await generation in try MLXLMCommon.generate(
            input: lmInput,
            cache: cache,
            parameters: request.params,
            context: context
        ) {
            // Check for cancellation
            if cancelledUIDs.contains(requestID) {
                activeReq.continuation.finish()
                return
            }
            
            switch generation {
            case .token(let token, let prob):
                if firstToken {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - startTime
                    firstToken = false
                }
                
                generatedTokens += 1
                
                // Generate text delta
                let textDelta = context.tokenizer.decode(tokens: [token])
                
                // Get logprobs if requested
                let logProbs: MLXArray? = config.returnLogProbs ? prob : nil
                
                // Emit token event
                let event = TokenEvent(
                    requestID: requestID,
                    token: token,
                    textDelta: textDelta,
                    finishReason: nil,
                    logProbs: logProbs
                )
                activeReq.continuation.yield(event)
                
                // Update stats
                updateStats(
                    for: requestID,
                    promptTokens: promptTokens,
                    generatedTokens: generatedTokens,
                    promptTime: promptTime,
                    generationTime: Date.timeIntervalSinceReferenceDate - startTime - promptTime
                )
                
                // Check if we should switch to batch mode
                if activeRequests.count >= 2 {
                    // New request arrived - need to switch to batch mode
                    // For simplicity, let this request finish in solo mode
                    // Next request will use batch mode
                }
                
            case .info(let info):
                // Generation complete
                let finishReason: BatchGenerateResult.FinishReason = 
                    stopTokens.contains(info.tokens.last ?? -1) ? .stop : .length
                
                let event = TokenEvent(
                    requestID: requestID,
                    token: nil,
                    textDelta: nil,
                    finishReason: finishReason,
                    logProbs: nil
                )
                activeReq.continuation.yield(event)
                activeReq.continuation.finish()
                
                completedRequests += 1
                activeRequests.removeValue(forKey: requestID)
                
                // Try to admit queued requests
                await tryAdmitFromQueue()
            }
        }
        
    } catch {
        // Handle errors by finishing the stream
        activeReq.continuation.finish()
        activeRequests.removeValue(forKey: requestID)
    }
}
```

---

## Complete Batch Mode Implementation

```swift
/// Run generation for multiple requests using batched processing
private func runBatchedGeneration() async {
    // Build stop tokens (union of all model-level stop tokens)
    var stopTokens = Set<Int>()
    if let eos = context.tokenizer.eosTokenId {
        stopTokens.insert(eos)
    }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            stopTokens.insert(id)
        }
    }
    
    // Build BatchGenerateParameters
    // Use a default GenerateParameters for batch-wide settings
    let batchParams = BatchGenerateParameters(
        maxTokens: config.completionBatchSize,
        completionBatchSize: config.completionBatchSize,
        prefillBatchSize: config.prefillBatchSize,
        prefillStepSize: config.prefillStepSize,
        generation: GenerateParameters(), // Will be overridden per-UID
        returnLogProbs: config.returnLogProbs
    )
    
    // Initialize iterator
    var iterator = BatchTokenIterator(
        model: context.model,
        parameters: batchParams,
        stopTokens: stopTokens,
        unknownTokenId: context.tokenizer.unknownTokenId
    )
    
    // Prepare initial batch
    let initialRequests = Array(activeRequests.values.prefix(config.completionBatchSize))
    if initialRequests.isEmpty { return }
    
    let prompts = initialRequests.map { $0.request.tokens }
    let maxTokens = initialRequests.map { $0.request.maxTokens }
    let samplers = initialRequests.map { $0.request.params.sampler() }
    
    // Insert and track UID mapping
    let uids = iterator.insert(prompts: prompts, maxTokens: maxTokens, samplers: samplers)
    
    var uidToRequestID: UIDMapping = [:]
    for (uid, activeReq) in zip(uids, initialRequests) {
        uidToRequestID[uid] = activeReq.request.id
        // Update activeRequest with UID
        if var req = activeRequests[activeReq.request.id] {
            req.uid = uid
            activeRequests[activeReq.request.id] = req
        }
    }
    
    let startTime = Date.timeIntervalSinceReferenceDate
    
    // Decode loop
    while let responses = iterator.next() {
        let tickTime = Date.timeIntervalSinceReferenceDate
        
        var finishedUIDs: Set<Int> = []
        
        for response in responses {
            guard let requestID = uidToRequestID[response.uid],
                  let activeReq = activeRequests[requestID] else {
                continue
            }
            
            // Check for cancellation
            if cancelledUIDs.contains(requestID) {
                finishedUIDs.insert(response.uid)
                uidToRequestID.removeValue(forKey: response.uid)
                continue
            }
            
            // Generate text delta
            let textDelta = context.tokenizer.decode(tokens: [response.token])
            
            // Create token event
            let event = TokenEvent(
                requestID: requestID,
                token: response.token,
                textDelta: textDelta,
                finishReason: response.finishReason,
                logProbs: response.logProbs
            )
            
            // Yield to client
            activeReq.continuation.yield(event)
            
            // Update stats
            updateStats(
                for: requestID,
                generatedTokens: 1,
                generationTime: tickTime - startTime
            )
            
            // Handle completion
            if let _ = response.finishReason {
                activeReq.continuation.finish()
                activeRequests.removeValue(forKey: requestID)
                uidToRequestID.removeValue(forKey: response.uid)
                completedRequests += 1
                finishedUIDs.insert(response.uid)
            }
        }
        
        // Remove cancelled UIDs
        if !cancelledUIDs.isEmpty {
            for requestID in cancelledUIDs {
                if let activeReq = activeRequests[requestID],
                   let uid = activeReq.uid {
                    finishedUIDs.insert(uid)
                    uidToRequestID.removeValue(forKey: uid)
                }
                activeRequests.removeValue(forKey: requestID)
            }
            cancelledUIDs.removeAll()
        }
        
        // Try to attach new requests if there's capacity
        let currentActive = activeRequests.count
        let capacity = config.completionBatchSize - currentActive
        
        if capacity > 0 && !queuedRequests.isEmpty {
            let newRequests = Array(queuedRequests.prefix(capacity))
            queuedRequests.removeFirst(min(capacity, queuedRequests.count))
            
            let newPrompts = newRequests.map(\.tokens)
            let newMaxTokens = newRequests.map(\.maxTokens)
            let newSamplers = newRequests.map { $0.params.sampler() }
            
            let newUIDs = iterator.insert(
                prompts: newPrompts,
                maxTokens: newMaxTokens,
                samplers: newSamplers
            )
            
            // Track new UIDs
            for (uid, request) in zip(newUIDs, newRequests) {
                uidToRequestID[uid] = request.id
                
                // Create ActiveRequest (need to get continuation from somewhere)
                // This is a simplification - actual implementation needs to
                // maintain continuations properly
            }
        }
        
        // If no more active requests, exit batch mode
        if activeRequests.isEmpty {
            break
        }
        
        await Task.yield() // Allow other tasks to run
    }
    
    // Try to admit queued requests
    await tryAdmitFromQueue()
}
```

---

## Mode Switching State Machine

```swift
/// Current scheduler mode
private var currentMode: Mode = .idle

/// Handle mode transitions based on active request count
private func updateMode() async {
    let activeCount = activeRequests.count
    
    switch (currentMode, activeCount) {
    case (.idle, 1):
        // Start solo mode
        if let firstRequest = activeRequests.values.first {
            let task = Task {
                await runSoloGeneration(firstRequest.request)
            }
            currentMode = .solo(requestID: firstRequest.request.id, task: task)
        }
        
    case (.idle, 2...):
        // Start batch mode directly
        let task = Task {
            await runBatchedGeneration()
        }
        currentMode = .batch(task: task)
        
    case (.solo(let requestID, let task), 2...):
        // Solo request is still running, but new requests arrived
        // Let solo finish, then switch to batch for remaining requests
        // This avoids complex state transfer
        
        // Mark that we need to switch
        // When solo finishes, it will check activeRequests.count
        // and the next processQueue() will start batch mode
        break
        
    case (.solo(_, let task), 0):
        // Solo request finished
        task.cancel()
        currentMode = .idle
        await tryAdmitFromQueue()
        
    case (.batch(let task), 0):
        // All batch requests finished
        task.cancel()
        currentMode = .idle
        await tryAdmitFromQueue()
        
    case (.batch(let task), 1):
        // Down to 1 request - could switch to solo
        // But for simplicity, keep in batch mode
        // The overhead is minimal for batch size of 1
        break
        
    default:
        // No transition needed
        break
    }
}

/// Try to admit queued requests when capacity is available
private func tryAdmitFromQueue() async {
    while activeRequests.count < config.maxConcurrentRequests,
          !queuedRequests.isEmpty
    {
        let request = queuedRequests.removeFirst()
        
        // Create continuation for this request
        // (This is normally done in submit())
        
        // Add to active requests
        // (Simplified - actual implementation tracks properly)
        
        // Update mode
        await updateMode()
    }
}
```

---

## Error Handling Patterns

```swift
/// Error types for scheduler operations
public enum SchedulerError: Error, Sendable {
    case modelInferenceFailed(underlying: Error)
    case cacheAllocationFailed
    case tokenizationFailed
    case requestCancelled
    case shutdownInProgress
}

/// Handle errors in solo generation
private func runSoloGenerationSafe(_ request: InferenceRequest) async {
    do {
        try await runSoloGeneration(request)
    } catch {
        // Log error
        print("Solo generation failed for request \(request.id): \(error)")
        
        // Emit error event (optional)
        if let activeReq = activeRequests[request.id] {
            // Could yield an error event here if TokenEvent supported it
            activeReq.continuation.finish()
        }
        
        // Clean up
        activeRequests.removeValue(forKey: request.id)
        requestStats.removeValue(forKey: request.id)
        
        // Try to recover by admitting next request
        await tryAdmitFromQueue()
    }
}

/// Handle errors in batch generation
private func runBatchedGenerationSafe() async {
    do {
        try await runBatchedGeneration()
    } catch {
        // Log error
        print("Batch generation failed: \(error)")
        
        // Finish all active continuations
        for (requestID, activeReq) in activeRequests {
            activeReq.continuation.finish()
        }
        
        // Clean up
        activeRequests.removeAll()
        
        // Try to recover
        await tryAdmitFromQueue()
    }
}
```

---

## Missing Implementation Details

### Stop Token Handling

```swift
/// Build unified stop token set from model context
private func buildStopTokens() -> Set<Int> {
    var stopTokens = Set<Int>()
    
    // Add EOS token
    if let eos = context.tokenizer.eosTokenId {
        stopTokens.insert(eos)
    }
    
    // Add extra EOS tokens from configuration
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            stopTokens.insert(id)
        }
    }
    
    return stopTokens
}

/// Note: Per-request stop tokens are not supported in this design.
/// All requests share the same model-level stop tokens.
/// To support per-request stop tokens, check after sampling in the client.
```

### LogProbs Handling

```swift
/// LogProbs handling strategy:
/// - When returnLogProbs = true, BatchTokenIterator returns full vocab logits
/// - We store the full MLXArray in TokenEvent.logProbs
/// - Client can process as needed (e.g., compute top-k)
/// 
/// For efficiency, consider:
/// - Only returning top-k logprobs per token
/// - Normalizing to probabilities (not log-probs)
/// - Storing as [Int: Float] dictionary instead of MLXArray
///
/// Current implementation returns raw logits array for maximum flexibility.
```

### Text Delta Generation

```swift
/// Convert token to text using tokenizer
private func tokenToText(_ token: Int) -> String {
    context.tokenizer.decode(tokens: [token])
}

/// Note: Some tokenizers may produce empty strings for special tokens.
/// Consider filtering or handling specially:
///
/// private func tokenToText(_ token: Int) -> String? {
///     let text = context.tokenizer.decode(tokens: [token])
///     return text.isEmpty ? nil : text
/// }
```

### Queue Admission Logic

```swift
/// Admit requests from queue when capacity available
private func tryAdmitFromQueue() async {
    while activeRequests.count < config.maxConcurrentRequests {
        guard let request = queuedRequests.first else { break }
        queuedRequests.removeFirst()
        
        // Add to active (details omitted)
        // ...
        
        // Update mode if needed
        await updateMode()
    }
}
```

### Shutdown Implementation

```swift
public func shutdown() async {
    isShuttingDown = true
    
    // 1. Finish all queued requests
    for request in queuedRequests {
        if let continuation = continuations[request.id] {
            continuation.finish()
        }
    }
    queuedRequests.removeAll()
    
    // 2. Cancel current mode
    switch currentMode {
    case .solo(_, let task), .batch(let task):
        task.cancel()
    case .idle:
        break
    }
    
    // 3. Finish all active requests
    for (requestID, activeReq) in activeRequests {
        activeReq.continuation.finish()
    }
    activeRequests.removeAll()
    
    // 4. Clean up
    continuations.removeAll()
    requestStats.removeAll()
    cancelledUIDs.removeAll()
    
    currentMode = .idle
}
```

---

## Complete InferenceScheduler Skeleton

```swift
public actor InferenceScheduler {
    // MARK: - Public Configuration
    
    private let context: ModelContext
    private let config: SchedulerConfig
    
    // MARK: - State
    
    private var currentMode: Mode = .idle
    private var isShuttingDown = false
    
    // Request tracking
    private var activeRequests: [UUID: ActiveRequest] = [:]
    private var queuedRequests: [InferenceRequest] = []
    private var continuations: [UUID: AsyncStream<TokenEvent>.Continuation] = [:]
    
    // Batch mode tracking
    private var uidToRequestID: UIDMapping = [:]
    private var cancelledUIDs: Set<UUID> = []
    
    // Statistics
    private var requestStats: [UUID: RequestStats] = [:]
    private var totalRequests = 0
    private var completedRequests = 0
    private var totalTokensGenerated = 0
    private var totalGenerationTime: TimeInterval = 0
    
    // MARK: - Initialization
    
    public init(context: ModelContext, config: SchedulerConfig = .init()) {
        self.context = context
        self.config = config
    }
    
    // MARK: - Public API
    
    public func submit(_ request: InferenceRequest) -> AsyncStream<TokenEvent> {
        AsyncStream { continuation in
            Task {
                await self.handleSubmit(request, continuation: continuation)
            }
        }
    }
    
    public func stats(for requestID: UUID) -> RequestStats? {
        requestStats[requestID]
    }
    
    public func aggregatedStats() -> AggregatedStats {
        AggregatedStats(
            activeRequests: activeRequests.count,
            queuedRequests: queuedRequests.count,
            totalRequestsProcessed: completedRequests,
            averageTPS: totalGenerationTime > 0
                ? Double(totalTokensGenerated) / totalGenerationTime
                : 0,
            peakMemoryGB: Double(GPU.peakMemory) / 1_000_000_000.0
        )
    }
    
    public func shutdown() async {
        // Implementation above
    }
    
    // MARK: - Private Methods
    
    private func handleSubmit(
        _ request: InferenceRequest,
        continuation: AsyncStream<TokenEvent>.Continuation
    ) async {
        guard !isShuttingDown else {
            continuation.finish()
            return
        }
        
        // Set up cancellation
        continuation.onTermination = { @Sendable _ in
            Task {
                await self.handleCancellation(request.id)
            }
        }
        
        // Track continuation
        continuations[request.id] = continuation
        
        // Initialize stats
        requestStats[request.id] = RequestStats(
            requestID: request.id,
            promptTokens: request.tokens.count,
            generatedTokens: 0,
            promptTime: 0,
            generationTime: 0
        )
        
        // Check capacity
        if activeRequests.count < config.maxConcurrentRequests {
            // Create active request
            let activeReq = ActiveRequest(
                request: request,
                continuation: continuation,
                stats: requestStats[request.id]!,
                uid: nil,
                startTime: Date()
            )
            activeRequests[request.id] = activeReq
            totalRequests += 1
            
            // Update mode
            await updateMode()
        } else {
            // Queue the request
            queuedRequests.append(request)
        }
    }
    
    private func handleCancellation(_ requestID: UUID) async {
        cancelledUIDs.insert(requestID)
        continuations.removeValue(forKey: requestID)
        // Actual removal happens in next tick() or at completion
    }
    
    private func updateStats(
        for requestID: UUID,
        promptTokens: Int? = nil,
        generatedTokens: Int? = nil,
        promptTime: TimeInterval? = nil,
        generationTime: TimeInterval? = nil
    ) {
        guard var stats = requestStats[requestID] else { return }
        
        if let pt = promptTokens { stats.promptTokens = pt }
        if let gt = generatedTokens { 
            stats.generatedTokens += gt 
            totalTokensGenerated += gt
        }
        if let pt = promptTime { stats.promptTime = pt }
        if let gt = generationTime { 
            stats.generationTime = gt
            totalGenerationTime += gt
        }
        
        requestStats[requestID] = stats
    }
    
    // Mode transition and generation methods defined above
    private func updateMode() async { /* See above */ }
    private func runSoloGeneration(_ request: InferenceRequest) async { /* See above */ }
    private func runBatchedGeneration() async { /* See above */ }
    private func tryAdmitFromQueue() async { /* See above */ }
}
```

---

## Example Usage

```swift
import MLXLMCommon
import MLXLLM

// 1. Load model
let modelContext = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")

// 2. Create scheduler
let config = SchedulerConfig(
    maxConcurrentRequests: 16,
    completionBatchSize: 8,
    returnLogProbs: false
)
let scheduler = InferenceScheduler(context: modelContext, config: config)

// 3. Submit requests
let request1 = InferenceRequest(
    id: UUID(),
    tokens: modelContext.tokenizer.encode(text: "What is the capital of France?"),
    params: GenerateParameters(temperature: 0.7, topP: 0.9),
    maxTokens: 100,
    createdAt: Date()
)

let request2 = InferenceRequest(
    id: UUID(),
    tokens: modelContext.tokenizer.encode(text: "Explain quantum computing"),
    params: GenerateParameters(temperature: 0.5, topP: 0.95),
    maxTokens: 200,
    createdAt: Date()
)

// 4. Process streams concurrently
await withTaskGroup(of: Void.self) { group in
    // First request
    group.addTask {
        var fullText = ""
        for await event in await scheduler.submit(request1) {
            if let delta = event.textDelta {
                fullText += delta
                print("Request 1: \(delta)", terminator: "")
            }
            if event.finishReason != nil {
                print("\nRequest 1 complete: \(fullText)")
                
                // Get stats
                if let stats = await scheduler.stats(for: request1.id) {
                    print("Request 1 stats: \(stats.tokensPerSecond) tok/s")
                }
            }
        }
    }
    
    // Second request
    group.addTask {
        var fullText = ""
        for await event in await scheduler.submit(request2) {
            if let delta = event.textDelta {
                fullText += delta
                print("Request 2: \(delta)", terminator: "")
            }
            if event.finishReason != nil {
                print("\nRequest 2 complete: \(fullText)")
                
                // Get stats
                if let stats = await scheduler.stats(for: request2.id) {
                    print("Request 2 stats: \(stats.tokensPerSecond) tok/s")
                }
            }
        }
    }
}

// 5. Get aggregated stats
let aggStats = await scheduler.aggregatedStats()
print("Total requests: \(aggStats.totalRequestsProcessed)")
print("Average TPS: \(aggStats.averageTPS)")

// 6. Shutdown
await scheduler.shutdown()
```

 ──────────────────────────────────────────

 File Structure

     ├── InferenceScheduler.swift         # ~200 lines: main actor
     ├── Types.swift                       # ~50 lines: public types
     └── BatchTokenIterator+PerUID.swift   # ~50 lines: extend for per-UID samplers

   Tests/MLXSchedulerTests/
     ├── SoloModeTests.swift              # Test solo path
     ├── BatchModeTests.swift             # Test batched path
     ├── CancellationTests.swift          # Test stream cancellation
     └── StatsTests.swift                 # Test per-request & aggregated stats


 Total new code: ~300 lines

 ──────────────────────────────────────────

 Implementation Plan

 Week 1: Core Scheduler
 [ ] Create InferenceScheduler actor with queue
 [ ] Implement solo mode using existing generate()
 [ ] Add concurrency limiting (maxConcurrentRequests)
 [ ] Test with 1 request at a time

 Week 2: Per-UID Samplers
 [ ] Modify BatchTokenIterator.step() for per-UID sampling
 [ ] Add samplers parameter to insert()
 [ ] Test different temps/top-p per request

 Week 3: Batch Mode
 [ ] Implement batch mode path with BatchTokenIterator
 [ ] Add switching logic (solo when count==1, batch when count>=2)
 [ ] Test with 2-16 concurrent requests

 Week 4: Stats & Cancellation
 [ ] Implement per-request stats tracking
 [ ] Implement aggregated stats
 [ ] Add cancellation support (remove from batch on stream termination)
 [ ] Performance benchmarks (solo vs batch TPS)

 ──────────────────────────────────────────

 Testing Strategy

 Unit Tests
 1. Solo Mode Correctness: Verify tokens match existing generate() output
 2. Batch Mode Correctness: Verify tokens match existing batchGenerate() output
 3. Per-Request Params: Submit 2 requests with temp=0.0 and temp=1.0, verify different outputs
 4. Cancellation: Cancel mid-generation, verify removed from batch and stats
 5. Stats Accuracy: Verify per-request and aggregated stats match actual token counts

 Performance Tests
 1. Solo TPS: Should match existing generate() (no regression)
 2. Batch TPS: Should match existing batchGenerate() (no regression)
 3. Switching Overhead: Measure cost of switching solo ↔ batch

 Integration Tests
 1. Queue Management: Submit 100 requests with maxConcurrent=16, verify queueing
 2. Mixed Workload: Random request arrivals, cancellations, and completions
 3. Memory Stability: Long-running test with continuous requests

 ──────────────────────────────────────────

 Design Decisions (Q&A)

 Q: Should mid-stream parameter changes be supported?
 A: No. Parameters are set by client when submitting the request and don't change during generation.


 Q: Should stats be per-request or aggregated?
 A: Both. Per-request stats for client reporting, aggregated stats for server logging/monitoring.

 Q: How to handle cancelled streams?
 A: Remove from batch immediately on stream termination. Use filter() on ActiveBatch to remove
 cancelled UIDs.

 ──────────────────────────────────────────

 Success Criteria

 1. ✅ Solo mode TPS ≥ existing generate() (no regression)
 2. ✅ Batch mode TPS ≥ existing batchGenerate() (no regression)
 3. ✅ Per-request parameters work (different temp/top-p per stream)
 4. ✅ Cancellation removes request from batch immediately
 5. ✅ Per-request stats available for all active/completed requests
 6. ✅ Aggregated stats track overall scheduler health
 7. ✅ Total new code ≤ 500 lines (excluding tests)
 8. ✅ Swift 6 concurrency-safe (no warnings)

 ──────────────────────────────────────────

 Future Enhancements (V2+)

 •  Multi-model support (run Qwen + Llama simultaneously)
 •  Prefill caching (reuse prompt prefixes)
 •  Speculative decoding (draft model + verification)
 •  Priority scheduling (low-latency vs high-throughput)
 •  Distributed prefill (CPU prefill, GPU decode)

 ──────────────────────────────────────────

 End of document.