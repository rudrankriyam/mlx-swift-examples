// Copyright © 2024 Apple Inc.
//
// Batched text generation inspired by the Python implementation at
// `mlx_lm/generate.py` (BatchGenerator, Batch, BatchStats, batch_generate).

import Foundation
import MLX
import Tokenizers

/// Statistics gathered for batched generation.
public struct BatchGenerateStats: Sendable {

    public var promptTokenCount: Int
    public var promptTime: TimeInterval
    public var generationTokenCount: Int
    public var generationTime: TimeInterval
    public var peakMemoryGB: Double

    public init(
        promptTokenCount: Int = 0,
        promptTime: TimeInterval = 0,
        generationTokenCount: Int = 0,
        generationTime: TimeInterval = 0,
        peakMemoryGB: Double = 0
    ) {
        self.promptTokenCount = promptTokenCount
        self.promptTime = promptTime
        self.generationTokenCount = generationTokenCount
        self.generationTime = generationTime
        self.peakMemoryGB = peakMemoryGB
    }

    public var promptTokensPerSecond: Double {
        guard promptTime > 0 else { return 0 }
        return Double(promptTokenCount) / promptTime
    }

    public var generationTokensPerSecond: Double {
        guard generationTime > 0 else { return 0 }
        return Double(generationTokenCount) / generationTime
    }
}

/// Configuration parameters for batched generation.
public struct BatchGenerateParameters: Sendable {

    /// Default maximum tokens to generate per prompt.
    public var defaultMaxTokens: Int

    /// Maximum number of completions to keep active simultaneously.
    public var completionBatchSize: Int

    /// Number of prompts to prefill at a time.
    public var prefillBatchSize: Int

    /// Number of prompt tokens to process per prefill step.
    public var prefillStepSize: Int

    /// Underlying single-sequence generation parameters (sampling, quantization, etc.).
    public var generation: GenerateParameters

    public init(
        maxTokens: Int = 128,
        completionBatchSize: Int = 32,
        prefillBatchSize: Int = 8,
        prefillStepSize: Int = 2_048,
        generation: GenerateParameters = .init()
    ) {
        self.defaultMaxTokens = maxTokens
        self.completionBatchSize = completionBatchSize
        self.prefillBatchSize = prefillBatchSize
        self.prefillStepSize = prefillStepSize

        var generationParameters = generation
        if generationParameters.maxTokens == nil {
            generationParameters.maxTokens = maxTokens
        }
        self.generation = generationParameters
    }
}

/// Result payload returned from `batchGenerate`.
public struct BatchGenerateResult: Sendable {

    public struct Sequence: Sendable {
        public let uid: Int
        public let tokens: [Int]
        public let text: String
        public let finishReason: FinishReason
        public let logProbs: MLXArray?

        public init(
            uid: Int,
            tokens: [Int],
            text: String,
            finishReason: FinishReason,
            logProbs: MLXArray? = nil
        ) {
            self.uid = uid
            self.tokens = tokens
            self.text = text
            self.finishReason = finishReason
            self.logProbs = logProbs
        }
    }

    public enum FinishReason: String, Sendable {
        case stop
        case length
        case unspecified
    }

    public let sequences: [Sequence]
    public let stats: BatchGenerateStats

    public init(sequences: [Sequence], stats: BatchGenerateStats) {
        self.sequences = sequences
        self.stats = stats
    }
}

fileprivate struct PendingPrompt {
    let uid: Int
    let tokens: [Int]
    let maxTokens: Int

    var length: Int { tokens.count }
}

fileprivate struct ActiveBatch {
    var uids: [Int]
    var tokens: MLXArray
    var logProbs: MLXArray
    var maxTokens: [Int]
    var numTokens: [Int]
    var cache: [KVCache]

    var count: Int { uids.count }

    mutating func filter(keeping indices: [Int]) {
        let keepIndices = MLXArray(indices.map(Int32.init))
        uids = indices.map { uids[$0] }
        maxTokens = indices.map { maxTokens[$0] }
        numTokens = indices.map { numTokens[$0] }

        tokens = tokens[keepIndices]
        logProbs = logProbs[keepIndices, .ellipsis]

        for i in cache.indices {
            switch cache[i] {
            case let batch as BatchKVCache:
                batch.filter(batchIndices: keepIndices)
            case let arrays as ArraysCache:
                arrays.filter(batchIndices: keepIndices)
            case let rotating as BatchRotatingKVCache:
                rotating.filter(batchIndices: keepIndices)
            case let list as CacheList:
                list.filter(batchIndices: keepIndices)
            default:
                fatalError("\(type(of: cache[i])) does not support batched filtering")
            }
        }
    }

    mutating func extend(_ other: ActiveBatch) {
        uids.append(contentsOf: other.uids)
        maxTokens.append(contentsOf: other.maxTokens)
        numTokens.append(contentsOf: other.numTokens)

        tokens = MLX.concatenated([tokens, other.tokens])
        logProbs = MLX.concatenated([logProbs, other.logProbs])

        for i in cache.indices {
            switch (cache[i], other.cache[i]) {
            case (let lhs as BatchKVCache, let rhs as BatchKVCache):
                lhs.extend(other: rhs)
            case (let lhs as ArraysCache, let rhs as ArraysCache):
                lhs.extend(other: rhs)
            case (let lhs as BatchRotatingKVCache, let rhs as BatchRotatingKVCache):
                lhs.extend(other: rhs)
            case (let lhs as CacheList, let rhs as CacheList):
                lhs.extend(other: rhs)
            default:
                fatalError("\(type(of: cache[i])) does not support batched extension")
            }
        }
    }
}

/// Iterator that mirrors the lifecycle of Python's `BatchGenerator`.
public struct BatchTokenIterator: Sequence, IteratorProtocol {

    public struct Response: Sendable {
        public let uid: Int
        public let token: Int
        public let logProbs: MLXArray
        public let finishReason: BatchGenerateResult.FinishReason?

        public init(
            uid: Int,
            token: Int,
            logProbs: MLXArray,
            finishReason: BatchGenerateResult.FinishReason?
        ) {
            self.uid = uid
            self.token = token
            self.logProbs = logProbs
            self.finishReason = finishReason
        }
    }

    private var model: any LanguageModel
    private let parameters: BatchGenerateParameters
    private let sampler: LogitSampler
    private let kvBits: Int?
    private let kvGroupSize: Int
    private let quantizedKVStart: Int
    private let stopTokens: Set<Int>
    private let unknownTokenId: Int?

    private var unprocessedPrompts: [PendingPrompt] = []
    private var activeBatch: ActiveBatch?
    private var uidCounter: Int = 0

    private var promptTokenCount = 0
    private var promptTime: TimeInterval = 0
    private var generationTokenCount = 0
    private var generationTime: TimeInterval = 0

    public init(
        model: any LanguageModel,
        parameters: BatchGenerateParameters,
        stopTokens: Set<Int>,
        unknownTokenId: Int?
    ) {
        self.model = model
        self.parameters = parameters
        self.sampler = parameters.generation.sampler()
        self.kvBits = parameters.generation.kvBits
        self.kvGroupSize = parameters.generation.kvGroupSize
        self.quantizedKVStart = parameters.generation.quantizedKVStart
        self.stopTokens = stopTokens
        self.unknownTokenId = unknownTokenId
    }

    @discardableResult
    public mutating func insert(prompts: [[Int]], maxTokens: [Int]? = nil) -> [Int] {
        guard maxTokens == nil || maxTokens!.count == prompts.count else {
            fatalError("maxTokens.count must match prompts.count")
        }

        let resolvedMaxTokens: [Int] = maxTokens ?? Array(
            repeating: parameters.defaultMaxTokens, count: prompts.count)

        var assigned = [Int]()
        for (tokens, limit) in zip(prompts, resolvedMaxTokens) {
            let uid = uidCounter
            uidCounter += 1
            unprocessedPrompts.append(.init(uid: uid, tokens: tokens, maxTokens: limit))
            assigned.append(uid)
        }

        unprocessedPrompts.sort { $0.length < $1.length }
        return assigned
    }

    public mutating func next() -> [Response]? {
        if activeBatch == nil && unprocessedPrompts.isEmpty {
            return nil
        }

        var batch = activeBatch
        var promptProcessing = false
        var tic = Date.timeIntervalSinceReferenceDate
        let prefillBatchSize = Swift.max(parameters.prefillBatchSize, 1)

        var numActive = batch?.count ?? 0
        var numToAdd = Swift.max(parameters.completionBatchSize - numActive, 0)

        while numToAdd >= prefillBatchSize {
            let chunk = Array(unprocessedPrompts.prefix(prefillBatchSize))

            if chunk.isEmpty {
                if numActive > 0 {
                    break
                } else {
                    activeBatch = nil
                    return []
                }
            }

            if var existing = batch, !promptProcessing {
                eval(existing.tokens, existing.logProbs)
                let now = Date.timeIntervalSinceReferenceDate
                generationTime += now - tic
                tic = now
            }

            let newBatch = processPrompts(chunk)
            unprocessedPrompts.removeFirst(chunk.count)
            promptProcessing = true

            if var current = batch {
                current.extend(newBatch)
                batch = current
            } else {
                batch = newBatch
            }

            numActive = batch?.count ?? 0
            numToAdd = Swift.max(parameters.completionBatchSize - numActive, 0)
        }

        guard var currentBatch = batch ?? activeBatch else {
            return []
        }

        if promptProcessing {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime += now - tic
            tic = now
        }

        // Store references to previous iteration's results
        let previousTokens = currentBatch.tokens
        let previousLogProbs = currentBatch.logProbs

        // Compute next iteration's tokens asynchronously
        var cache = currentBatch.cache
        let (nextTokens, nextLogProbs) = step(
            inputTokens: previousTokens[0..., .newAxis], cache: &cache)
        currentBatch.cache = cache
        asyncEval(nextTokens, nextLogProbs)

        // Synchronize only when we need to read the results
        // This allows GPU to work on next iteration while CPU processes current results
        let tokenArray = previousTokens.asArray(Int.self)


        var responses: [Response] = []
        var keepIndices: [Int] = []
        var endIndices: [Int] = []

        for idx in 0 ..< tokenArray.count {
            var finish: BatchGenerateResult.FinishReason? = nil
            let token = tokenArray[idx]

            currentBatch.numTokens[idx] += 1

            if stopTokens.contains(token) {
                finish = .stop
                endIndices.append(idx)
            } else if currentBatch.numTokens[idx] >= currentBatch.maxTokens[idx] {
                finish = .length
                endIndices.append(idx)
            } else {
                keepIndices.append(idx)
            }

            let logProbRow = previousLogProbs[idx]
            responses.append(
                Response(
                    uid: currentBatch.uids[idx],
                    token: token,
                    logProbs: logProbRow,
                    finishReason: finish))
        }

        currentBatch.tokens = nextTokens
        currentBatch.logProbs = nextLogProbs

        generationTokenCount += responses.count
        let toc = Date.timeIntervalSinceReferenceDate
        generationTime += toc - tic

        if !endIndices.isEmpty {
            if keepIndices.isEmpty {
                activeBatch = nil
            } else {
                currentBatch.filter(keeping: keepIndices)
                activeBatch = currentBatch
            }
        } else {
            activeBatch = currentBatch
        }

        return responses
    }

    public func stats() -> BatchGenerateStats {
        BatchGenerateStats(
            promptTokenCount: promptTokenCount,
            promptTime: promptTime,
            generationTokenCount: generationTokenCount,
            generationTime: generationTime,
            peakMemoryGB: Double(GPU.peakMemory) / 1_000_000_000.0
        )
    }

    private mutating func processPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        let tokens = prompts.map(\.tokens)
        let lengths = tokens.map(\.count)

        promptTokenCount += lengths.reduce(0, +)

        let (padded, leftPadding) = leftPadPrompts(tokens)

        var cache = makeBatchCache(
            model: model, parameters: parameters, leftPadding: leftPadding)

        var remaining = padded
        while remaining.dim(1) > 1 {
            let nToProcess = Swift.min(parameters.prefillStepSize, remaining.dim(1) - 1)
            let slice = remaining[0..., ..<nToProcess]
            _ = model(slice, cache: cache)
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart)
            let states = cache.flatMap(\.state)
            if !states.isEmpty {
                eval(states)
            }
            let length = remaining.dim(1)
            remaining = remaining[0..., nToProcess..<length]
            GPU.clearCache()
        }

        let (y, logProbs) = step(inputTokens: remaining, cache: &cache)
        asyncEval(y, logProbs)

        return ActiveBatch(
            uids: prompts.map(\.uid),
            tokens: y,
            logProbs: logProbs,
            maxTokens: prompts.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: prompts.count),
            cache: cache)
    }

    private mutating func step(
        inputTokens: MLXArray,
        cache: inout [KVCache]
    ) -> (MLXArray, MLXArray) {
        let logits = model(inputTokens, cache: cache)
        var selected = logits[0..., -1, 0...]

        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart)

        let logSum = logSumExp(selected, axis: selected.ndim - 1, keepDims: true)
        let logProbs = selected - logSum

        let sampled = sampler.sample(logits: selected)

        return (sampled, logProbs)
    }
}

/// High-level API mirroring the Python `batch_generate` helper.
@discardableResult
public func batchGenerate(
    prompts: [String],
    maxTokens: [Int]? = nil,
    parameters: BatchGenerateParameters,
    context: ModelContext
) throws -> BatchGenerateResult {
    let tokenized: [[Int]] = prompts.map { context.tokenizer.encode(text: $0) }

    let defaultMaxTokens = maxTokens ?? Array(
        repeating: parameters.defaultMaxTokens, count: tokenized.count)
    guard defaultMaxTokens.count == tokenized.count else {
        fatalError("maxTokens.count must match prompts.count")
    }

    var stopTokens = Set<Int>()
    if let eos = context.tokenizer.eosTokenId {
        stopTokens.insert(eos)
    }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            stopTokens.insert(id)
        }
    }

    var iterator = BatchTokenIterator(
        model: context.model,
        parameters: parameters,
        stopTokens: stopTokens,
        unknownTokenId: context.tokenizer.unknownTokenId)

    let uids = iterator.insert(prompts: tokenized, maxTokens: defaultMaxTokens)

    var generatedTokens = [Int: [Int]]()
    var logProbHistory = [Int: [MLXArray]]()
    var finishReasons = [Int: BatchGenerateResult.FinishReason]()

    while let responses = iterator.next(), !responses.isEmpty {
        for response in responses {
            if response.finishReason != .stop {
                generatedTokens[response.uid, default: []].append(response.token)
            }
            response.logProbs.eval()
            logProbHistory[response.uid, default: []].append(response.logProbs)

            if let finish = response.finishReason {
                finishReasons[response.uid] = finish
            }
        }
    }

    Stream().synchronize()

    let stats = iterator.stats()
    var sequences: [BatchGenerateResult.Sequence] = []
    sequences.reserveCapacity(uids.count)

    for uid in uids {
        let tokens = generatedTokens[uid] ?? []
        let text = context.tokenizer.decode(tokens: tokens)
        let finish = finishReasons[uid] ?? .unspecified
        let logProbRows = logProbHistory[uid] ?? []
        let logProbs = logProbRows.isEmpty ? nil : MLX.stacked(logProbRows, axis: 0)

        sequences.append(
            .init(uid: uid, tokens: tokens, text: text, finishReason: finish, logProbs: logProbs))
    }

    return BatchGenerateResult(sequences: sequences, stats: stats)
}

// MARK: - Helpers

private func leftPadPrompts(_ prompts: [[Int]]) -> (MLXArray, [Int]) {
    guard let maxLength = prompts.map(\.count).max() else {
        return (MLXArray.zeros([0, 0], type: Int32.self), [])
    }

    var padded = [Int]()
    padded.reserveCapacity(prompts.count * maxLength)

    var leftPadding = [Int]()
    leftPadding.reserveCapacity(prompts.count)

    for prompt in prompts {
        let pad = maxLength - prompt.count
        leftPadding.append(pad)
        padded.append(contentsOf: Array(repeating: 0, count: pad))
        padded.append(contentsOf: prompt)
    }

    let array = MLXArray(padded, [prompts.count, maxLength])
    return (array, leftPadding)
}

private func makeBatchCache(
    model: any LanguageModel,
    parameters: BatchGenerateParameters,
    leftPadding: [Int]
) -> [KVCache] {
    let caches = model.newCache(parameters: parameters.generation)
    return caches.map {
        convertCache($0, parameters: parameters, leftPadding: leftPadding)
    }
}

private func convertCache(
    _ cache: KVCache,
    parameters: BatchGenerateParameters,
    leftPadding: [Int]
) -> KVCache {
    switch cache {
    case is BatchKVCache:
        return cache
    case is KVCacheSimple:
        return BatchKVCache(leftPadding: leftPadding)
    case let rotating as RotatingKVCache:
        guard rotating.keepTokens == 0 else {
            fatalError("RotatingKVCache with keep tokens is not supported in batched generation")
        }
        guard let window = rotating.maxSize else {
            fatalError("RotatingKVCache expected to have a max size for batching")
        }
        return BatchRotatingKVCache(maxSize: window, leftPadding: leftPadding)
    case let arrays as ArraysCache:
        arrays.setLeftPadding(leftPadding)
        return arrays
    case let list as CacheList:
        let converted = (0 ..< list.count).map {
            convertCache(list[$0], parameters: parameters, leftPadding: leftPadding)
        }
        return CacheList(converted)
    default:
        fatalError("\(type(of: cache)) does not yet support batching")
    }
}
