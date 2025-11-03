import Foundation
import MLX
import Tokenizers

public actor InferenceScheduler {
    private struct ActiveRequest {
        var request: InferenceRequest
        var continuation: AsyncStream<TokenEvent>.Continuation
        var uid: Int?
        var startTime: TimeInterval
        var firstTokenAt: TimeInterval?
        var lastEmitAt: TimeInterval?
    }

    private enum Mode {
        case idle
        case solo(requestID: UUID, task: Task<Void, Never>)
        case batch(task: Task<Void, Never>)
    }

    private let context: ModelContext
    private let config: SchedulerConfig

    private var currentMode: Mode = .idle
    private var isShuttingDown = false

    private var activeRequests: [UUID: ActiveRequest] = [:]
    private var queuedRequests: [InferenceRequest] = []
    private var continuations: [UUID: AsyncStream<TokenEvent>.Continuation] = [:]
    private var requestStats: [UUID: RequestStats] = [:]

    private var uidToRequestID: [Int: UUID] = [:]
    private var pendingBatchRemovals: Set<Int> = []
    private var cancelledRequests: Set<UUID> = []

    private var totalRequests = 0
    private var completedRequests = 0
    private var totalTokensGenerated = 0
    private var totalGenerationTime: TimeInterval = 0

    public init(context: ModelContext, config: SchedulerConfig = .init()) {
        self.context = context
        self.config = config
    }

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
        let averageTPS = totalGenerationTime > 0
            ? Double(totalTokensGenerated) / totalGenerationTime
            : 0
        return AggregatedStats(
            activeRequests: activeRequests.count,
            queuedRequests: queuedRequests.count,
            totalRequestsProcessed: completedRequests,
            averageTPS: averageTPS,
            peakMemoryGB: Double(GPU.peakMemory) / 1_000_000_000.0
        )
    }

    public func shutdown() async {
        guard !isShuttingDown else { return }
        isShuttingDown = true

        switch currentMode {
        case .solo(_, let task), .batch(let task):
            task.cancel()
        case .idle:
            break
        }
        currentMode = .idle

        for request in queuedRequests {
            continuations[request.id]?.finish()
            continuations.removeValue(forKey: request.id)
            requestStats.removeValue(forKey: request.id)
        }
        queuedRequests.removeAll()

        for (requestID, active) in activeRequests {
            active.continuation.finish()
            continuations.removeValue(forKey: requestID)
            requestStats.removeValue(forKey: requestID)
        }
        activeRequests.removeAll()

        uidToRequestID.removeAll()
        pendingBatchRemovals.removeAll()
        cancelledRequests.removeAll()
    }

    private func handleSubmit(
        _ request: InferenceRequest,
        continuation: AsyncStream<TokenEvent>.Continuation
    ) async {
        guard !isShuttingDown else {
            continuation.finish()
            return
        }

        continuation.onTermination = { termination in
            if case .cancelled = termination {
                Task { await self.handleCancellation(request.id) }
            }
        }

        continuations[request.id] = continuation
        requestStats[request.id] = RequestStats(
            requestID: request.id,
            promptTokens: request.tokens.count
        )

        if activeRequests.count < config.maxConcurrentRequests {
            let active = ActiveRequest(
                request: request,
                continuation: continuation,
                uid: nil,
                startTime: Date.timeIntervalSinceReferenceDate,
                firstTokenAt: nil,
                lastEmitAt: nil
            )
            activeRequests[request.id] = active
            totalRequests += 1
            await updateMode()
        } else {
            queuedRequests.append(request)
        }
    }

    private func handleCancellation(_ requestID: UUID) async {
        if let index = queuedRequests.firstIndex(where: { $0.id == requestID }) {
            queuedRequests.remove(at: index)
            continuations[requestID]?.finish()
            continuations.removeValue(forKey: requestID)
            requestStats.removeValue(forKey: requestID)
            return
        }

        guard var active = activeRequests[requestID] else {
            continuations[requestID]?.finish()
            continuations.removeValue(forKey: requestID)
            requestStats.removeValue(forKey: requestID)
            return
        }

        cancelledRequests.insert(requestID)
        if let uid = active.uid {
            pendingBatchRemovals.insert(uid)
        }

        active.continuation.finish()
        activeRequests[requestID] = active
        continuations.removeValue(forKey: requestID)

        await tryAdmitFromQueue()
    }

    private func tryAdmitFromQueue() async {
        while activeRequests.count < config.maxConcurrentRequests,
            let next = queuedRequests.first
        {
            queuedRequests.removeFirst()
            guard let continuation = continuations[next.id] else { continue }
            let stats = requestStats[next.id]
                ?? RequestStats(requestID: next.id, promptTokens: next.tokens.count)
            requestStats[next.id] = stats

            let active = ActiveRequest(
                request: next,
                continuation: continuation,
                uid: nil,
                startTime: Date.timeIntervalSinceReferenceDate,
                firstTokenAt: nil,
                lastEmitAt: nil
            )
            activeRequests[next.id] = active
            totalRequests += 1
        }

        await updateMode()
    }

    private func updateMode() async {
        if isShuttingDown { return }

        if activeRequests.isEmpty {
            switch currentMode {
            case .solo(_, let task), .batch(let task):
                task.cancel()
            case .idle:
                break
            }
            currentMode = .idle
            return
        }

        switch currentMode {
        case .idle:
            if activeRequests.count == 1, let requestID = activeRequests.keys.first {
                let task = Task { await self.runSoloGeneration(for: requestID) }
                currentMode = .solo(requestID: requestID, task: task)
            } else {
                let task = Task { await self.runBatchedGeneration() }
                currentMode = .batch(task: task)
            }
        case .solo(let requestID, let task):
            if activeRequests[requestID] == nil {
                task.cancel()
                currentMode = .idle
                await updateMode()
            }
        case .batch:
            break
        }
    }

    private func runSoloGeneration(for requestID: UUID) async {
        guard var entry = activeRequests[requestID] else { return }

        let stopTokens = buildStopTokens()
        let tokens = MLXArray(entry.request.tokens)
        let lmInput = LMInput(tokens: tokens)

        do {
            var iterator = try TokenIterator(
                input: lmInput,
                model: context.model,
                cache: context.model.newCache(parameters: entry.request.params),
                parameters: entry.request.params
            )

            var finishReason: BatchGenerateResult.FinishReason = .unspecified
            var emitCount = 0
            var lastTimestamp = Date.timeIntervalSinceReferenceDate

            while let token = iterator.next() {
                if cancelledRequests.contains(requestID) {
                    finishReason = .unspecified
                    break
                }

                if stopTokens.contains(token) {
                    finishReason = .stop
                    break
                }

                let now = Date.timeIntervalSinceReferenceDate
                if entry.firstTokenAt == nil {
                    let promptTime = now - entry.startTime
                    entry.firstTokenAt = now
                    updateStats(for: requestID, promptTime: promptTime)
                } else {
                    let delta = now - lastTimestamp
                    updateStats(for: requestID, generationDelta: delta)
                }
                lastTimestamp = now

                emitCount += 1
                updateStats(for: requestID, incrementGeneratedTokens: 1)

                entry.lastEmitAt = now
                activeRequests[requestID] = entry

                let text = context.tokenizer.decode(tokens: [token])
                let event = TokenEvent(
                    requestID: requestID,
                    token: token,
                    textDelta: text.isEmpty ? nil : text,
                    finishReason: nil,
                    logProbs: nil
                )
                entry.continuation.yield(event)

                await Task.yield()

                if emitCount >= entry.request.maxTokens {
                    finishReason = .length
                    break
                }
            }

            let completionEvent = TokenEvent(
                requestID: requestID,
                token: nil,
                textDelta: nil,
                finishReason: finishReason,
                logProbs: nil
            )
            entry.continuation.yield(completionEvent)
            entry.continuation.finish()

        } catch {
            entry.continuation.finish()
            requestStats.removeValue(forKey: requestID)
        }

        activeRequests.removeValue(forKey: requestID)
        cancelledRequests.remove(requestID)
        continuations.removeValue(forKey: requestID)
        completedRequests += 1

        currentMode = .idle
        await tryAdmitFromQueue()
    }

    private func runBatchedGeneration() async {
        var iterator = BatchTokenIterator(
            model: context.model,
            parameters: BatchGenerateParameters(
                maxTokens: config.completionBatchSize,
                completionBatchSize: config.completionBatchSize,
                prefillBatchSize: config.prefillBatchSize,
                prefillStepSize: config.prefillStepSize,
                generation: .init(),
                returnLogProbs: config.returnLogProbs
            ),
            stopTokens: buildStopTokens(),
            unknownTokenId: context.tokenizer.unknownTokenId
        )

        while !Task.isCancelled {
            if isShuttingDown { break }

            let unattached = activeRequests.values.filter { $0.uid == nil }
            if !unattached.isEmpty {
                let prompts = unattached.map { $0.request.tokens }
                let maxTokens = unattached.map { $0.request.maxTokens }
                let samplers = unattached.map { $0.request.params.sampler() }
                let processors = unattached.map { $0.request.params.processor() }
                let uids = iterator.insert(
                    prompts: prompts,
                    maxTokens: maxTokens,
                    samplers: samplers,
                    processors: processors
                )
                for (uid, request) in zip(uids, unattached) {
                    uidToRequestID[uid] = request.request.id
                    var updated = activeRequests[request.request.id]
                    updated?.uid = uid
                    if let updated {
                        activeRequests[request.request.id] = updated
                    }
                }
            }

            if !pendingBatchRemovals.isEmpty {
                iterator.remove(uids: pendingBatchRemovals)
                for uid in pendingBatchRemovals {
                    uidToRequestID.removeValue(forKey: uid)
                }
                pendingBatchRemovals.removeAll()
            }

            guard let responses = iterator.next(), !responses.isEmpty else {
                if activeRequests.isEmpty { break }
                await Task.yield()
                continue
            }

            let now = Date.timeIntervalSinceReferenceDate
            var admittedDuringTick = false

            for response in responses {
                guard let requestID = uidToRequestID[response.uid],
                    var entry = activeRequests[requestID]
                else { continue }

                if cancelledRequests.contains(requestID) {
                    iterator.remove(uids: [response.uid])
                    uidToRequestID.removeValue(forKey: response.uid)
                    cancelledRequests.remove(requestID)
                    activeRequests.removeValue(forKey: requestID)
                    continuations.removeValue(forKey: requestID)
                    requestStats.removeValue(forKey: requestID)
                    admittedDuringTick = true
                    continue
                }

                if entry.firstTokenAt == nil {
                    let promptTime = now - entry.startTime
                    entry.firstTokenAt = now
                    updateStats(for: requestID, promptTime: promptTime)
                } else if let last = entry.lastEmitAt {
                    let delta = now - last
                    updateStats(for: requestID, generationDelta: delta)
                }
                entry.lastEmitAt = now

                updateStats(for: requestID, incrementGeneratedTokens: 1)
                activeRequests[requestID] = entry

                let text = context.tokenizer.decode(tokens: [response.token])
                let logProbs: MLXArray? =
                    config.returnLogProbs && response.logProbs.size > 0 ? response.logProbs : nil

                let event = TokenEvent(
                    requestID: requestID,
                    token: response.token,
                    textDelta: text.isEmpty ? nil : text,
                    finishReason: response.finishReason,
                    logProbs: logProbs
                )
                entry.continuation.yield(event)

                if response.finishReason != nil {
                    entry.continuation.finish()
                    activeRequests.removeValue(forKey: requestID)
                    uidToRequestID.removeValue(forKey: response.uid)
                    continuations.removeValue(forKey: requestID)
                    cancelledRequests.remove(requestID)
                    completedRequests += 1
                    admittedDuringTick = true
                } else {
                    activeRequests[requestID] = entry
                }
            }

            if activeRequests.isEmpty {
                break
            }

            if admittedDuringTick {
                await tryAdmitFromQueue()
            }

            await Task.yield()
        }

        currentMode = .idle
        await tryAdmitFromQueue()
    }

    private func buildStopTokens() -> Set<Int> {
        var stopTokens = Set<Int>()
        if let eos = context.tokenizer.eosTokenId {
            stopTokens.insert(eos)
        }
        for token in context.configuration.extraEOSTokens ?? [] {
            if let id = context.tokenizer.convertTokenToId(token) {
                stopTokens.insert(id)
            }
        }
        return stopTokens
    }

    private func updateStats(
        for requestID: UUID,
        incrementGeneratedTokens: Int = 0,
        promptTime: TimeInterval? = nil,
        generationDelta: TimeInterval? = nil
    ) {
        guard var stats = requestStats[requestID] else { return }

        if let promptTime {
            stats.promptTime = promptTime
        }

        if incrementGeneratedTokens > 0 {
            stats.generatedTokens += incrementGeneratedTokens
            totalTokensGenerated += incrementGeneratedTokens
        }

        if let generationDelta {
            stats.generationTime += generationDelta
            totalGenerationTime += generationDelta
        }

        requestStats[requestID] = stats
    }
}
