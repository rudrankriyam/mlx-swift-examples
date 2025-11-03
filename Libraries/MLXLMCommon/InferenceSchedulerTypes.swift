import Foundation
import MLX

public struct SchedulerConfig: Sendable {
    public var maxConcurrentRequests: Int
    public var completionBatchSize: Int
    public var prefillBatchSize: Int
    public var prefillStepSize: Int
    public var returnLogProbs: Bool

    public init(
        maxConcurrentRequests: Int = 64,
        completionBatchSize: Int = 16,
        prefillBatchSize: Int = 8,
        prefillStepSize: Int = 2_048,
        returnLogProbs: Bool = true
    ) {
        self.maxConcurrentRequests = maxConcurrentRequests
        self.completionBatchSize = completionBatchSize
        self.prefillBatchSize = prefillBatchSize
        self.prefillStepSize = prefillStepSize
        self.returnLogProbs = returnLogProbs
    }
}

public struct InferenceRequest: Sendable, Identifiable {
    public let id: UUID
    public let tokens: [Int]
    public let params: GenerateParameters
    public let maxTokens: Int
    public let createdAt: Date

    public init(
        id: UUID = UUID(),
        tokens: [Int],
        params: GenerateParameters,
        maxTokens: Int,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.tokens = tokens
        self.params = params
        self.maxTokens = maxTokens
        self.createdAt = createdAt
    }
}

public struct TokenEvent: Sendable {
    public let requestID: UUID
    public let token: Int?
    public let textDelta: String?
    public let finishReason: BatchGenerateResult.FinishReason?
    public let logProbs: MLXArray?

    public init(
        requestID: UUID,
        token: Int?,
        textDelta: String?,
        finishReason: BatchGenerateResult.FinishReason?,
        logProbs: MLXArray?
    ) {
        self.requestID = requestID
        self.token = token
        self.textDelta = textDelta
        self.finishReason = finishReason
        self.logProbs = logProbs
    }
}

public struct RequestStats: Sendable {
    public let requestID: UUID
    public var promptTokens: Int
    public var generatedTokens: Int
    public var promptTime: TimeInterval
    public var generationTime: TimeInterval

    public init(
        requestID: UUID,
        promptTokens: Int,
        generatedTokens: Int = 0,
        promptTime: TimeInterval = 0,
        generationTime: TimeInterval = 0
    ) {
        self.requestID = requestID
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.promptTime = promptTime
        self.generationTime = generationTime
    }

    public var tokensPerSecond: Double {
        guard generationTime > 0 else { return 0 }
        return Double(generatedTokens) / generationTime
    }
}

public struct AggregatedStats: Sendable {
    public let activeRequests: Int
    public let queuedRequests: Int
    public let totalRequestsProcessed: Int
    public let averageTPS: Double
    public let peakMemoryGB: Double

    public init(
        activeRequests: Int,
        queuedRequests: Int,
        totalRequestsProcessed: Int,
        averageTPS: Double,
        peakMemoryGB: Double
    ) {
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
        self.totalRequestsProcessed = totalRequestsProcessed
        self.averageTPS = averageTPS
        self.peakMemoryGB = peakMemoryGB
    }
}

public enum SchedulerError: Error, Sendable {
    case modelInferenceFailed(underlying: Error)
    case cacheAllocationFailed
    case tokenizationFailed
    case requestCancelled
    case shutdownInProgress
}
