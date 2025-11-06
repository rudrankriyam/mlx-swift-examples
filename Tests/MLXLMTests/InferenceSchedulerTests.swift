@testable import MLXLMCommon
import XCTest

final class InferenceSchedulerTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        setenv("MLX_METAL", "0", 1)
    }

    func testSoloModeEmitsTokens() async throws {
        let scheduler = InferenceScheduler(context: makeContext(), config: SchedulerConfig(
            maxConcurrentRequests: 2,
            completionBatchSize: 2,
            prefillBatchSize: 1,
            prefillStepSize: 2,
            returnLogProbs: false
        ))

        let request = InferenceRequest(
            tokens: [1, 2, 3],
            params: GenerateParameters(maxTokens: 2, temperature: 0),
            maxTokens: 2,
            createdAt: Date()
        )

        var events: [TokenEvent] = []
        for await event in await scheduler.submit(request) {
            events.append(event)
            if event.finishReason != nil { break }
        }

        XCTAssertEqual(events.filter { $0.token != nil }.count, 2)
        XCTAssertEqual(events.last?.finishReason, .length)

        let stats = await scheduler.stats(for: request.id)
        XCTAssertEqual(stats?.generatedTokens, 2)
    }

    func testCancellationClearsActiveRequests() async throws {
        let scheduler = InferenceScheduler(context: makeContext(), config: SchedulerConfig(
            maxConcurrentRequests: 1,
            completionBatchSize: 1,
            prefillBatchSize: 1,
            prefillStepSize: 2,
            returnLogProbs: false
        ))

        let request = InferenceRequest(
            tokens: [4, 5, 6],
            params: GenerateParameters(maxTokens: 4, temperature: 0),
            maxTokens: 4,
            createdAt: Date()
        )

        let stream = await scheduler.submit(request)

        let consumer = Task {
            var iterator = stream.makeAsyncIterator()
            _ = await iterator.next()
        }

        try await Task.sleep(nanoseconds: 5_000_000)
        consumer.cancel()

        try await Task.sleep(nanoseconds: 50_000_000)

        let stats = await scheduler.stats(for: request.id)
        XCTAssertNil(stats)

        let aggregated = await scheduler.aggregatedStats()
        XCTAssertEqual(aggregated.activeRequests, 0)
    }

    private func makeContext() -> ModelContext {
        let configuration = ModelConfiguration(id: "mock-scheduler", extraEOSTokens: [])
        let model = MockLanguageModel(vocabularySize: 16)
        let tokenizer = SimpleTokenizer()
        return ModelContext(
            configuration: configuration,
            model: model,
            processor: StandInUserInputProcessor(),
            tokenizer: tokenizer
        )
    }
}
