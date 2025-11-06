@testable import MLXLMCommon
import MLX
import XCTest

final class BatchTokenIteratorTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        setenv("MLX_METAL", "0", 1)
    }

    func testPerRequestSamplersProduceDistinctTokens() throws {
        throw XCTSkip("BatchTokenIterator requires Metal backend unavailable in test environment")
        var iterator = BatchTokenIterator(
            model: MockLanguageModel(vocabularySize: 16),
            parameters: BatchGenerateParameters(
                maxTokens: 1,
                completionBatchSize: 2,
                prefillBatchSize: 2,
                prefillStepSize: 1,
                generation: .init(maxTokens: 1, temperature: 0),
                returnLogProbs: false
            ),
            stopTokens: [],
            unknownTokenId: nil
        )

        let samplers: [LogitSampler] = [FixedTokenSampler(token: 3), FixedTokenSampler(token: 7)]
        let uids = iterator.insert(
            prompts: [[1, 2], [3, 4]],
            maxTokens: [1, 1],
            samplers: samplers,
            processors: nil
        )

        XCTAssertEqual(uids.count, 2)

        guard let responses = iterator.next(), responses.count == 2 else {
            XCTFail("Expected responses for both requests")
            return
        }

        let tokensByUID = Dictionary(uniqueKeysWithValues: responses.map { ($0.uid, $0.token) })
        XCTAssertEqual(tokensByUID[uids[0]], 3)
        XCTAssertEqual(tokensByUID[uids[1]], 7)
    }
}

private struct FixedTokenSampler: LogitSampler {
    let value: Int32

    init(token: Int) {
        self.value = Int32(token)
    }

    func sample(logits: MLXArray) -> MLXArray {
        MLXArray([value])
    }
}
