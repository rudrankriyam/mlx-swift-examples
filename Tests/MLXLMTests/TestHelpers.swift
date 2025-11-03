@testable import MLXLMCommon
import MLX
import MLXNN
import Tokenizers

final class MockLanguageModel: Module, LanguageModel {
    private let vocabularySize: Int

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    override init() {
        self.vocabularySize = 16
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.dim(0)
        let sequence = inputs.dim(1)
        let total = batch * sequence * vocabularySize
        var values = [Float](repeating: 0, count: total)

        for batchIndex in 0..<batch {
            for step in 0..<sequence {
                let offset = (batchIndex * sequence + step) * vocabularySize
                values[offset] = 1
            }
        }

        return MLXArray(values, [batch, sequence, vocabularySize])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

struct SimpleTokenizer: Tokenizer {
    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map(String.init)
    }

    func encode(text: String) -> [Int] {
        tokenize(text: text).enumerated().map { index, _ in index }
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map { "t\($0)" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        let digits = token.filter { $0.isNumber }
        return digits.isEmpty ? nil : Int(digits)
    }

    func convertIdToToken(_ id: Int) -> String? {
        "t\(id)"
    }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    var fuseUnknownTokens: Bool { false }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] { [] }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws
        -> [Int]
    { [] }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int] { [] }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] { [] }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] { [] }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int] { [] }
}
