// Copyright © 2024 Apple Inc.

import ArgumentParser
import CoreImage
import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

@main
struct LLMTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for generating text and manipulating LLMs",
        subcommands: [
            EvaluateCommand.self,
            BenchmarkCommand.self,
            ChatCommand.self,
            LoRACommand.self,
            ListCommands.self,
        ],
        defaultSubcommand: EvaluateCommand.self)
}

/// Command line arguments for loading a model.
struct ModelArguments: ParsableArguments, Sendable {

    @Option(name: .long, help: "Name of the Hugging Face model or absolute path to directory")
    var model: String?

    @Option(help: "Hub download directory")
    var download: URL?

    @Sendable
    func load(defaultModel: String, modelFactory: ModelFactory) async throws -> ModelContainer {
        let modelConfiguration: ModelConfiguration

        let modelName = self.model ?? defaultModel

        print("Loading \(modelName)...")

        if modelName.hasPrefix("/") {
            // path
            modelConfiguration = ModelConfiguration(directory: URL(filePath: modelName))
        } else {
            // identifier
            modelConfiguration = modelFactory.configuration(id: modelName)
        }

        let hub =
            if let download {
                HubApi(downloadBase: download)
            } else {
                HubApi()
            }

        return try await modelFactory.loadContainer(hub: hub, configuration: modelConfiguration)
    }
}

struct PromptArguments: ParsableArguments, Sendable {
    @Option(
        name: .shortAndLong,
        help:
            "The message to be processed by the model. Use @path,@path to load from files, e.g. @/tmp/prompt.txt"
    )
    var prompt: String?

    func resolvePrompt(configuration: ModelConfiguration) throws -> String {
        let prompt = self.prompt ?? configuration.defaultPrompt
        if prompt.hasPrefix("@") {
            let names = prompt.split(separator: ",").map { String($0.dropFirst()) }
            return try names.map { try String(contentsOfFile: $0) }.joined(separator: "\n")
        } else {
            return prompt
        }
    }
}

/// Argument package for supplying media files
struct MediaArguments: ParsableArguments, Sendable {

    @Option(parsing: .upToNextOption, help: "Resize images to this size (width, height)")
    var resize: [Int] = []

    @Option(parsing: .upToNextOption, help: "Paths or URLs for input images")
    var image: [URL] = []

    @Option(parsing: .upToNextOption, help: "Paths or URLs for input videos")
    var video: [URL] = []

    var images: [UserInput.Image] {
        image.map { UserInput.Image.url($0) }
    }
    var videos: [UserInput.Video] {
        video.map { UserInput.Video.url($0) }
    }

    var processing: UserInput.Processing {
        var processing = UserInput.Processing()
        if !resize.isEmpty {
            let size: CGSize
            if resize.count == 1 {
                // Single value represents width/height
                let v = resize[0]
                size = CGSize(width: v, height: v)
            } else {
                let v0 = resize[0]
                let v1 = resize[1]
                size = CGSize(width: v0, height: v1)
            }
            processing.resize = size
        }
        return processing
    }
}

/// Command line arguments for controlling generation of text.
struct GenerateArguments: ParsableArguments, Sendable {

    @Option(
        name: .shortAndLong,
        help:
            "The system prompt"
    )
    var system: String = ""

    @Option(name: .shortAndLong, help: "Maximum number of tokens to generate")
    var maxTokens = 100

    @Option(name: .shortAndLong, help: "The sampling temperature")
    var temperature: Float = 0.6

    @Option(name: .long, help: "The top p sampling")
    var topP: Float = 1.0

    @Option(name: .long, help: "The penalty factor for repeating tokens")
    var repetitionPenalty: Float?

    @Option(name: .long, help: "The number of tokens to consider for repetition penalty")
    var repetitionContextSize: Int = 20

    @Option(name: .long, help: "Additional end-of-sequence token to stop generation")
    var extraEosToken: String?

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64 = 0

    @Option(name: .long, help: "Number of bits for KV cache quantization (nil = no quantization)")
    var kvBits: Int?

    @Option(name: .long, help: "Group size for KV cache quantization")
    var kvGroupSize: Int = 64

    @Option(name: .long, help: "Step to begin using quantized KV cache when kv-bits is set")
    var quantizedKvStart: Int = 0

    @Flag(name: .shortAndLong, help: "If true only print the generated output")
    var quiet = false

    var generateParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKvStart,
            temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize)
    }

    func prepare(
        _ context: inout ModelContext
    ) {
        if let extraEosToken {
            context.configuration.extraEOSTokens.insert(extraEosToken)
        }
    }

    func generate(
        input: LMInput, context: ModelContext
    ) async throws -> (GenerateCompletionInfo, String) {
        var output = ""
        for await item in try MLXLMCommon.generate(
            input: input, parameters: generateParameters, context: context)
        {
            switch item {
            case .chunk(let string):
                output += string
                print(string, terminator: "")
            case .info(let info):
                return (info, output)
            case .toolCall:
                break
            }
        }
        fatalError("exited loop without seeing .info")
    }
}

/// Arguments controlling benchmark execution.
struct BenchmarkArguments: ParsableArguments, Sendable {

    enum Mode: String, ExpressibleByArgument {
        case batch
        case stream
    }

    @Option(
        name: .long,
        help: "Length of randomly generated prompt in tokens (ignored when providing prompts)."
    )
    var promptTokens: Int = 512

    @Option(
        name: .long,
        help: "Maximum number of tokens to generate per prompt."
    )
    var generationTokens: Int = 1_024

    @Option(
        name: .long,
        help: "Number of prompts in the batch. Ignored when prompts are provided."
    )
    var batchSize: Int = 1

    @Option(
        name: .long,
        help: "Number of timed benchmark trials."
    )
    var numTrials: Int = 5

    @Option(
        name: .long,
        help: "Seed used when generating random token prompts."
    )
    var seed: UInt64 = 0

    @Option(
        name: .long,
        help: "Benchmark mode: batched generation or single-stream generation."
    )
    var mode: Mode = .batch

    @Option(
        name: .long,
        parsing: .upToNextOption,
        help: "Explicit prompts to benchmark (repeat flag to supply multiple prompts)."
    )
    var prompts: [String] = []

    @Option(
        name: .long,
        help: "Path to a file containing prompts (one per line or separated by blank lines)."
    )
    var promptsFile: String?

    @Flag(
        name: .long,
        help: "Apply the tokenizer chat template to provided prompts."
    )
    var applyChatTemplate = false

    @Option(
        name: .long,
        help: "System prompt to prepend when applying the chat template."
    )
    var systemPrompt: String?
}

/// Argument package for adjusting and reporting memory use.
struct MemoryArguments: ParsableArguments, Sendable {

    @Flag(name: .long, help: "Show memory stats")
    var memoryStats = false

    @Option(name: .long, help: "Maximum cache size in M")
    var cacheSize: Int?

    @Option(name: .long, help: "Maximum memory size in M")
    var memorySize: Int?

    var startMemory: GPU.Snapshot?

    mutating func start<L>(_ load: @Sendable () async throws -> L) async throws -> L {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }

        let result = try await load()
        startMemory = GPU.snapshot()

        return result
    }

    mutating func start() {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }

        startMemory = GPU.snapshot()
    }

    func reportCurrent() {
        if memoryStats {
            let memory = GPU.snapshot()
            print(memory.description)
        }
    }

    func reportMemoryStatistics() {
        if memoryStats, let startMemory {
            let endMemory = GPU.snapshot()

            print("=======")
            print("Memory size: \(GPU.memoryLimit / 1024)K")
            print("Cache size:  \(GPU.cacheLimit / 1024)K")

            print("")
            print("=======")
            print("Starting memory")
            print(startMemory.description)

            print("")
            print("=======")
            print("Ending memory")
            print(endMemory.description)

            print("")
            print("=======")
            print("Growth")
            print(startMemory.delta(endMemory).description)

        }
    }
}

struct BenchmarkCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Benchmark batched or streaming generation throughput."
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var benchmark: BenchmarkArguments

    private static let defaultModel = MLXLLM.LLMRegistry.llama3_2_3B_4bit

    mutating func run() async throws {
        guard benchmark.numTrials > 0 else {
            throw ValidationError("--num-trials must be greater than zero")
        }

        let hub: HubApi
        if let download = args.download {
            hub = HubApi(downloadBase: download)
        } else {
            hub = HubApi()
        }

        let modelContainer = try await memory.start { [args] in
            try await args.load(defaultModel: Self.defaultModel.name, modelFactory: LLMModelFactory.shared)
        }

        let configuration = await modelContainer.configuration
        let vocabSize = loadVocabSize(configuration: configuration, hub: hub) ?? 32_000
        let providedPrompts = try loadPrompts()

        try await modelContainer.perform { context in
            let tokenPrompts: [[Int]]
            if let providedPrompts, !providedPrompts.isEmpty {
                tokenPrompts = try encodePrompts(
                    providedPrompts,
                    tokenizer: context.tokenizer,
                    applyChatTemplate: benchmark.applyChatTemplate,
                    systemPrompt: benchmark.systemPrompt
                )
            } else {
                guard benchmark.batchSize > 0 else {
                    throw ValidationError("--batch-size must be greater than zero when prompts are not provided")
                }

                guard benchmark.promptTokens > 0 else {
                    throw ValidationError("--prompt-tokens must be greater than zero when prompts are not provided")
                }

                MLXRandom.seed(benchmark.seed)
                tokenPrompts = makeRandomPrompts(
                    batchSize: benchmark.batchSize,
                    promptLength: benchmark.promptTokens,
                    vocabSize: vocabSize
                )
            }

            guard !tokenPrompts.isEmpty else {
                throw ValidationError("No prompts available for benchmarking")
            }

            let effectiveBatchSize = benchmark.mode == .stream ? 1 : tokenPrompts.count
            let completionBatchSize = max(effectiveBatchSize, 1)
            let prefillBatchSize = max(1, min(8, completionBatchSize))

            var generationParameters = GenerateParameters(maxTokens: benchmark.generationTokens)
            generationParameters.temperature = 0.0

            let batchParameters = BatchGenerateParameters(
                maxTokens: benchmark.generationTokens,
                completionBatchSize: completionBatchSize,
                prefillBatchSize: prefillBatchSize,
                prefillStepSize: 2_048,
                generation: generationParameters
            )

            let maxTokens = Array(repeating: benchmark.generationTokens, count: tokenPrompts.count)
            let promptSummaryLength: Int = {
                if providedPrompts != nil {
                    return tokenPrompts.map(\.count).max() ?? 0
                } else {
                    return benchmark.promptTokens
                }
            }()

            print("Running warmup..")
            switch benchmark.mode {
            case .batch:
                _ = try runBatch(
                    context: context,
                    prompts: tokenPrompts,
                    maxTokens: maxTokens,
                    parameters: batchParameters
                )
            case .stream:
                guard let firstPrompt = tokenPrompts.first else {
                    throw ValidationError("Stream mode requires at least one prompt")
                }
                _ = try runStream(
                    context: context,
                    prompt: firstPrompt,
                    parameters: generationParameters
                )
            }
            GPU.clearCache()

            let reportedBatchSize = benchmark.mode == .stream ? 1 : tokenPrompts.count
            print(
                "Timing with promptTokens=\(promptSummaryLength), generationTokens=\(benchmark.generationTokens), batchSize=\(reportedBatchSize)."
            )

            var trials: [BatchGenerateStats] = []
            trials.reserveCapacity(benchmark.numTrials)

            for trialIndex in 0..<benchmark.numTrials {
                let stats: BatchGenerateStats
                switch benchmark.mode {
                case .batch:
                    stats = try runBatch(
                        context: context,
                        prompts: tokenPrompts,
                        maxTokens: maxTokens,
                        parameters: batchParameters
                    )
                case .stream:
                    guard let firstPrompt = tokenPrompts.first else {
                        throw ValidationError("Stream mode requires at least one prompt")
                    }
                    stats = try runStream(
                        context: context,
                        prompt: firstPrompt,
                        parameters: generationParameters
                    )
                }

                trials.append(stats)
                print(
                    "Trial \(trialIndex + 1):  prompt_tps=\(format(stats.promptTokensPerSecond)), generation_tps=\(format(stats.generationTokensPerSecond)), peak_memory=\(format(stats.peakMemoryGB))"
                )
                GPU.clearCache()
            }

            if let averages = averageStats(trials) {
                print(
                    "Averages: prompt_tps=\(format(averages.promptTPS)), generation_tps=\(format(averages.generationTPS)), peak_memory=\(format(averages.peakMemory))"
                )
            } else {
                print("Averages: prompt_tps=0.000, generation_tps=0.000, peak_memory=0.000")
            }
        }

        memory.reportMemoryStatistics()
    }
}

extension BenchmarkCommand {
    private func loadPrompts() throws -> [String]? {
        var prompts = benchmark.prompts.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let filePath = benchmark.promptsFile {
            let url = URL(fileURLWithPath: filePath)
            let contents = try String(contentsOf: url, encoding: .utf8)
            prompts.append(contentsOf: parsePromptFile(contents))
        }

        let filtered = prompts.filter { !$0.isEmpty }
        return filtered.isEmpty ? nil : filtered
    }

    private func parsePromptFile(_ contents: String) -> [String] {
        let normalized = contents.replacingOccurrences(of: "\r\n", with: "\n")

        let dashSeparated = normalized.components(separatedBy: "\n---\n")
        var prompts: [String] = []
        for segment in dashSeparated {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                continue
            }
            let blankSeparated = trimmed
                .components(separatedBy: "\n\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if blankSeparated.count > 1 {
                prompts.append(contentsOf: blankSeparated)
            } else {
                prompts.append(trimmed)
            }
        }

        if !prompts.isEmpty {
            return prompts
        }

        return normalized
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func loadVocabSize(configuration: ModelConfiguration, hub: HubApi) -> Int? {
        let configURL = configuration.modelDirectory(hub: hub).appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
            let json = try? JSONSerialization.jsonObject(with: data)
        else {
            return nil
        }
        return extractVocabSize(from: json)
    }

    private func extractVocabSize(from value: Any) -> Int? {
        if let intValue = value as? Int {
            return intValue
        }

        if let doubleValue = value as? Double {
            return Int(doubleValue)
        }

        if let dict = value as? [String: Any] {
            let preferredKeys = [
                "vocab_size",
                "text_vocab_size",
                "vocabulary_size",
                "tokenizer_vocab_size",
                "vocab_size_per_layer_input",
            ]
            for key in preferredKeys {
                if let entry = dict[key], let size = extractVocabSize(from: entry) {
                    return size
                }
            }
            for (_, nested) in dict {
                if let size = extractVocabSize(from: nested) {
                    return size
                }
            }
        }

        if let array = value as? [Any] {
            for element in array {
                if let size = extractVocabSize(from: element) {
                    return size
                }
            }
        }

        return nil
    }

    private func makeRandomPrompts(batchSize: Int, promptLength: Int, vocabSize: Int) -> [[Int]] {
        guard batchSize > 0, promptLength > 0 else {
            return []
        }

        let tokens = MLXRandom.randInt(0 ..< vocabSize, [batchSize, promptLength])
        tokens.eval()
        let flat: [Int32] = tokens.asArray(Int32.self)

        var prompts: [[Int]] = []
        prompts.reserveCapacity(batchSize)

        let strideLength = promptLength
        for row in 0..<batchSize {
            let start = row * strideLength
            let end = start + strideLength
            let slice = flat[start..<end].map { Int($0) }
            prompts.append(slice)
        }

        return prompts
    }

    private func encodePrompts(
        _ prompts: [String],
        tokenizer: Tokenizer,
        applyChatTemplate: Bool,
        systemPrompt: String?
    ) throws -> [[Int]] {
        if applyChatTemplate {
            return try prompts.map { prompt in
                var messages: [[String: String]] = []
                if let systemPrompt, !systemPrompt.isEmpty {
                    messages.append(["role": "system", "content": systemPrompt])
                }
                messages.append(["role": "user", "content": prompt])
                return try tokenizer.applyChatTemplate(messages: messages)
            }
        } else {
            return prompts.map { tokenizer.encode(text: $0) }
        }
    }

    private func runBatch(
        context: ModelContext,
        prompts: [[Int]],
        maxTokens: [Int],
        parameters: BatchGenerateParameters
    ) throws -> BatchGenerateStats {
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
            unknownTokenId: context.tokenizer.unknownTokenId
        )

        _ = iterator.insert(prompts: prompts, maxTokens: maxTokens)

        // Consume all responses without evaluating logprobs (matches Python benchmark)
        while let _ = iterator.next() {
            // Python doesn't evaluate logprobs during benchmarking
        }

        Stream().synchronize()
        return iterator.stats()
    }

    private func runStream(
        context: ModelContext,
        prompt: [Int],
        parameters: GenerateParameters
    ) throws -> BatchGenerateStats {
        let input = LMInput(tokens: MLXArray(prompt))
        let info = try MLXLMCommon.generate(
            input: input,
            parameters: parameters,
            context: context,
            didGenerate: { (_: Int) -> GenerateDisposition in .more }
        )

        return BatchGenerateStats(
            promptTokenCount: info.promptTokenCount,
            promptTime: info.promptTime,
            generationTokenCount: info.generationTokenCount,
            generationTime: info.generateTime,
            peakMemoryGB: Double(GPU.peakMemory) / 1_000_000_000.0
        )
    }

    private func averageStats(_ stats: [BatchGenerateStats])
        -> (promptTPS: Double, generationTPS: Double, peakMemory: Double)?
    {
        guard !stats.isEmpty else {
            return nil
        }

        let count = Double(stats.count)
        let promptTPS = stats.reduce(0) { $0 + $1.promptTokensPerSecond } / count
        let generationTPS = stats.reduce(0) { $0 + $1.generationTokensPerSecond } / count
        let peakMemory = stats.reduce(0) { $0 + $1.peakMemoryGB } / count

        return (promptTPS, generationTPS, peakMemory)
    }

    private func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

struct EvaluateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "evaluate prompt and generate text"
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments
    @OptionGroup var prompt: PromptArguments
    @OptionGroup var media: MediaArguments

    private func userInput(modelConfiguration: ModelConfiguration) -> UserInput {
        let prompt =
            (try? self.prompt.resolvePrompt(configuration: modelConfiguration))
            ?? modelConfiguration.defaultPrompt

        return UserInput(
            chat: [
                .system(generate.system),
                .user(prompt, images: media.images, videos: media.videos),
            ],
            processing: media.processing
        )
    }

    @MainActor
    mutating func run() async throws {
        let modelFactory: ModelFactory
        let defaultModel: ModelConfiguration

        // Switch between LLM and VLM based on presence of media
        let vlm = !media.image.isEmpty || !media.video.isEmpty
        if vlm {
            modelFactory = VLMModelFactory.shared
            defaultModel = MLXVLM.VLMRegistry.qwen2VL2BInstruct4Bit
        } else {
            modelFactory = LLMModelFactory.shared
            defaultModel = MLXLLM.LLMRegistry.mistral7B4bit
        }

        // Load the model
        let modelContainer = try await memory.start { [args] in
            try await args.load(defaultModel: defaultModel.name, modelFactory: modelFactory)
        }

        // update the context/configuration with any command line parameters
        await modelContainer.update { [generate] context in
            generate.prepare(&context)
        }

        // Get the resolved configuration (this has the default prompt)
        let modelConfiguration = await modelContainer.configuration

        if !generate.quiet {
            print("Loaded \(modelConfiguration.name)")
        }

        let userInput = self.userInput(modelConfiguration: modelConfiguration)

        if !generate.quiet {
            print("Starting generation ...")
            print(userInput.prompt, terminator: " ")
        }

        let (result, _) = try await modelContainer.perform { [generate] context in
            let input = try await context.processor.prepare(input: userInput)
            return try await generate.generate(input: input, context: context)
        }

        // wait for any asynchronous cleanup, e.g. tearing down compiled functions
        // before the task exits -- this would race with mlx::core shutdown
        try await Task.sleep(for: .milliseconds(30))

        if !generate.quiet {
            print("------")
            print(result.summary())

            memory.reportMemoryStatistics()
        }
    }
}
