// Copyright © 2024 Apple Inc.

import AsyncAlgorithms
import MLX
import MLXLLM
import MLXLMCommon
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {
    @Environment(DeviceStat.self) private var deviceStat

    @State var llm = LLMEvaluator()

    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = displayStyle.markdown
    @State private var useBatchMode = false

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                HStack {
                    Toggle(isOn: $llm.includeWeatherTool) {
                        Text("Include tools")
                    }
                    .frame(maxWidth: 150, alignment: .leading)
                    Toggle(isOn: $llm.enableThinking) {
                        Text("Thinking")
                            .help(
                                "Switches between thinking and non-thinking modes. Support: Qwen3")
                    }
                    .frame(maxWidth: 150, alignment: .leading)
                    Toggle(isOn: $useBatchMode) {
                        Text("Batch Mode")
                            .help(
                                "Generate responses for multiple prompts in parallel. Enter prompts separated by newlines.")
                    }
                    .frame(maxWidth: 150, alignment: .leading)
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                    Picker("", selection: $selectedDisplayStyle) {
                        ForEach(displayStyle.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }

                    }
                    .pickerStyle(.segmented)
                    #if os(visionOS)
                        .frame(maxWidth: 250)
                    #else
                        .frame(maxWidth: 150)
                    #endif
                }
            }

            // show the model output
            if useBatchMode && !llm.batchResults.isEmpty {
                // Side-by-side batch results
                HStack(spacing: 0) {
                    ForEach(Array(llm.batchResults.enumerated()), id: \.offset) { idx, result in
                        VStack(alignment: .leading, spacing: 8) {
                            // Header
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Prompt \(idx + 1)")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(.blue)
                                Text(result.prompt)
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary)
                                    .lineLimit(2)
                                Divider()
                            }
                            .padding(.horizontal, 12)
                            .padding(.top, 8)

                            // Response in scrollable area
                            ScrollView(.vertical) {
                                VStack(alignment: .leading, spacing: 8) {
                                    if selectedDisplayStyle == .plain {
                                        Text(result.response)
                                            .font(.system(size: 11))
                                            .textSelection(.enabled)
                                    } else {
                                        Markdown(result.response)
                                            .markdownTextStyle(\.text) {
                                                FontSize(11)
                                            }
                                            .textSelection(.enabled)
                                    }

                                    // Footer
                                    Text("\(result.tokenCount) tokens • \(result.finishReason)")
                                        .font(.system(size: 9))
                                        .foregroundColor(.secondary)
                                        .padding(.top, 8)
                                }
                                .padding(.horizontal, 12)
                                .padding(.bottom, 8)
                            }
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color.gray.opacity(0.05))

                        if idx < llm.batchResults.count - 1 {
                            Divider()
                        }
                    }
                }
            } else {
                // Standard single output
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            if selectedDisplayStyle == .plain {
                                Text(llm.output)
                                    .textSelection(.enabled)
                            } else {
                                Markdown(llm.output)
                                    .textSelection(.enabled)
                            }
                        }
                        .onChange(of: llm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }

                        Spacer()
                            .frame(width: 1, height: 1)
                            .id("bottom")
                    }
                }
            }

            HStack {
                if useBatchMode {
                    ZStack(alignment: .topLeading) {
                        if llm.prompt.isEmpty {
                            Text("Enter multiple prompts, one per line:\nHi\nWhat is quantum computing and why is it important?")
                                .foregroundColor(.gray.opacity(0.6))
                                .padding(.top, 8)
                                .padding(.leading, 5)
                        }
                        TextEditor(text: Bindable(llm).prompt)
                            .frame(height: 60)
                            .disabled(llm.running)
                            .opacity(llm.prompt.isEmpty ? 0.5 : 1)
                    }
                    #if os(visionOS)
                        .border(Color.gray, width: 1)
                    #else
                        .border(Color.gray.opacity(0.5), width: 1)
                    #endif
                } else {
                    TextField("prompt", text: Bindable(llm).prompt)
                        .onSubmit(generate)
                        .disabled(llm.running)
                        #if os(visionOS)
                            .textFieldStyle(.roundedBorder)
                        #endif
                }
                Button(llm.running ? "stop" : "generate", action: llm.running ? cancel : generate)
            }
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }

        }
        .task {
            do {
                // pre-load the weights on launch to speed up the first generation
                _ = try await llm.load()
            } catch {
                llm.output = "Failed: \(error)"
            }
        }
    }

    private func generate() {
        if useBatchMode {
            llm.generateBatch()
        } else {
            llm.generate()
        }
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}

@Observable
@MainActor
class LLMEvaluator {

    var running = false

    // Store batch results for side-by-side display
    struct BatchResult {
        let prompt: String
        let response: String
        let tokenCount: Int
        let finishReason: String
    }
    var batchResults: [BatchResult] = []

    var includeWeatherTool = false
    var enableThinking = false

    var prompt = ""
    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. Using Llama 3.2 3B to match Python's batch generation example.
    let modelConfiguration = LLMRegistry.llama3_2_3B_4bit

    /// parameters controlling the output
    let generateParameters = GenerateParameters(maxTokens: 500, temperature: 0.6)
    let updateInterval = Duration.seconds(0.25)

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    let currentWeatherTool = Tool<WeatherInput, WeatherOutput>(
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: [
            .required(
                "location", type: .string, description: "The city and state, e.g. San Francisco, CA"
            ),
            .optional(
                "unit",
                type: .string,
                description: "The unit of temperature",
                extraProperties: [
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                ]
            ),
        ]
    ) { input in
        let range = input.unit == "celsius" ? (min: -20.0, max: 40.0) : (min: 0, max: 100)
        let temperature = Double.random(in: range.min ... range.max)

        let conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy"].randomElement()!

        return WeatherOutput(temperature: temperature, conditions: conditions)
    }

    let addTool = Tool<AddInput, AddOutput>(
        name: "add_two_numbers",
        description: "Add two numbers together",
        parameters: [
            .required("first", type: .int, description: "The first number to add"),
            .required("second", type: .int, description: "The second number to add"),
        ]
    ) { input in
        AddOutput(result: input.first + input.second)
    }

    let timeTool = Tool<EmptyInput, TimeOutput>(
        name: "get_time",
        description: "Get the current time",
        parameters: []
    ) { _ in
        TimeOutput(time: Date.now.formatted())
    }

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            self.prompt = modelConfiguration.defaultPrompt
            self.modelInfo =
                "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func generate(prompt: String, toolResult: String? = nil) async {

        self.output = ""
        var chat: [Chat.Message] = [
            .system("You are a helpful assistant"),
            .user(prompt),
        ]

        if let toolResult {
            chat.append(.tool(toolResult))
        }

        let userInput = UserInput(
            chat: chat,
            tools: includeWeatherTool
                ? [currentWeatherTool.schema, addTool.schema, timeTool.schema] : nil,
            additionalContext: ["enable_thinking": enableThinking]
        )

        do {
            let modelContainer = try await load()

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) -> Void in
                let lmInput = try await context.processor.prepare(input: userInput)
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context)

                // generate and output in batches
                for await batch in stream._throttle(
                    for: updateInterval, reducing: Generation.collect)
                {
                    let output = batch.compactMap { $0.chunk }.joined(separator: "")
                    if !output.isEmpty {
                        Task { @MainActor [output] in
                            self.output += output
                        }
                    }

                    if let completion = batch.compactMap({ $0.info }).first {
                        Task { @MainActor in
                            self.stat = "\(completion.tokensPerSecond) tokens/s"
                        }
                    }

                    if let toolCall = batch.compactMap({ $0.toolCall }).first {
                        try await handleToolCall(toolCall, prompt: prompt)
                    }
                }
            }

        } catch {
            output = "Failed: \(error)"
        }
    }

    func generate() {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt)
            running = false
        }
    }

    func generateBatch() {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        batchResults = [] // Clear previous results
        generationTask = Task {
            running = true
            await generateBatchPrompts(promptText: currentPrompt)
            running = false
        }
    }

    private func generateBatchPrompts(promptText: String) async {
        self.output = ""

        // Split prompts by newlines and filter empty ones
        let prompts = promptText
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !prompts.isEmpty else {
            self.output = "No prompts provided. Enter prompts separated by newlines."
            return
        }

        self.output = "Processing \(prompts.count) prompts in batch...\n\n"

        do {
            let modelContainer = try await load()

            // Seed for deterministic results
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) -> Void in
                // Create batch parameters with deterministic sampling
                var batchGenParams = generateParameters
                batchGenParams.temperature = 0.0  // Greedy/deterministic sampling

                let batchParams = BatchGenerateParameters(
                    maxTokens: generateParameters.maxTokens ?? 128,
                    completionBatchSize: 32,
                    prefillBatchSize: 8,
                    prefillStepSize: 2048,
                    generation: batchGenParams
                )

                // Use streaming batch generation
                // Apply chat template to each prompt
                let tokenized: [[Int]] = try prompts.map { prompt in
                    let messages = [[
                        "role": "user",
                        "content": prompt
                    ]]
                    return try context.tokenizer.applyChatTemplate(messages: messages)
                }
                let defaultMaxTokens = Array(
                    repeating: batchParams.defaultMaxTokens, count: tokenized.count)

                // Set up stop tokens (matches batchGenerate pattern)
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
                    parameters: batchParams,
                    stopTokens: stopTokens,
                    unknownTokenId: context.tokenizer.unknownTokenId
                )

                let uids = iterator.insert(prompts: tokenized, maxTokens: defaultMaxTokens)

                // Initialize batch results with empty responses
                Task { @MainActor in
                    self.batchResults = prompts.enumerated().map { (idx, prompt) in
                        BatchResult(
                            prompt: prompt,
                            response: "",
                            tokenCount: 0,
                            finishReason: "generating"
                        )
                    }
                }

                var generatedTokens = [Int: [Int]]()
                var finishReasons = [Int: BatchGenerateResult.FinishReason]()

                // Stream tokens and update UI incrementally
                while let responses = iterator.next(), !responses.isEmpty {
                    for response in responses {
                        if response.finishReason != .stop {
                            generatedTokens[response.uid, default: []].append(response.token)
                        }
                        if let reason = response.finishReason {
                            finishReasons[response.uid] = reason
                        }
                    }

                    // Update UI with current state
                    Task { @MainActor in
                        self.batchResults = uids.enumerated().map { (idx, uid) in
                            let tokens = generatedTokens[uid] ?? []
                            let text = context.tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                            let finish = finishReasons[uid]?.rawValue ?? "generating"

                            return BatchResult(
                                prompt: prompts[idx],
                                response: text,
                                tokenCount: tokens.count,
                                finishReason: finish
                            )
                        }
                    }

                    // Small delay to batch UI updates
                    try? await Task.sleep(for: .milliseconds(50))
                }

                Stream().synchronize()

                let stats = iterator.stats()

                // Final update with statistics
                Task { @MainActor in
                    self.batchResults = uids.enumerated().map { (idx, uid) in
                        let tokens = generatedTokens[uid] ?? []
                        let text = context.tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                        let finish = finishReasons[uid]?.rawValue ?? "complete"

                        return BatchResult(
                            prompt: prompts[idx],
                            response: text,
                            tokenCount: tokens.count,
                            finishReason: finish
                        )
                    }

                    // Also set output for non-batch mode fallback
                    var formattedOutput = "# Batch Generation Results\n\n"
                    for (idx, batch) in self.batchResults.enumerated() {
                        formattedOutput += "### Prompt \(idx + 1)\n"
                        formattedOutput += "> \(batch.prompt)\n\n"
                        formattedOutput += "**Response:**\n"
                        formattedOutput += "\(batch.response)\n\n"
                        formattedOutput += "*\(batch.tokenCount) tokens • \(batch.finishReason)*\n\n"
                        if idx < self.batchResults.count - 1 {
                            formattedOutput += "---\n\n"
                        }
                    }

                    // Add statistics
                    formattedOutput += "\n---\n\n## 📊 Statistics\n\n"
                    formattedOutput += "| Metric | Value |\n"
                    formattedOutput += "|--------|-------|\n"
                    formattedOutput += "| Prompt Tokens | \(stats.promptTokenCount) |\n"
                    formattedOutput += "| Prompt TPS | \(String(format: "%.2f", stats.promptTokensPerSecond)) |\n"
                    formattedOutput += "| Prompt Time | \(String(format: "%.2f", stats.promptTime))s |\n"
                    formattedOutput += "| Generation Tokens | \(stats.generationTokenCount) |\n"
                    formattedOutput += "| Generation TPS | \(String(format: "%.2f", stats.generationTokensPerSecond)) |\n"
                    formattedOutput += "| Generation Time | \(String(format: "%.2f", stats.generationTime))s |\n"
                    formattedOutput += "| Peak Memory | \(String(format: "%.2f", stats.peakMemoryGB)) GB |\n"

                    self.output = formattedOutput
                    self.stat = "\(String(format: "%.2f", stats.generationTokensPerSecond)) tokens/s (batch)"
                }
            }

        } catch {
            output = "Failed: \(error)"
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }

    private func handleToolCall(_ toolCall: ToolCall, prompt: String) async throws {
        let result =
            switch toolCall.function.name {
            case currentWeatherTool.name:
                try await toolCall.execute(with: currentWeatherTool).toolResult
            case addTool.name:
                try await toolCall.execute(with: addTool).toolResult
            case timeTool.name:
                try await toolCall.execute(with: timeTool).toolResult
            default:
                "No tool match"
            }

        await generate(prompt: prompt, toolResult: result)
    }
}

struct WeatherInput: Codable {
    let location: String
    let unit: String?
}

struct WeatherOutput: Codable {
    let temperature: Double
    let conditions: String
}

struct AddInput: Codable {
    let first: Int
    let second: Int
}

struct AddOutput: Codable {
    let result: Int
}

struct EmptyInput: Codable {}

struct TimeOutput: Codable {
    let time: String
}
