// Copyright 2024 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseek_vl_v2

import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration

public struct DeepSeekVLConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let attentionBias: Bool
        public let scoringFunction: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case attentionBias = "attention_bias"
            case scoringFunction = "scoring_function"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let layers: Int
        public let width: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let numChannels: Int
        public let layerNormEps: Float
        public let mlpRatio: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case layers
            case width
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case numChannels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case mlpRatio = "mlp_ratio"
        }
    }

    public struct ProjectorConfiguration: Codable, Sendable {
        public let projectorType: String
        public let inputDim: Int
        public let nEmbed: Int
        public let depth: Int
        public let mlpRatio: Int
        public let downsampleRatio: Int
        public let tokenPooling: Bool

        enum CodingKeys: String, CodingKey {
            case projectorType = "projector_type"
            case inputDim = "input_dim"
            case nEmbed = "n_embed"
            case depth
            case mlpRatio = "mlp_ratio"
            case downsampleRatio = "downsample_ratio"
            case tokenPooling = "token_pooling"
        }
    }

    public let modelType: String
    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let projectorConfig: ProjectorConfiguration
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let vocabularySize: Int
    public let padId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case projectorConfig = "projector_config"
        case ignoreIndex = "ignore_index"
        case imageTokenIndex = "image_token_index"
        case vocabularySize = "vocab_size"
        case padId = "pad_id"
    }
}

// MARK: - Processor Configuration

public struct DeepSeekVLProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case temporalPatchSize = "temporal_patch_size"
    }
}

// MARK: - Language

private enum Language {
    fileprivate class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            let headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: config.attentionBias)

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rotaryEmbedding(queries, offset: cache.offset)
                keys = rotaryEmbedding(keys, offset: cache.offset)
                (keys, values) = cache.update(keys: keys, values: values)
            } else {
                queries = rotaryEmbedding(queries)
                keys = rotaryEmbedding(keys)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(gate(x)) * up(x))
        }
    }

    fileprivate class DeepSeekBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        let mlp: MLP

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        var layers: [DeepSeekBlock]
        let norm: RMSNorm
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        var headDim: MLX.IntOrPair { .init(config.hiddenSize / config.attentionHeads) }

        let config: DeepSeekVLConfiguration.TextConfiguration

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
            self.config = config
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize
            )

            self.layers = (0..<config.hiddenLayers).map { _ in
                DeepSeekBlock(config)
            }

            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self.kvHeads = (0..<config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            h = norm(h)
            let out = lmHead != nil ? lmHead!(h) : embedTokens.asLinear(h)
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum Vision {
    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, dims * 3, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let qkv = qkv(x)
            let (queries, keys, values) = split(qkv, parts: 3, axis: -1)

            let (B, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2))
            let S = keys.dim(1)

            let queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            let keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
            let values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return proj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.fc1 = Linear(config.width, config.intermediateSize, bias: true)
            self.fc2 = Linear(config.intermediateSize, config.width, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    fileprivate class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(
                dims: config.width, numHeads: config.numAttentionHeads)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.width, eps: config.layerNormEps)
            self.mlp = MLP(config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.width, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var r = attention(layerNorm1(x), mask: mask)
            let h = x + r
            r = mlp(layerNorm2(h))
            return h + r
        }
    }

    fileprivate class VisionEmbeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        let positions: Int
        let positionIds: MLXArray

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.width,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize)
            )

            let d = config.imageSize / config.patchSize
            self.positions = d * d
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: positions,
                dimensions: config.width
            )
            self.positionIds = MLXArray(0..<positions)[.newAxis, 0...]
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patchEmbeddings = patchEmbedding(x)
            patchEmbeddings = patchEmbeddings.flattened(start: 1, end: 2)
            let embeddings = patchEmbeddings + positionEmbedding(positionIds)
            return embeddings
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: [EncoderLayer]
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.embeddings = VisionEmbeddings(config)
            self.encoder = (0..<config.layers).map { _ in EncoderLayer(config) }
            self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.width)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            let x = embeddings(x)

            var encoderStates: [MLXArray]? = outputHiddenStates ? [] : nil
            var h = x
            for layer in encoder {
                h = layer(h)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }

            let poolerOutput = postLayerNorm(h)
            return (poolerOutput, x, encoderStates?.last)
        }
    }
}

// MARK: - Processor

public class DeepSeekVLProcessor: UserInputProcessor {
    private let config: DeepSeekVLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: DeepSeekVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) throws -> LMInput {
        if input.images.isEmpty {
            let prompt = input.prompt.asMessages().last?["content"] ?? ""
            let tokens = try tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(tokens))
        }

        guard input.images.count == 1 else { throw VLMError.singleImageAllowed }

        var image = try input.images[0].asCIImage()
        image = MediaProcessing.inSRGBToneCurveSpace(image)
        image = MediaProcessing.apply(image, processing: input.processing)
        image = MediaProcessing.resampleBicubic(image, to: CGSize(width: config.size.width, height: config.size.height))
        image = MediaProcessing.normalize(image, mean: config.imageMeanTuple, std: config.imageStdTuple)

        var pixels = MediaProcessing.asMLXArray(image)
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }

        let prompt = input.prompt.asMessages().last?["content"] ?? ""
        let promptTokens = try tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: pixels)
        )
    }
}

// MARK: - Model

public class DeepSeekVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var projector: MLPProjector

    public let config: DeepSeekVLConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: DeepSeekVLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
        self._projector.wrappedValue = MLPProjector(config: config.projectorConfig)
    }

    private func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, mask: MLXArray
    ) -> MLXArray {
        guard let pixelValues else {
            return languageModel.embedTokens(inputIds)
        }

        let inputEmbeddings = languageModel.embedTokens(inputIds)
        let (hiddenState, _, _) = visionModel(pixelValues.transposed(0, 2, 3, 1))
        let imageFeatures = projector(hiddenState)

        // Find positions of image tokens and replace them with image features
        let imagePositions = inputIds .== config.imageTokenIndex
        let result = where(imagePositions, imageFeatures, inputEmbeddings)

        return result
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        guard let mask = input.text.mask else { throw VLMError.maskRequired }

        let embeddings = inputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            mask: mask
        )

        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }
}

// MARK: - Projector

private class MLPProjector: Module, UnaryLayer {
    let layers: [Module & UnaryLayer]
    let config: DeepSeekVLConfiguration.ProjectorConfiguration

    init(config: DeepSeekVLConfiguration.ProjectorConfiguration) {
        self.config = config

        if config.projectorType == "mlp_gelu" {
            var modules: [Module & UnaryLayer] = [
                Linear(config.inputDim, config.nEmbed)
            ]

            for _ in 1..<config.depth {
                modules.append(GELU())
                modules.append(Linear(config.nEmbed, config.nEmbed))
            }

            self.layers = modules
        } else if config.projectorType == "downsample_mlp_gelu" {
            let mlpRatio = config.mlpRatio
            var modules: [Module & UnaryLayer] = [
                Linear(
                    config.inputDim * config.downsampleRatio * config.downsampleRatio,
                    config.nEmbed * mlpRatio
                )
            ]

            for _ in 1..<(config.depth - 1) {
                modules.append(GELU())
                modules.append(
                    Linear(
                        config.nEmbed * mlpRatio,
                        config.nEmbed * mlpRatio
                    )
                )
            }

            modules.append(GELU())
            modules.append(
                Linear(
                    config.nEmbed * mlpRatio,
                    config.nEmbed
                )
            )

            self.layers = modules
        } else {
            fatalError("Unsupported projector type: \(config.projectorType)")
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            out = layer(out)
        }
        return out
    }
}
