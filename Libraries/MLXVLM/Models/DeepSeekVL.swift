// Copyright © 2024 Apple Inc.

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
