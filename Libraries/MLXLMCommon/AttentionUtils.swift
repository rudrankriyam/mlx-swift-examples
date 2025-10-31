import Foundation
import MLX
import MLXFast

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// Apply RoPE with automatic per-batch offset handling
///
/// This function automatically handles BatchKVCache offsets:
/// - For BatchKVCache: applies RoPE per batch element with individual offsets
/// - For regular KVCache: applies RoPE with single offset
/// - For no cache: applies RoPE with offset 0
///
/// Matches Python's `mx.fast.rope()` which accepts array offsets natively.
///
/// **Usage in models:**
/// ```swift
/// queries = applyRoPE(queries, cache: cache, rope: rope)
/// keys = applyRoPE(keys, cache: cache, rope: rope)
/// ```
///
/// - Parameters:
///   - x: Input tensor [B, heads, seq, head_dim]
///   - cache: Cache instance (can be BatchKVCache or regular KVCache)
///   - rope: RoPE function that takes (MLXArray, offset: Int) -> MLXArray
/// - Returns: RoPE-transformed tensor with same shape as input
public func applyRoPE(
    _ x: MLXArray,
    cache: KVCache?,
    rope: (MLXArray, Int) -> MLXArray
) -> MLXArray {
    guard let cache else {
        return rope(x, 0)
    }

    if let batchCache = cache as? BatchKVCache {
        let B = x.dim(0)
        let offsets = batchCache.offsets.asArray(Int.self)
        var batches = [MLXArray]()

        for b in 0..<B {
            let batchX = x[b, 0..., 0..., 0...].expandedDimensions(axis: 0)
            // Clamp offset to 0 - negative offsets from padding can break RoPE
            let clampedOffset = max(0, offsets[b])
            let transformed = rope(batchX, clampedOffset)
            batches.append(transformed)
        }

        return concatenated(batches, axis: 0)
    } else {
        return rope(x, max(0, cache.offset))
    }
}
