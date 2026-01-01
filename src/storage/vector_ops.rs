//! Optimized vector math utilities for embedding operations.
//!
//! This module provides SIMD-friendly, high-performance vector operations
//! optimized for common embedding dimensions (768, 1536, 3072).
//!
//! ## Design Principles
//!
//! 1. **SIMD-Friendly**: Loop structures designed for auto-vectorization
//! 2. **Cache-Efficient**: Sequential memory access patterns
//! 3. **Batch Operations**: Amortized overhead for bulk operations
//! 4. **Zero Allocations**: In-place operations where possible
//!
//! ## Common Embedding Dimensions
//!
//! | Model | Dimension |
//! |-------|-----------|
//! | OpenAI text-embedding-3-small | 1536 |
//! | OpenAI text-embedding-3-large | 3072 |
//! | BGE-M3 | 1024 |
//! | E5-small-v2 | 384 |
//! | Cohere embed-v3 | 1024 |
//! | all-MiniLM-L6-v2 | 384 |
//! | sentence-transformers | 768 |
//!
//! ## Performance Notes
//!
//! - For best SIMD performance, compile with: `RUSTFLAGS="-C target-cpu=native"`
//! - The batch operations use parallel processing via `rayon` for large candidate sets
//! - All operations avoid heap allocations in hot paths where possible

use rayon::prelude::*;

/// Compute the dot product of two vectors.
///
/// This is the core operation for similarity computations. The implementation
/// is designed for SIMD auto-vectorization.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// The dot product (sum of element-wise products)
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::dot_product;
///
/// let a = vec![1.0_f32, 2.0, 3.0];
/// let b = vec![4.0_f32, 5.0, 6.0];
/// let result = dot_product(&a, &b);
/// assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    // SIMD-friendly loop: sequential access, no branches, accumulator pattern
    // Modern compilers will auto-vectorize this with proper flags
    let mut sum = 0.0f32;

    // Process in chunks of 4 for better SIMD utilization
    // The compiler can recognize this pattern for AVX/SSE vectorization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    // Main vectorizable loop
    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base];
        sum += a[base + 1] * b[base + 1];
        sum += a[base + 2] * b[base + 2];
        sum += a[base + 3] * b[base + 3];
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

/// Compute the L2 (squared) norm of a vector.
///
/// Returns the sum of squared elements (not the square root).
/// Use this for comparisons where only relative ordering matters.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Sum of squared elements
#[inline]
fn l2_norm_squared(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let chunks = v.len() / 4;
    let remainder = v.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        sum += v[base] * v[base];
        sum += v[base + 1] * v[base + 1];
        sum += v[base + 2] * v[base + 2];
        sum += v[base + 3] * v[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        sum += v[base + i] * v[base + i];
    }

    sum
}

/// Compute cosine similarity between two vectors.
///
/// Cosine similarity measures the angle between two vectors, ranging from
/// -1 (opposite) through 0 (orthogonal) to 1 (identical direction).
///
/// Formula: cos(theta) = (a . b) / (||a|| * ||b||)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity in range [-1.0, 1.0], or 0.0 if either vector is zero.
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::cosine_similarity;
///
/// let a = vec![1.0_f32, 0.0, 0.0];
/// let b = vec![0.0_f32, 1.0, 0.0];
/// let similarity = cosine_similarity(&a, &b);
/// assert!((similarity - 0.0).abs() < 1e-6); // Orthogonal vectors
///
/// let c = vec![1.0_f32, 0.0, 0.0];
/// let similarity = cosine_similarity(&a, &c);
/// assert!((similarity - 1.0).abs() < 1e-6); // Identical vectors
/// ```
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    // Compute dot product and norms in a single pass for cache efficiency
    // This is faster than separate dot_product and l2_norm calls
    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    // Main vectorizable loop - all three accumulations in one pass
    for i in 0..chunks {
        let base = i * 4;

        // Unroll 4x for better instruction-level parallelism
        let a0 = a[base];
        let a1 = a[base + 1];
        let a2 = a[base + 2];
        let a3 = a[base + 3];

        let b0 = b[base];
        let b1 = b[base + 1];
        let b2 = b[base + 2];
        let b3 = b[base + 3];

        dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a_sq += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b_sq += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let ai = a[base + i];
        let bi = b[base + i];
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute L2 (Euclidean) distance between two vectors.
///
/// L2 distance is the straight-line distance in n-dimensional space.
///
/// Formula: d = sqrt(sum((a[i] - b[i])^2))
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Non-negative Euclidean distance
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::l2_distance;
///
/// let a = vec![0.0_f32, 0.0, 0.0];
/// let b = vec![3.0_f32, 4.0, 0.0];
/// let distance = l2_distance(&a, &b);
/// assert!((distance - 5.0).abs() < 1e-6); // 3-4-5 triangle
/// ```
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let mut sum = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    // Main vectorizable loop
    for i in 0..chunks {
        let base = i * 4;

        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];

        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum.sqrt()
}

/// Compute squared L2 distance (avoids sqrt for comparison operations).
///
/// Use this when you only need to compare distances, not the actual values.
/// Avoiding the sqrt operation provides a measurable performance improvement.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Squared Euclidean distance
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let mut sum = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;

        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];

        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

/// Normalize a vector in-place to unit length.
///
/// After normalization, the vector will have L2 norm of 1.0.
/// If the input is a zero vector, it remains unchanged.
///
/// # Arguments
/// * `v` - Vector to normalize (modified in place)
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::normalize_inplace;
///
/// let mut v = vec![3.0_f32, 4.0];
/// normalize_inplace(&mut v);
/// assert!((v[0] - 0.6).abs() < 1e-6);
/// assert!((v[1] - 0.8).abs() < 1e-6);
/// ```
#[inline]
pub fn normalize_inplace(v: &mut [f32]) {
    let norm_sq = l2_norm_squared(v);

    if norm_sq == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm_sq.sqrt();

    // SIMD-friendly in-place update
    let chunks = v.len() / 4;
    let remainder = v.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        v[base] *= inv_norm;
        v[base + 1] *= inv_norm;
        v[base + 2] *= inv_norm;
        v[base + 3] *= inv_norm;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        v[base + i] *= inv_norm;
    }
}

/// Normalize a vector, returning a new vector with unit length.
///
/// The original vector is not modified.
///
/// # Arguments
/// * `v` - Vector to normalize
///
/// # Returns
/// New vector with L2 norm of 1.0, or copy of zero vector if input is zero.
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::normalize;
///
/// let v = vec![3.0_f32, 4.0];
/// let normalized = normalize(&v);
/// assert!((normalized[0] - 0.6).abs() < 1e-6);
/// assert!((normalized[1] - 0.8).abs() < 1e-6);
/// // Original unchanged
/// assert!((v[0] - 3.0).abs() < 1e-6);
/// ```
#[inline]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm_sq = l2_norm_squared(v);

    if norm_sq == 0.0 {
        return v.to_vec();
    }

    let inv_norm = 1.0 / norm_sq.sqrt();

    // Use parallel processing for large vectors
    if v.len() >= 1024 {
        v.par_iter().map(|x| x * inv_norm).collect()
    } else {
        v.iter().map(|x| x * inv_norm).collect()
    }
}

/// Batch cosine similarity: compute similarity between query and many candidates.
///
/// This is the most common retrieval operation. The implementation uses:
/// - Parallel processing for large candidate sets (via rayon)
/// - Cache-efficient memory access patterns
/// - Partial sorting for top-k extraction (O(n) instead of O(n log n))
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `candidates` - Slice of candidate embedding vectors
/// * `top_k` - Number of top results to return
///
/// # Returns
/// Vector of (index, similarity) tuples sorted by similarity descending.
/// Returns at most `top_k` results.
///
/// # Panics
/// Panics if any candidate has a different dimension than the query.
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::batch_cosine_similarity;
///
/// let query = vec![1.0_f32, 0.0, 0.0];
/// let candidates = vec![
///     vec![1.0_f32, 0.0, 0.0],  // Most similar
///     vec![0.5_f32, 0.5, 0.0],  // Partially similar
///     vec![0.0_f32, 1.0, 0.0],  // Orthogonal
/// ];
/// let results = batch_cosine_similarity(&query, &candidates, 2);
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].0, 0); // Index of most similar
/// assert!((results[0].1 - 1.0).abs() < 1e-6);
/// ```
pub fn batch_cosine_similarity(
    query: &[f32],
    candidates: &[Vec<f32>],
    top_k: usize,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let top_k = top_k.min(candidates.len());

    // Pre-compute query norm once
    let query_norm_sq = l2_norm_squared(query);
    if query_norm_sq == 0.0 {
        return (0..top_k).map(|i| (i, 0.0)).collect();
    }
    let query_norm = query_norm_sq.sqrt();

    // Threshold for parallel processing
    const PARALLEL_THRESHOLD: usize = 256;

    let scores: Vec<(usize, f32)> = if candidates.len() >= PARALLEL_THRESHOLD {
        // Parallel computation for large candidate sets
        candidates
            .par_iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let similarity = cosine_similarity_with_query_norm(query, candidate, query_norm);
                (idx, similarity)
            })
            .collect()
    } else {
        // Sequential for small sets (avoids thread pool overhead)
        candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let similarity = cosine_similarity_with_query_norm(query, candidate, query_norm);
                (idx, similarity)
            })
            .collect()
    };

    // Use partial sort for top-k extraction
    select_top_k(scores, top_k)
}

/// Cosine similarity with pre-computed query norm.
///
/// Internal helper for batch operations where query norm is computed once.
#[inline]
fn cosine_similarity_with_query_norm(query: &[f32], candidate: &[f32], query_norm: f32) -> f32 {
    assert_eq!(
        query.len(),
        candidate.len(),
        "Vector dimensions must match"
    );

    let mut dot = 0.0f32;
    let mut candidate_norm_sq = 0.0f32;

    let chunks = query.len() / 4;
    let remainder = query.len() % 4;

    for i in 0..chunks {
        let base = i * 4;

        let q0 = query[base];
        let q1 = query[base + 1];
        let q2 = query[base + 2];
        let q3 = query[base + 3];

        let c0 = candidate[base];
        let c1 = candidate[base + 1];
        let c2 = candidate[base + 2];
        let c3 = candidate[base + 3];

        dot += q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3;
        candidate_norm_sq += c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let qi = query[base + i];
        let ci = candidate[base + i];
        dot += qi * ci;
        candidate_norm_sq += ci * ci;
    }

    let candidate_norm = candidate_norm_sq.sqrt();

    if candidate_norm == 0.0 {
        return 0.0;
    }

    dot / (query_norm * candidate_norm)
}

/// Select top-k elements from a vector by score.
///
/// Uses a partial sort algorithm that is O(n + k log k) instead of O(n log n).
#[inline]
fn select_top_k(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    if k >= scores.len() {
        // Just sort everything
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        return scores;
    }

    // Partial sort: bring top k elements to the front
    // This uses a selection algorithm internally
    scores.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate and sort the top k
    scores.truncate(k);
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scores
}

/// Batch L2 distance: compute distance between query and many candidates.
///
/// Similar to `batch_cosine_similarity` but for L2 distance.
/// Results are sorted by distance ascending (closest first).
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `candidates` - Slice of candidate embedding vectors
/// * `top_k` - Number of closest results to return
///
/// # Returns
/// Vector of (index, distance) tuples sorted by distance ascending.
pub fn batch_l2_distance(
    query: &[f32],
    candidates: &[Vec<f32>],
    top_k: usize,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let top_k = top_k.min(candidates.len());

    const PARALLEL_THRESHOLD: usize = 256;

    let scores: Vec<(usize, f32)> = if candidates.len() >= PARALLEL_THRESHOLD {
        candidates
            .par_iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let distance = l2_distance(query, candidate);
                (idx, distance)
            })
            .collect()
    } else {
        candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let distance = l2_distance(query, candidate);
                (idx, distance)
            })
            .collect()
    };

    // Select top-k by ascending distance (smallest first)
    select_bottom_k(scores, top_k)
}

/// Select bottom-k elements (smallest scores first).
#[inline]
fn select_bottom_k(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    if k >= scores.len() {
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        return scores;
    }

    scores.select_nth_unstable_by(k - 1, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    scores.truncate(k);
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    scores
}

/// Average multiple embeddings to create a centroid.
///
/// This is commonly used for:
/// - RAPTOR tree node summarization
/// - Cluster centroid computation
/// - Query expansion averaging
///
/// # Arguments
/// * `embeddings` - Slice of embedding slices to average
///
/// # Returns
/// New vector representing the centroid (element-wise mean).
/// Returns an empty vector if input is empty.
///
/// # Panics
/// Panics if embeddings have different dimensions.
///
/// # Example
/// ```
/// use reasonkit_mem::storage::vector_ops::average_embeddings;
///
/// let e1 = [1.0_f32, 2.0, 3.0];
/// let e2 = [3.0_f32, 4.0, 5.0];
/// let embeddings: Vec<&[f32]> = vec![&e1, &e2];
/// let centroid = average_embeddings(&embeddings);
/// assert!((centroid[0] - 2.0).abs() < 1e-6);
/// assert!((centroid[1] - 3.0).abs() < 1e-6);
/// assert!((centroid[2] - 4.0).abs() < 1e-6);
/// ```
pub fn average_embeddings(embeddings: &[&[f32]]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let count = embeddings.len() as f32;

    // Verify all have same dimension
    for (i, emb) in embeddings.iter().enumerate().skip(1) {
        assert_eq!(
            emb.len(),
            dim,
            "Embedding {} has dimension {} but expected {}",
            i,
            emb.len(),
            dim
        );
    }

    // Initialize accumulator
    let mut result = vec![0.0f32; dim];

    // Sum all embeddings
    for emb in embeddings {
        for (i, &val) in emb.iter().enumerate() {
            result[i] += val;
        }
    }

    // Divide by count
    let inv_count = 1.0 / count;
    for val in &mut result {
        *val *= inv_count;
    }

    result
}

/// Weighted average of multiple embeddings.
///
/// Similar to `average_embeddings` but allows specifying weights for each embedding.
///
/// # Arguments
/// * `embeddings` - Slice of embedding slices
/// * `weights` - Weights for each embedding (will be normalized internally)
///
/// # Returns
/// Weighted centroid vector.
///
/// # Panics
/// Panics if embeddings and weights have different lengths, or if embeddings have different dimensions.
pub fn weighted_average_embeddings(embeddings: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    assert_eq!(
        embeddings.len(),
        weights.len(),
        "Embeddings and weights must have same length"
    );

    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let weight_sum: f32 = weights.iter().sum();

    if weight_sum == 0.0 {
        return vec![0.0f32; dim];
    }

    let mut result = vec![0.0f32; dim];

    for (emb, &weight) in embeddings.iter().zip(weights.iter()) {
        assert_eq!(emb.len(), dim, "All embeddings must have same dimension");
        let normalized_weight = weight / weight_sum;
        for (i, &val) in emb.iter().enumerate() {
            result[i] += val * normalized_weight;
        }
    }

    result
}

/// Add two vectors element-wise.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// New vector where result[i] = a[i] + b[i]
#[inline]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Subtract two vectors element-wise.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector (subtracted from a)
///
/// # Returns
/// New vector where result[i] = a[i] - b[i]
#[inline]
pub fn subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Scale a vector by a scalar value.
///
/// # Arguments
/// * `v` - Vector to scale
/// * `scalar` - Scaling factor
///
/// # Returns
/// New vector where result[i] = v[i] * scalar
#[inline]
pub fn scale(v: &[f32], scalar: f32) -> Vec<f32> {
    v.iter().map(|&x| x * scalar).collect()
}

/// Scale a vector in-place by a scalar value.
///
/// # Arguments
/// * `v` - Vector to scale (modified in place)
/// * `scalar` - Scaling factor
#[inline]
pub fn scale_inplace(v: &mut [f32], scalar: f32) {
    for x in v.iter_mut() {
        *x *= scalar;
    }
}

/// Compute the L2 norm (magnitude) of a vector.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Non-negative L2 norm
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    l2_norm_squared(v).sqrt()
}

/// Check if a vector is normalized (L2 norm approximately 1.0).
///
/// # Arguments
/// * `v` - Vector to check
/// * `tolerance` - Acceptable deviation from 1.0
///
/// # Returns
/// true if |norm(v) - 1.0| < tolerance
#[inline]
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    let norm = l2_norm(v);
    (norm - 1.0).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ========================================
    // Dot Product Tests
    // ========================================

    #[test]
    fn test_dot_product_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!(approx_eq(result, 32.0)); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = dot_product(&a, &b);
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn test_dot_product_self() {
        let a = vec![3.0, 4.0];
        let result = dot_product(&a, &a);
        assert!(approx_eq(result, 25.0)); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_dot_product_large_dimension() {
        // Test with 768-dim vectors (common embedding size)
        let a: Vec<f32> = (0..768).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..768).map(|i| (768 - i) as f32 * 0.01).collect();
        let result = dot_product(&a, &b);
        assert!(result > 0.0); // Just verify it runs
    }

    #[test]
    fn test_dot_product_1536_dimension() {
        // OpenAI text-embedding-3-small dimension
        let a: Vec<f32> = (0..1536).map(|i| (i as f32 / 1536.0)).collect();
        let b: Vec<f32> = (0..1536).map(|i| (1.0 - i as f32 / 1536.0)).collect();
        let _ = dot_product(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match")]
    fn test_dot_product_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        dot_product(&a, &b);
    }

    // ========================================
    // Cosine Similarity Tests
    // ========================================

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &a);
        assert!(approx_eq(result, 1.0));
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!(approx_eq(result, -1.0));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &zero), 0.0));
        assert!(approx_eq(cosine_similarity(&zero, &a), 0.0));
        assert!(approx_eq(cosine_similarity(&zero, &zero), 0.0));
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        // For normalized vectors, cosine similarity equals dot product
        let mut a = vec![3.0, 4.0];
        let mut b = vec![4.0, 3.0];
        normalize_inplace(&mut a);
        normalize_inplace(&mut b);

        let cos_sim = cosine_similarity(&a, &b);
        let dot = dot_product(&a, &b);
        assert!(approx_eq(cos_sim, dot));
    }

    #[test]
    fn test_cosine_similarity_scaling_invariance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let a_scaled = scale(&a, 10.0);
        let b_scaled = scale(&b, 0.1);

        let result1 = cosine_similarity(&a, &b);
        let result2 = cosine_similarity(&a_scaled, &b_scaled);
        assert!(approx_eq(result1, result2));
    }

    // ========================================
    // L2 Distance Tests
    // ========================================

    #[test]
    fn test_l2_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let result = l2_distance(&a, &a);
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn test_l2_distance_345_triangle() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = l2_distance(&a, &b);
        assert!(approx_eq(result, 5.0));
    }

    #[test]
    fn test_l2_distance_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d1 = l2_distance(&a, &b);
        let d2 = l2_distance(&b, &a);
        assert!(approx_eq(d1, d2));
    }

    #[test]
    fn test_l2_distance_squared() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = l2_distance_squared(&a, &b);
        assert!(approx_eq(result, 25.0)); // 3^2 + 4^2 = 25
    }

    // ========================================
    // Normalization Tests
    // ========================================

    #[test]
    fn test_normalize_inplace() {
        let mut v = vec![3.0, 4.0];
        normalize_inplace(&mut v);
        assert!(approx_eq(v[0], 0.6));
        assert!(approx_eq(v[1], 0.8));
        assert!(approx_eq(l2_norm(&v), 1.0));
    }

    #[test]
    fn test_normalize_inplace_zero() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize_inplace(&mut v);
        assert!(approx_eq(v[0], 0.0));
        assert!(approx_eq(v[1], 0.0));
        assert!(approx_eq(v[2], 0.0));
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = normalize(&v);
        assert!(approx_eq(normalized[0], 0.6));
        assert!(approx_eq(normalized[1], 0.8));
        // Original unchanged
        assert!(approx_eq(v[0], 3.0));
        assert!(approx_eq(v[1], 4.0));
    }

    #[test]
    fn test_normalize_large_vector() {
        let v: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let normalized = normalize(&v);
        assert!(approx_eq(l2_norm(&normalized), 1.0));
    }

    #[test]
    fn test_is_normalized() {
        let mut v = vec![3.0, 4.0];
        assert!(!is_normalized(&v, 0.01));
        normalize_inplace(&mut v);
        assert!(is_normalized(&v, 0.01));
    }

    // ========================================
    // Batch Operations Tests
    // ========================================

    #[test]
    fn test_batch_cosine_similarity_empty() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<Vec<f32>> = vec![];
        let results = batch_cosine_similarity(&query, &candidates, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_cosine_similarity_basic() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0], // Most similar (1.0)
            vec![0.0, 1.0, 0.0], // Orthogonal (0.0)
            vec![-1.0, 0.0, 0.0], // Opposite (-1.0)
        ];

        let results = batch_cosine_similarity(&query, &candidates, 3);
        assert_eq!(results.len(), 3);

        // Should be sorted by similarity descending
        assert_eq!(results[0].0, 0); // Index of most similar
        assert!(approx_eq(results[0].1, 1.0));
        assert_eq!(results[1].0, 1); // Index of orthogonal
        assert!(approx_eq(results[1].1, 0.0));
        assert_eq!(results[2].0, 2); // Index of opposite
        assert!(approx_eq(results[2].1, -1.0));
    }

    #[test]
    fn test_batch_cosine_similarity_top_k() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::PI / 200.0;
                vec![angle.cos(), angle.sin(), 0.0]
            })
            .collect();

        let results = batch_cosine_similarity(&query, &candidates, 5);
        assert_eq!(results.len(), 5);

        // Results should be sorted descending by similarity
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_batch_cosine_similarity_top_k_exceeds_candidates() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let results = batch_cosine_similarity(&query, &candidates, 10);
        assert_eq!(results.len(), 2); // Should return all available
    }

    #[test]
    fn test_batch_cosine_similarity_large_parallel() {
        // Test with enough candidates to trigger parallel processing
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<Vec<f32>> = (0..1000).map(|_| vec![1.0, 0.5, 0.0]).collect();

        let results = batch_cosine_similarity(&query, &candidates, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_batch_l2_distance_basic() {
        let query = vec![0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0], // Distance 1.0
            vec![3.0, 4.0], // Distance 5.0
            vec![0.0, 0.5], // Distance 0.5 (closest)
        ];

        let results = batch_l2_distance(&query, &candidates, 3);
        assert_eq!(results.len(), 3);

        // Should be sorted by distance ascending (closest first)
        assert_eq!(results[0].0, 2); // Index of closest
        assert!(approx_eq(results[0].1, 0.5));
    }

    // ========================================
    // Average Embeddings Tests
    // ========================================

    #[test]
    fn test_average_embeddings_empty() {
        let embeddings: Vec<&[f32]> = vec![];
        let result = average_embeddings(&embeddings);
        assert!(result.is_empty());
    }

    #[test]
    fn test_average_embeddings_single() {
        let e1 = [1.0, 2.0, 3.0];
        let embeddings: Vec<&[f32]> = vec![&e1];
        let result = average_embeddings(&embeddings);
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 2.0));
        assert!(approx_eq(result[2], 3.0));
    }

    #[test]
    fn test_average_embeddings_multiple() {
        let e1 = [1.0, 2.0, 3.0];
        let e2 = [3.0, 4.0, 5.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2];
        let result = average_embeddings(&embeddings);
        assert!(approx_eq(result[0], 2.0)); // (1+3)/2
        assert!(approx_eq(result[1], 3.0)); // (2+4)/2
        assert!(approx_eq(result[2], 4.0)); // (3+5)/2
    }

    #[test]
    fn test_average_embeddings_three() {
        let e1 = [0.0, 0.0, 0.0];
        let e2 = [3.0, 6.0, 9.0];
        let e3 = [6.0, 12.0, 18.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2, &e3];
        let result = average_embeddings(&embeddings);
        assert!(approx_eq(result[0], 3.0)); // (0+3+6)/3
        assert!(approx_eq(result[1], 6.0)); // (0+6+12)/3
        assert!(approx_eq(result[2], 9.0)); // (0+9+18)/3
    }

    #[test]
    #[should_panic(expected = "has dimension")]
    fn test_average_embeddings_dimension_mismatch() {
        let e1 = [1.0, 2.0, 3.0];
        let e2 = [1.0, 2.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2];
        average_embeddings(&embeddings);
    }

    #[test]
    fn test_weighted_average_embeddings() {
        let e1 = [1.0, 0.0];
        let e2 = [0.0, 1.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2];
        let weights = [3.0, 1.0]; // e1 has 3x the weight

        let result = weighted_average_embeddings(&embeddings, &weights);
        assert!(approx_eq(result[0], 0.75)); // (3*1 + 1*0) / 4
        assert!(approx_eq(result[1], 0.25)); // (3*0 + 1*1) / 4
    }

    #[test]
    fn test_weighted_average_zero_weights() {
        let e1 = [1.0, 2.0];
        let e2 = [3.0, 4.0];
        let embeddings: Vec<&[f32]> = vec![&e1, &e2];
        let weights = [0.0, 0.0];

        let result = weighted_average_embeddings(&embeddings, &weights);
        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[1], 0.0));
    }

    // ========================================
    // Vector Arithmetic Tests
    // ========================================

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = add(&a, &b);
        assert!(approx_eq(result[0], 5.0));
        assert!(approx_eq(result[1], 7.0));
        assert!(approx_eq(result[2], 9.0));
    }

    #[test]
    fn test_subtract() {
        let a = vec![4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = subtract(&a, &b);
        assert!(approx_eq(result[0], 3.0));
        assert!(approx_eq(result[1], 3.0));
        assert!(approx_eq(result[2], 3.0));
    }

    #[test]
    fn test_scale() {
        let v = vec![1.0, 2.0, 3.0];
        let result = scale(&v, 2.0);
        assert!(approx_eq(result[0], 2.0));
        assert!(approx_eq(result[1], 4.0));
        assert!(approx_eq(result[2], 6.0));
    }

    #[test]
    fn test_scale_inplace() {
        let mut v = vec![1.0, 2.0, 3.0];
        scale_inplace(&mut v, 2.0);
        assert!(approx_eq(v[0], 2.0));
        assert!(approx_eq(v[1], 4.0));
        assert!(approx_eq(v[2], 6.0));
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!(approx_eq(l2_norm(&v), 5.0));
    }

    // ========================================
    // Edge Cases and Stress Tests
    // ========================================

    #[test]
    fn test_single_element_vectors() {
        let a = vec![5.0];
        let b = vec![3.0];
        assert!(approx_eq(dot_product(&a, &b), 15.0));
        assert!(approx_eq(cosine_similarity(&a, &b), 1.0));
        assert!(approx_eq(l2_distance(&a, &b), 2.0));
    }

    #[test]
    fn test_large_values() {
        let a = vec![1e10, 1e10];
        let b = vec![1e10, 1e10];
        let cos_sim = cosine_similarity(&a, &b);
        assert!(approx_eq(cos_sim, 1.0));
    }

    #[test]
    fn test_small_values() {
        let a = vec![1e-10, 1e-10];
        let b = vec![1e-10, 1e-10];
        let cos_sim = cosine_similarity(&a, &b);
        assert!(approx_eq(cos_sim, 1.0));
    }

    #[test]
    fn test_negative_values() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 1.0));
    }

    #[test]
    fn test_mixed_sign_values() {
        let a = vec![1.0, -1.0, 1.0];
        let b = vec![-1.0, 1.0, -1.0];
        assert!(approx_eq(cosine_similarity(&a, &b), -1.0));
    }

    #[test]
    fn test_performance_1536_dimensions() {
        // OpenAI text-embedding-3-small dimension
        let a: Vec<f32> = (0..1536).map(|i| (i as f32 / 1000.0).sin()).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i as f32 / 1000.0).cos()).collect();

        // These should complete quickly without issues
        let _ = dot_product(&a, &b);
        let _ = cosine_similarity(&a, &b);
        let _ = l2_distance(&a, &b);
        let _ = normalize(&a);
    }

    #[test]
    fn test_performance_3072_dimensions() {
        // OpenAI text-embedding-3-large dimension
        let a: Vec<f32> = (0..3072).map(|i| (i as f32 / 1000.0).sin()).collect();
        let b: Vec<f32> = (0..3072).map(|i| (i as f32 / 1000.0).cos()).collect();

        let _ = dot_product(&a, &b);
        let _ = cosine_similarity(&a, &b);
        let _ = l2_distance(&a, &b);
        let _ = normalize(&a);
    }

    #[test]
    fn test_batch_with_varying_similarities() {
        let query = vec![1.0, 0.0, 0.0];

        // Create candidates with known similarities
        let candidates = vec![
            vec![1.0, 0.0, 0.0],         // similarity = 1.0
            vec![0.707, 0.707, 0.0],     // similarity ~ 0.707
            vec![0.0, 1.0, 0.0],         // similarity = 0.0
            vec![-0.707, 0.707, 0.0],    // similarity ~ -0.707
            vec![-1.0, 0.0, 0.0],        // similarity = -1.0
        ];

        let results = batch_cosine_similarity(&query, &candidates, 5);

        // Verify correct ordering (descending by similarity)
        assert_eq!(results[0].0, 0);
        assert_eq!(results[4].0, 4);
    }

    // ========================================
    // Consistency Tests
    // ========================================

    #[test]
    fn test_normalize_preserves_direction() {
        let original = vec![3.0, 4.0, 5.0];
        let normalized = normalize(&original);

        // Cosine similarity between original and normalized should be 1.0
        assert!(approx_eq(cosine_similarity(&original, &normalized), 1.0));
    }

    #[test]
    fn test_cosine_vs_normalized_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let cos_sim = cosine_similarity(&a, &b);

        let norm_a = normalize(&a);
        let norm_b = normalize(&b);
        let dot = dot_product(&norm_a, &norm_b);

        assert!(approx_eq(cos_sim, dot));
    }

    #[test]
    fn test_triangle_inequality() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];

        let ab = l2_distance(&a, &b);
        let bc = l2_distance(&b, &c);
        let ac = l2_distance(&a, &c);

        // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert!(ac <= ab + bc + EPSILON);
    }
}
