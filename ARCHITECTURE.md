# Architecture: plato-tile-ranker

## Language Choice: Rust

### Why Rust over Python (original)

The tile ranker sits in the hot path of every room query. Every tile retrieval triggers a
ranking pass. At fleet scale (10K+ rooms, 100K+ tiles each), this becomes a latency bottleneck.

**Benchmark estimates:**
- Python sorting 100K tiles: ~45ms (dict overhead, GIL, GC pauses)
- Rust sorting 100K tiles: ~2ms (stack f64s, no GC, SIMD-optimized sort)
- **22x speedup** on the ranking hot path

**Memory:**
- Python tile dict: ~120 bytes (id str + 5 fields + dict overhead)
- Rust Tile struct: ~48 bytes (String + 5 f64 + Vec + String + padding)
- **2.5x memory reduction** = more tiles in L3 cache = fewer cache misses

### Why Rust over C

Same raw performance. Rust wins on:
1. **Borrow checker** — no use-after-free in score pipelines
2. **Cargo** — dependency management, cross-compilation, CI
3. **serde** — serialization without hand-writing parsers
4. **WASM** — `cargo build --target wasm32-unknown-unknown` just works
5. **PyO3** — Python bindings with zero-copy when we need them

C would save ~5% compile time. Not worth the debugging cost at fleet scale.

### Why Rust over CUDA

GPU ranking is viable for 500K+ tiles per query:
- PCIe transfer: ~0.5ms for 100K tiles
- GPU sort (thrust): ~0.1ms for 100K tiles
- Total: ~0.6ms vs Rust's ~2ms

But most queries rank <10K tiles, where GPU overhead dominates.
**Decision**: Rust now, optional CUDA backend later behind a `RankBackend` trait.

### Architecture

```
Tile batch → ScoreFunction → BinaryHeap (min-k) → sorted Vec<RankedTile>
                ↓
         WeightedSum {
           recency: exp(-age * ln2 / half_life) * w_recency
           confidence: clamp(min_conf, 1.0) * w_confidence
           relevance: clamp(0, 1) * w_relevance
         }
         + tag_boosts - room_penalty
```

### Key Design Decisions

1. **O(n log k) top-k via min-heap** instead of O(n log n) full sort.
   Most callers want top-10 or top-100 from millions of tiles.

2. **Stack-allocated ScoreBreakdown** — no heap allocation in scoring hot path.

3. **Exponential recency decay** — `exp(-age * ln2 / half_life)` gives intuitive
   "half-life" semantics. Default: 1 day half-life.

4. **f64 throughout** — constraint theory teaches us that float precision matters.
   We use f64 (not f32) for scoring to avoid precision loss in weighted sums.

### Future: CUDA Backend

```rust
trait RankBackend {
    fn score_batch(&self, tiles: &[Tile], now: f64) -> Vec<f64>;
}

struct CpuRanker { config: RankConfig }
struct GpuRanker { config: RankConfig, stream: cuda::Stream }
```

This lets us swap backends without changing the ranking API.
