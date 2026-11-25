# Spatial latent-change cache (idea A)

This prototype partitions the Wan latent/token grid into spatio-temporal blocks,
ranks every block by mean-squared latent change from its own cached reference,
and refreshes whole blocks in score order up to the requested token budget. Conditional and CFG-unconditional passes
share the same selection mask.

## Run

Place `sencache.py`, `spatial_cache.py`, and the official Wan2.1 code in the
same directory, then run for example:

```bash
python sencache.py \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size '832*480' \
  --frame_num 81 \
  --sample_steps 50 \
  --prompt_file ./test_prompt.json \
  --output_dir ./output_spatial \
  --spatial_cache \
  --spatial_cache_ratio 0.25 \
  --spatial_cache_block 1,4,4
```

`--spatial_cache_ratio` is the fraction of blocks selected at each denoising
step. `--spatial_cache_block` is `T,H,W` measured on the DiT token grid. Spatial
mode does not require `sensitivity_wan21.npz`.

## Important limitation

This is an **oracle quality prototype**, not yet a wall-clock acceleration.
Stock Wan self-attention still evaluates the dense token sequence; the code
then merges only selected fresh residual blocks into the cached result. Logs
report the theoretical refreshed-token ratio and explicitly label execution as
`oracle/full-density`. Real speedup requires a partial-query/sparse-attention
implementation in the Wan transformer blocks.

## HybridCache: SenCache with AdaCache-style local refresh

Hybrid mode keeps SenCache's Jacobian sensitivity estimate and replaces its
binary reuse/recompute decision with three levels:

1. sensitivity below the SenCache threshold: reuse the full cached residual;
2. sensitivity between the SenCache threshold and the full threshold: refresh
   the latent blocks with the largest change;
3. sensitivity above the full threshold: run a full refresh.

The middle-level refresh ratio is linearly interpolated from
`--hybrid_min_ratio` to 100%. When SenCache reaches its `K` consecutive-reuse
limit while still below the low threshold, HybridCache performs a minimum-ratio
local refresh instead of a full refresh.

```bash
python sencache.py \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size '832*480' \
  --frame_num 17 \
  --sample_steps 20 \
  --base_seed 42 \
  --prompt_file ./test_prompt.json \
  --output_dir ./output_hybrid \
  --sencache_thresh_start 0.045 \
  --sencache_thresh_main 2.0 \
  --sencache_K 3 \
  --hybrid_cache \
  --hybrid_full_multiplier 2.0 \
  --hybrid_min_ratio 0.10 \
  --spatial_cache_block 1,4,4
```

Full reuse is a real compute skip. Local refresh currently uses the same oracle
implementation described above: it runs dense DiT blocks and then commits only
selected token residuals. Therefore the log reports both actual dense DiT calls
and the theoretical refreshed-token ratio. Do not interpret the latter as a
measured wall-clock speedup until partial-query attention is implemented.

### Experimental partial-query backend

AdaCache itself does **not** use partial-query attention. The official method
caches transformer residuals across complete denoising steps and adapts the
cache interval to the generated video's content (optionally with motion
regularization). Partial-query execution is an additional experimental backend
for this repository's spatial-refresh extension.

Enable it with:

```bash
  --hybrid_cache \
  --partial_query_attention \
  --hybrid_min_ratio 0.25 \
  --spatial_cache_block 1,4,4
```

For each Wan transformer block, selected tokens compute fresh Q, cross-attention
and FFN outputs. K/V still cover the full token sequence so selected queries
retain global context. Unselected tokens advance through cached per-block
residuals. This reduces attention from `L x L` to `M x L` for a local refresh
and reduces Q/cross-attention/FFN work to `M` tokens, while dense K/V projection
remains. It currently supports batch size one.

The SenCache safety rule remains intact: after `K` consecutive full-cache
reuses, HybridCache performs a dense refresh. Partial refresh is used only when
the sensitivity lies strictly between the reuse and full-refresh thresholds.
