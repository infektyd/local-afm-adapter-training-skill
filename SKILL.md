---
name: local-afm-adapter-training
description: Use when training, debugging, exporting, or applying Apple Foundation Models adapters locally on memory-constrained Macs or when AFM toolkit, .fmadapter, QLoRA, dataset schema, launchd, or MLX memory issues appear.
---

# Local AFM Adapter Training

## Principle

Treat local Apple Foundation Models adapter work as a constrained systems task: prove the toolkit, data, memory profile, and export path with tiny runs before spending hours on a full train.

## Workflow

1. **Establish the toolkit boundary**
   - Locate the AFM adapter toolkit, examples, scripts, and expected Python version before editing.
   - Prefer vendor examples and their config schema over inventing new flags.
   - Record exact paths and versions in the user-facing summary.

2. **Validate data shape early**
   - Keep examples compact. Start with tens of records, then hundreds, then full data.
   - Use the toolkit's native schema when available. A common chat-style JSONL pattern is:

```jsonl
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

   - Run a schema/sample loader before training. Reject empty answers, huge records, inconsistent roles, and accidental secrets.
   - Split train/eval deterministically if no split exists.

3. **Load for memory safety**
   - Assume unified-memory Macs can OOM or freeze long before disk or CPU look stressed.
   - Prefer QLoRA or the toolkit's lowest-memory adapter path first: 4-bit/8-bit base loading, LoRA-only trainable weights, small rank, short sequence length, micro-batch 1, gradient accumulation.
   - Avoid eager full-model materialization, high `num_workers`, large eval batches, and long context until a smoke run succeeds.
   - Watch `memory_pressure`, Activity Monitor, logs, and process RSS during first runs.

### Memory-Constrained Mac Defaults

On 16 GB Apple Silicon, start with these settings and only loosen one value at a time after a successful smoke run:

| Setting | Conservative default | Why |
| --- | --- | --- |
| `OMP_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` / `MKL_NUM_THREADS` | `2` | Avoid CPU thread pressure while MPS is paging unified memory. |
| `TOKENIZERS_PARALLELISM` | `false` | Prevent tokenizer worker churn during tiny runs. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Allow unsupported MPS ops to fall back instead of crashing. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.70`-`0.72` | Keep a hard ceiling below system-danger levels. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `0.60`-`0.62` | Encourage earlier MPS cache release. |
| batch size | `1` | Do not raise until full pipeline is proven. |
| train/eval limits | `1`/`1` first | Proves load, labels, checkpoint, export. |
| max sequence length | `384` before `512` | 256 often truncates labels; 512 may OOM. Verify label tokens. |
| dataloader workers | `0` | Avoid extra processes and hidden RAM. |

Never set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` as a casual workaround; that disables the cap and can make the desktop unstable.

Stop immediately if free pages collapse before the first batch, if Activity Monitor shows memory pressure rising fast, or if the log has not progressed beyond base checkpoint load for several minutes. Capture the stack/log and reduce loader allocation, sequence length, sample count, rank, or evaluation before retrying.

4. **Smoke-first training**
   - Run the smallest useful command first: tiny dataset, tiny max steps, checkpoint every few steps, eval disabled or tiny.
   - Example pattern, adapt to the real toolkit:

```bash
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.72 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.62 \
./train-adapter.sh \
  --run-dir runs/smoke \
  --train-limit 1 --eval-limit 1 \
  --batch-size 1 --max-sequence-length 384
```

   - Only scale one dimension at a time: records, sequence length, LoRA rank, steps, or eval.

5. **Export and apply**
   - Export using the toolkit's supported `.fmadapter` path, then verify the file exists and is non-trivial in size.
   - Keep a small "known prompt" check for before/after behavior.
   - For local apps, apply adapters via the documented environment variable or launch mechanism. For GUI apps on macOS, prefer `launchctl setenv NAME /absolute/path/model.fmadapter` plus app restart when the runtime reads environment from launchd.
   - Confirm the target process actually sees the env var; shell exports do not automatically reach already-running GUI apps.

## Safety Gates

- Do not train on private data unless the user explicitly says it is intended.
- Do not run full training before smoke data, schema validation, and a memory-safe config pass.
- Do not overwrite prior adapters or checkpoints; write dated or named run directories.
- If the machine becomes sluggish, stop training, capture logs, reduce context/batch/rank/workers, and resume from a smaller checkpoint.

## Completion Evidence

Report changed commands/configs, dataset counts, smoke-run result, peak-memory clues if available, export path, adapter size, and how the adapter was applied.
