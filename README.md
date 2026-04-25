# Local AFM Adapter Training Skill

A reusable agent skill for training, debugging, exporting, and applying Apple
Foundation Models adapters on memory-constrained Macs.

The skill was written after a real local adapter-training session where the
limiting factor was not code structure, but unified-memory pressure: Apple
Silicon MPS could push the desktop close to instability before the first
training batch. The resulting workflow emphasizes smoke-first training,
conservative defaults, explicit stop conditions, and evidence-based verification
before scaling up.

## Who This Is For

Use this skill when an agent is helping with:

- Apple Foundation Models adapter training toolkit workflows
- `.fmadapter` export and local application
- QLoRA or low-memory adapter experiments
- MPS out-of-memory errors on Apple Silicon
- dataset schema and token-limit issues
- launchd or environment-variable adapter loading
- replacing local MLX/Gemma-style extractors with Apple Foundation Models

It is intentionally not tied to one project. The examples are generic enough for
any local-first AFM adapter workflow.

## What The Skill Teaches Agents

- Treat AFM adapter work as constrained systems work, not just model training.
- Prove toolkit, data, memory, checkpoint, export, and apply steps with tiny
  smoke runs before full training.
- Start 16 GB Apple Silicon runs with conservative settings:
  `batch-size=1`, `train/eval=1/1`, `max-sequence-length=384`,
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.70-0.72`, and CPU thread caps.
- Never use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` casually, because it disables
  the MPS cap and can make the desktop unstable.
- Stop and reduce memory pressure when free pages collapse before the first
  batch, instead of pushing through.
- Verify exported adapters and confirm the target process actually sees adapter
  environment variables.

## Install

For Codex:

```bash
mkdir -p "$HOME/.codex/skills/local-afm-adapter-training"
cp SKILL.md "$HOME/.codex/skills/local-afm-adapter-training/SKILL.md"
```

For other agent runtimes that read `~/.agents/skills`:

```bash
mkdir -p "$HOME/.agents/skills/local-afm-adapter-training"
cp SKILL.md "$HOME/.agents/skills/local-afm-adapter-training/SKILL.md"
```

## Repository Contents

```text
SKILL.md    The reusable agent skill
README.md  Human-facing overview and install notes
LICENSE    MIT license
```

## Notes

The skill does not include Apple toolkit assets, model weights, datasets,
checkpoints, or adapter binaries. It is process guidance for agents operating in
local environments where the user already has lawful access to the relevant
Apple tooling and data.

## License

MIT. See [LICENSE](LICENSE).
