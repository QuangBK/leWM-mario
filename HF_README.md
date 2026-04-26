---
license: mit
tags:
- world-model
- jepa
- mario
- super-mario-bros
- reinforcement-learning
- world-models
library_name: pytorch
pipeline_tag: other
---

# LeWM Mario — Phase 1 checkpoint

Adaptation of [LeWorldModel](https://arxiv.org/abs/2603.19312) (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026) — a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) world model — to **Super Mario Bros** raw pixels.

- **Code:** https://github.com/QuangBK/leWM-mario
- **Paper:** [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **Trained on:** 120 TAS speedruns replayed through `nes_py` against the SMB ROM bundled with `gym-super-mario-bros`. 156k action frames → 31k blocked transitions (`frame_skip=5`).
- **Hardware:** single Targon H100 (80 GB), batch 128, bf16, `--compile`. ~25 min wall time for 50 epochs.

## Files

- `best.pt` — full checkpoint (model state, optimizer, scheduler, action library, config). 208 MB. Saved at epoch 45 (lowest validation loss).
- `eval_results.json` — open-loop latent MSE by horizon, x_pos linear probe results.
- `metadata.json` — training metadata (config, episode list, action library, button order).

## Phase-1 results

| Metric | Value |
|---|---|
| val `pred_loss` (epoch 0 / 45) | 0.36 → **0.127** |
| val `sigreg_loss` final | 4.0 |
| open-loop latent MSE @ horizon 1 | 0.12 |
| open-loop latent MSE @ horizon 16 | 1.83 |
| **`x_pos` linear probe R²** | **0.83** |
| `x_pos` MAE | 65 px (range 0–1571) |

The probe demonstrates that the unsupervised latent encodes Mario's spatial position without ever being trained against `x_pos` labels — the LeWM paper's "physics emerges unsupervised" property reproduced on Mario.

## Usage

```python
import torch, sys
sys.path.insert(0, "path/to/lewm_mario")  # https://github.com/0xShug0/lewm_mario
from mario_lewm.model import LeWorldModel, LeWorldModelConfig

ck = torch.load("best.pt", map_location="cpu", weights_only=False)
cfg = LeWorldModelConfig(**ck["config"])
model = LeWorldModel(cfg)
model.load_state_dict(ck["model_state"])
model.eval()
```

Reproduction scripts (data generation, training, evaluation): https://github.com/QuangBK/leWM-mario

## Limitations

- TAS desync: TAS files were recorded against FCEUX with the canonical SMB1 (JU) PRG0 iNES dump (sha256 `F61548FD…`). `gym-super-mario-bros` ships a normalized ROM (sha256 `ec299b9…`); replay through `nes_py` desyncs after ~200–3000 frames per TAS. Each TAS still contributes a clean prefix.
- Goal-conditioned only: the included planner targets a goal *image*. Autonomous play needs a reward head (Phase 2).

## License

MIT (matching upstream `lewm_mario`).
