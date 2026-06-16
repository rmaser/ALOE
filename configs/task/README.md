# `configs/task/` (run / schedule bundles)

This directory holds **trainer, optimizer LR, data batching, `resume_slurm`**, and similar
**run-length** chunks. Compose from experiments via `defaults: - /task/<name>`.

## Three-layer layout (native Hub distillation)

| Layer | Where | Responsibility |
|--------|--------|----------------|
| **Run** | `task/` (this folder) | Steps, epochs, batch size, run-specific optimizer LR, Slurm resume |
| **Objective** | [`objective/`](../objective/README.md) | Default distillation loss for native Hub runs (composed via [`experiment/native/_distill_hf_native.yaml`](../experiment/native/_distill_hf_native.yaml) from root [`config.yaml`](../config.yaml)) |
| **Module** | Root `config.yaml` + [`model/`](../model/) | Lightning `model` + `student_factory` / `teacher_factory` live in `/experiment/native/_distill_hf_native`; backbones and factories under `model/`. Run-level optimizer values belong in `task/` or the experiment. |

**Merge caveat:** Keys under `model.cfg` that you set only inside a **task** can be dropped unless you repeat them in the **experiment** body (see `experiment=native/distill_aloe_170k` and `model.cfg.distillation_layers`).

**Override caveat:** `override /loss: …` only works if `loss` appears in **that YAML file’s** `defaults:` list. Including `/task/native_hub_distill_train` does **not** register `loss` on the experiment node by itself. Prefer **`model.cfg.loss_fn`** in the experiment body (see [`experiment/native/align_last_mse.yaml`](../experiment/native/align_last_mse.yaml)), or add explicit anchors **without** duplicating the `loss` group (anchoring before a task that also composes `loss` causes “loss appears more than once”).

## Bundles in this repo

| File | Role |
|------|------|
| `native_hub_distill_short_run.yaml` | Trainer / optimizer / data / `resume_slurm` for 30k native Hub runs (composed by `native_hub_distill_train`). |
| `native_hub_distill_train.yaml` | Chains short run + `data` + `one_cycle`; distillation wiring comes from root `config.yaml` (`/experiment/native/_distill_hf_native`). |
| `native_hub_align_last.yaml` | `model.cfg.distillation_layers: [-1]` for align-last experiments. |
| `native_hub_distill_hidden_layers.yaml` | `select_distillation_layers` for multi-block + pooler native runs. |
| `native_hub_drop_register_tokens.yaml` | Sets `model.cfg.drop_register_tokens: true` so distillation ignores register tokens while leaving the rest of the run stack unchanged. |
| `distill_resume_student_ckpt_30k.yaml` | 30k finetune + resume student from `checkpoints[backbone]` (after other defaults so it overrides e.g. `max_steps: -1`). |
| `distill_converged_170k.yaml` | Full converged distill run: 170k steps, 3e-4 AdamW, batch 1024, `select_distillation_layers`, `resume_slurm`. Pair with `override /scheduler: aloe_scheduler` (and optionally MSE via `model.cfg.loss_fn`). |
| `native_hub_distill_converged_170k.yaml` | Same 170k trainer/opt/data/resume as above, **without** `model:` — pair with root `config.yaml` (includes `_distill_hf_native`) and set `model.cfg` in the experiment body (see `experiment=native/distill_aloe_170k`). |
| `native_hub_distill_converged_300k.yaml` | Extends the native-Hub 170k long-run bundle to 300k steps. Pair with `override /scheduler: aloe_scheduler_300k` and set `model.cfg` in the experiment body (see `experiment=native/distill_aloe_300k`). |
| `native_hub_distill_432_safe.yaml` | 432px-only launcher fragment that lowers host-RAM pressure by reducing workers and disabling pin/persistent worker amplification. Compose it after `experiment=native/distill_aloe_170k_432_safe`. |

## Audit: native distillation wiring

Shared **DistillHFModel** + factories + objective defaults: [`experiment/native/_distill_hf_native.yaml`](../experiment/native/_distill_hf_native.yaml), included from root [`config.yaml`](../config.yaml).

- **Experiments:** `native/distill_aloe_170k`, `native/distill_aloe_300k`, `native/align_last_170k`, and variants (e.g. `distill_aloe_170k_432_safe`, `distill_aloe_170k_drop_register_tokens`).

Add a new task file when several experiments repeat the same global keys and the chunk is not a factory shard.

**Hydra rule:** in any `defaults:` list, every `override …` entry must come **after** all non-override entries (including `- /task/...`). Put the checkpoint-resume task **before** trailing `override /scheduler: …` lines when both appear in the same experiment.

Document each file at the top with example `defaults` usage.
