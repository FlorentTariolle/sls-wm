# DeepDash Version History

## V0 -- Ha & Schmidhuber Baseline (2026-03-03)

Starting point: the World Models architecture (Ha & Schmidhuber, 2018).

| Component | Design |
|-----------|--------|
| **V** | beta-VAE, continuous Gaussian latents, 64x64 RGB input |
| **M** | MDN-RNN (LSTM + Mixture Density Network), models stochastic dynamics |
| **C** | Linear controller optimized with CMA-ES |

### Limitations discovered

- beta-VAE reconstructions fundamentally blurry (posterior averaging). Spikes indistinguishable from blocks.
- MDN unnecessary: Geometry Dash physics are fully deterministic, visual stochasticity is noise.
- CMA-ES converged to local optimum (~12 steps survival).
- RGB input wastes capacity on particles, backgrounds, decorations.

---

## V1 -- Current (2026-03-03 to 2026-03-17)

Full architecture redesign, cherry-picking techniques from recent world model papers.

| Component | V0 | V1 | Why |
|-----------|----|----|-----|
| **V** | beta-VAE (continuous) | FSQ-VAE [8,5,5,5] (discrete) | Sharp reconstructions, no codebook collapse, 100% utilization |
| **M** | MDN-RNN | Transformer 256d/8H/8L | Discrete token classification, 51x compression, spatial attention |
| **C** | Linear + CMA-ES | CNNPolicy + BC + PPO | Spatial reasoning on 8x8 grid, BC pretraining for signal |
| **Input** | 64x64 RGB | 64x64 Sobel grayscale, square crop | Noise filtering, UI removal |
| **Decoding** | N/A (continuous) | Parallel single-step (all 65 tokens) | Real-time compatible |

### V1 sources

- FSQ-VAE tokenizer (from **FSQ**, Mentzer et al. 2023)
- Transformer world model on discrete tokens (from **IRIS**, Micheli et al. 2023)
- Parallel single-step decoding (from **IRIS**, simplified by dropping MaskGIT iterative decoding)
- AC-CPC contrastive loss (from **TWISTER**, Burchi & Timofte 2025)
- 3D-RoPE (from **V-JEPA 2**, Bardes et al. 2025)
- PPO controller training (from **PPO**, Schulman et al. 2017)

### V1 techniques

- FSQ-structured label smoothing (novel): Gaussian kernel over FSQ distance, sigma=0.9
- Dual token noise (novel): random 5% + FSQ neighbor 5%
- Vertical-only shift augmentation: 5x data multiplier
- Death token as 65th position: death prediction = token classification

### V1 results

- Transformer: 36.06% val acc, death F1 0.72
- Controller: 19.86 -> 23.04 eval survival (9K PPO iters)
- Deployment: 27ms avg, 30 FPS real-time
- Dodges simple obstacles, fails on complex sequences

### V1 known issues

- **Train/inference mismatch**: partial masking during training (cosine schedule), full masking at inference. Model can peek at unmasked true tokens during training but gets none at inference.
- **PPO signal is slow**: first 2K iterations are near-random noise even with BC.
- **PPO cosine LR schedule**: decayed too early, caused plateau before 9K iters.
- **Dream quality**: world model still predicts death too aggressively near obstacles.
- **Controller**: jumps too much/too early on complex obstacles.

See [experiments/v1/](experiments/v1/) for full logs and hyperparameters.

---

## V2 -- Complete (2026-03-18 to 2026-03-22)

### Changes applied

#### Vision (V) / FSQ-VAE

- Retrained on global episode-level split + expert episodes. Val recon 2.49 (vs V1 2.57).

#### Transformer (M)

- Removed masking entirely (all tokens predicted, no ground truth leakage)
- AC-CPC weight 0.1 -> 1.0 (swept 0.5, then 1.0. Metrics similar, but 1.0 produced visibly better dreams)
- torch.compile full model (static graph)
- Val acc 34.2%, death F1 0.73, train/val gap halved (1.6% vs V1 3.7%)

#### Controller (C) / PPO

- Jump penalty 0.2/jump, BC class weight 1.5x (was 2.6x)
- Percentile-based advantage normalization (EMA 5th-95th)
- EMA target critic (decay 0.98)
- Multi-token prediction: auxiliary loss predicting 8 future actions (coeff 0.1)
- 45-step dream rollouts (was 30)
- Constant LR, 5K iterations (plateaus after 3-5K)
- BC val acc 83.6% (vs V1 78%)

### V2 final results

- **Level 1 progress**: 11% (vs V1 10%), better generalization to other levels (1-2 extra obstacles cleared), slightly better use of jump orbs and pads
- **PPO eval survival**: 31.41 best (45-step rollouts, not comparable to V1's 30-step cap)
- Jump ratio 0.32 (vs V1 0.43)
- GPU Sobel in deployment: 7ms (was 10ms CPU)

### What worked

- Masking removal: halved train/val gap, more honest metrics, no quality loss
- Jump penalty (0.2) + lower BC weight (1.5x): jump ratio 0.43 -> 0.32, less over-jumping
- Percentile advantage normalization + EMA critic: stable training, no NaN
- Strict data split: proper train/val separation across all models
- AC-CPC 1.0: dream quality visibly better despite similar metrics to 0.5
- 45-step rollouts: agent learns longer-horizon consequences of actions
- MTP auxiliary loss: stable, adds timing awareness
- 5K iterations sufficient: eval plateaus after 3-5K, saves 2/3 compute

### What didn't work

- **Soft continuation gating**: agent exploited airtime (jump once, idle while airborne for full reward). Reverted to hard death cutoff.
- **PMPO**: sign-based advantages oscillated wildly with binary action space. Entropy collapsed then recovered repeatedly. Reverted to clipped PPO.
- **Symlog discrete value prediction**: skipped, +1/step reward doesn't need it
- **Resume state loss**: percentile normalizer + EMA critic reset on resume caused artificial eval jump. Fixed by saving/restoring in checkpoint.
- **Eval context RNG**: shared with training RNG, changed on resume. Fixed with dedicated eval RNG.

### V2 known issues

- **Jump timing**: main bottleneck. Off by 1-2 frames at 30 FPS.
- **Dream/reality gap**: policy learned in dream doesn't fully transfer.
- **Controller ignores game mechanics**: h_t encodes jump orbs/pads but PPO doesn't learn to use them.

---

## V3 -- Complete (2026-03-22 to 2026-03-23)

### Changes applied

#### Transformer (M)

- Scaled embedding dim 256 -> 384 (14.7M params, was 6.7M)
- Val acc 35.6% (vs V2 34.2%), death F1 0.78 (vs V2 0.73), val CPC 0.166 (vs V2 0.203)
- Train/val gap increased (3.6% vs V2 1.6%) due to extra capacity. May need dropout 0.15 later.
- Best epoch 146, inference 14.6ms (was 12.8ms)

#### Controller (C)

- BC val acc 87.1% (vs V2 83.6%)
- PPO best eval 33.48 (45-step rollouts), plateau at ~32.5 around 7K iters
- Jump ratio 0.30

### V3 deployment results (best model overall)

- **Level 1**: 20% (vs V2 11%, V1 10%)
- **Level 3**: 12%
- **Level 5**: 8%
- **Level 6**: 8%
- **Polargeist VE** (custom): 21%
- **Polargeist V2** (custom): 17%
- Best generalization across all tested levels

### What worked

- 384d embedding: significant quality jump across all metrics (acc, death F1, CPC)
- Richer h_t directly improved BC (78% -> 83.6% -> 87.1% across V1/V2/V3)
- PPO continued improving past 5K iters (unlike 256d which plateaued at 3-5K)

### V3 known issues

- **Overfitting**: train/val gap 3.6% (up from V2 1.6%). Dropout 0.15 may help.
- **Jump timing**: still the main bottleneck
- **Dream/reality gap**: persists

---

## V4 -- Complete (2026-03-28 to 2026-04-12)

### Changes applied

#### Vision (V) / FSQ-VAE

- Rebuilt FSQ without the 5x shift augmentation local optimum. Prior training accidentally fed the encoder 5 copies of every frame (one per vertical shift), oversmoothing the latent space and locking training into a higher-RMSE but artificially readable optimum.
- New FSQ has better RMSE but a "harder" latent space -- more faithful to the data, less convenient for downstream reads.
- This is the origin of V4's dream/reality gap widening (see known issues).

#### Transformer (M)

- Scaled embedding dim 384 -> 512 (~35M params, was 14.7M)
- **AdaLN-Zero action conditioning** (DiT/LeWorldModel-style): actions removed from the token sequence entirely, injected via adaptive layer norm in every block. Seq_len 325 -> 260.
- **QK-norm** (SD3/MMDiT-style): per-head RMSNorm on Q,K before attention. Prevents attention logit explosion that killed 5 earlier training runs.
- **AdaLN gate bias=1 init** (not 0). Zero-init killed all learning by collapsing residual flow. Bias=1 preserves it.
- Float32 AdaLN modulation (prevents FP16 overflow on scale/shift params)
- Dropout 0.15 (up from 0.10) to address V3's overfitting gap
- Val CPC **0.1328** (best ever, vs V3 0.166), val acc 27.7% (down from 35.6%, due to FSQ distribution shift not worse learning), death F1 ~0.79

#### Controller (C)

Three architectures tested in sequence, keeping the third:

1. **Small CNN (original, V3 shape carried over)**: ~50K params, linear heads on CNN+h_t features. Plateaued at eval survival ~29.9 around 10k PPO iters on the new V4 transformer.
2. **Big CNN scale-up**: 966K params, 3-layer MLP(512) trunk with LayerNorm+SiLU on CNN+h_t. Plateaued at 31.85 EMA survival around 20-25k iters. Nominal +1.95 dream survival gain over small CNN, but the gain came from memorizing dream-specific token grids rather than generalizing. Real-game deploy was sloppy despite better dream numbers -- died on trivial patterns it had seen at different positions.
3. **MLPPolicy (final, keeper)**: 265K params, 1-layer trunk (512 hidden, LayerNorm, SiLU) on h_t only. Zero-init actor/critic heads. Plateaus around 30.4 EMA survival at ~5k iters -- 2x faster than small CNN, 5x faster than big CNN. Beats both CNNs on real-game deployment across all levels.

Rationale for dropping the CNN branch entirely: no published world model uses CNN on latent tokens. IRIS/DIAMOND use CNN on raw pixels because they have no separate world model hidden state. TWISTER/DreamerV3 use MLP on latent state. Our h_t is already the output of 8 transformer layers with 3D-RoPE and AdaLN action conditioning -- a CNN on z_t would duplicate work the transformer already did. Empirically, the MLPPolicy has the smallest train/eval gap of any controller we trained, confirming it generalizes where the CNNs memorized.

**MTP auxiliary loss dropped entirely**: had an off-by-one bug in its targets for most of its life, and predicted from features that no longer exist once the CNN was removed. No evidence MTP helped at any point.

BC pretraining still used but gets overwritten within 20 PPO iterations. Candidate for removal in V5 (tracked as open question).

#### Training infrastructure

- `--config v4.yaml` flag for all training scripts (single source of truth)
- Bfloat16 for PPO rollouts (FP16 NaN'd, reverted)
- W&B run merging script (for splitting long runs across local and A100)
- `train/jump_ratio` logged every iteration (was only eval)
- Full pipeline now trainable in ~1.5 days wall-clock (FSQ ~1h, transformer ~12h, PPO ~12h, plus manual handoff)

### V4 deployment results (MLPPolicy, 5k PPO iters, on V4 transformer)

- **Level 1**: 14-16% (half the time), 16% when lucky
- **Level 2**: 12% (half the time)
- **Level 3**: 7% (consistent)
- **Level 4**: 4% (consistent)
- **Level 5**: 4-6%
- **Level 6**: 2%
- **Level 7**: 2%
- **Level 8**: 2-5%
- **Level 9**: 3%
- **Polargeist VE**: 2-10% (high variance, possibly custom mini-spikes)
- On the current V4 WM, MLPPolicy beats every CNN variant on every level.
- Still trails V3's 20% L1 on the old local-optimum FSQ. The V3 FSQ was easier to act on precisely because it was overfit to smoothed data.

### What worked

- **AdaLN-Zero + QK-norm**: best CPC score ever, stable 200-epoch training, cleaner sequence layout
- **Gate bias=1 init** (not 0): the single fix that unblocked AdaLN training after 5 failed runs. Zero-init killed residual flow.
- **MLPPolicy on h_t only**: 2-5x faster convergence, matches/beats CNN on real game despite half the params, smallest train/eval gap
- **Dropping MTP**: buggy from day one, tied to CNN features. Simpler controller, same or better results.

### What didn't work

- **Zero-init AdaLN gates** (DiT-style): killed all learning. Fixed with bias=1.
- **Tanh-bounding AdaLN modulation**: unnecessary once QK-norm was added. Was treating symptom not cause.
- **FP16 PPO rollouts**: NaN'd. Reverted to bfloat16.
- **Big CNN controller (966K params)**: nominal dream survival gain of +1.35 over small CNN was pure overfitting. Real-game deploy was sloppy, died on trivial patterns. Memorization, not generalization.
- **Small CNN carryover**: kept the V3 controller shape at the start of V4 assuming it was still appropriate. It wasn't -- h_t was already rich enough to not need a second visual encoder.

### V4 known issues

- **Dream/reality gap wider than V3**: new FSQ is more faithful to the data but the controller hasn't fully learned to act in the new latent space. V3's 20% L1 benchmark was on an easier latent.
- **FSQ trade-off**: honest FSQ (higher RMSE, cleaner latent) is theoretically correct but empirically harder to act on than the old oversmoothed one. Needs investigation.
- **BC pretraining may be unnecessary**: gets overwritten in 20 iterations by PPO. Ablation pending.
- **Controller is no longer the bottleneck**: MLPPolicy wins across all CNN variants, so future real-game gains must come from upstream (FSQ quality or richer h_t).

---

## Further ideas

### Vision (V) / FSQ-VAE

| Idea | Description |
|------|-------------|
| **Asymmetric encoder/decoder** | Lighter encoder (1 ResBlock/stage), heavier decoder (3 ResBlocks/stage). Faster deploy since only encoder runs at inference. |
| **FSQ levels sweep** | Try [8,8,5,5]=1600 or [10,5,5,4]=1000 to redistribute capacity across dimensions. |

### Transformer (M)

| Idea | Description |
|------|-------------|
| **Distill to smaller model** | Train large, distill to deployment-sized model (h_t matching). Novel for world models. |
| **Separate space/time attention** (Dreamer 4) | 3 space-only + 1 temporal layer, repeated. Space layers skip KV cache from prior frames. |
| **Multi-level feature extraction** (V-JEPA 2.1) | Concatenate h_t from multiple intermediate transformer layers. Gives controller access to different abstraction levels. |
| **More training data** | Record on more diverse levels and custom levels. |
| **Context frames 4 -> 6** | More temporal context for obstacle distance estimation. |
| **Increase dropout** (0.1 -> 0.15) | Address V3 overfitting from larger model capacity. |

### Controller (C) / PPO

| Idea | Description |
|------|-------------|
| **MTP ablation** | Already implemented (8-step). Evaluate impact by comparing with/without. |

---

## V5 -- Ideas

### Vision (V) / FSQ-VAE

| Idea | Description |
|------|-------------|
| **Investigate FSQ trade-off** | Understand why V3's oversmoothed FSQ was easier to act on than V4's cleaner one. Retrain with controlled augmentation to isolate the effect. |

### Transformer (M)

| Idea | Description |
|------|-------------|
| **Motion-compensated token prediction** | Exploit side-scroller structure: shift token grid left by learned scroll offset, only predict new right column (~8 tokens instead of 64). Speed is constant within a level, learnable from context frames. Massive reduction in prediction complexity. |
| **Multi-level h_t readout** (V-JEPA 2.1) | Concatenate h_t from multiple intermediate transformer layers. Gives the MLP controller access to different abstraction levels without adding a second encoder. |
| **Separate space/time attention** (Dreamer 4) | 3 space-only + 1 temporal layer, repeated. Space layers skip KV cache from prior frames, cheaper inference. |
| **Per-Layer Embeddings** (Gemma 3n) | Each transformer layer gets its own lightweight learnable embedding that modulates activations, adding per-layer capacity without a full param scale-up. Originally designed for edge inference (PLEs can live in CPU RAM), but the per-layer expressiveness could enrich h_t and tighten the dream/reality gap. Cheap to add to an existing 512d transformer. |

### Controller (C) / PPO

| Idea | Description |
|------|-------------|
| **BC ablation** | MLPPolicy recovers from BC overwrite in ~20 iters. Test PPO without `--pretrained` to see if BC is still net-positive or pure overhead. |
| **MLPPolicy depth sweep** | We picked 1 hidden layer because h_t is pre-processed. Validate by training 2-3 layer variants and comparing real-game. |

---

### Discarded ideas

- ~~16x16 grid~~: 8x8 dream quality is sufficient, timing issue is dream/reality gap not spatial resolution (triple spike consistently cleared)
- ~~PMPO~~: oscillated with binary action space
- ~~Soft continuation gating~~: agent exploited airtime
- ~~Symlog value prediction~~: unnecessary for +1/step reward
- ~~KL penalty to BC prior~~: caps improvement ceiling
