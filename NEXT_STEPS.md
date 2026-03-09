# Next Steps

## Tier 1 — High impact, feasible

### 1. Death token (replaces broken death head)
- Remove the separate binary death head
- Add a `DEATH` token (index 1024) appended after each frame's 36 tokens
- Death prediction folds into the same cross-entropy objective
- Solves the 98.6% = always-alive baseline problem

### 2. More data (especially deaths)
- 318 death frames in 22K total (1.4%) is insufficient
- Record 200+ more episodes with intentional deaths (suicide runs)
- Target: 40K+ windows, diverse death sequences across all levels
- Directly addresses the 6% train/val gap

### 3. Scheduled sampling 10-15%
- Current 5% token noise is too conservative
- Autoregressive drift at step 3 suggests the model rarely sees its own errors during training
- Quick experiment: bump to 10%, measure multi-step rollout quality

### 4. Delta-state prediction
- GD is constant-velocity side-scrolling: ~80% of tokens between frames are spatially shifted
- Predict token changes instead of full token sequences
- Options: `(target - context) mod 1024`, or binary changed/unchanged mask + new values
- Reduces prediction difficulty, slows autoregressive drift compounding

## Tier 2 — Moderate impact, moderate effort

### 5. FSQ replacing VQ-VAE
- Eliminates codebook collapse risk, removes commitment loss
- Provides topological ordering: nearby pixel changes map to nearby latent changes
- Config: L = [7, 5, 5, 5, 5] gives 4375 implicit codes
- Requires full pipeline retrain (tokenizer + re-tokenize episodes + retrain Transformer)

### 6. 8x8 token grid (up from 6x6)
- Small game elements (jump pads, orbs) are invisible at current resolution
- Sequence length 184 -> 328 (~2x attention cost)
- Bundle with FSQ retrain if doing both

### 7. GRWM geometric regularization
- Add temporal slowness loss: penalizes large distances between consecutive frame latents
- Add uniformity loss: prevents collapse to single point
- Lightweight auxiliary losses on encoder, plug-and-play
- Worth adding if retraining tokenizer anyway (FSQ or 8x8)
