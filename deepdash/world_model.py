"""Transformer world model V4 — block-causal + RoPE + death + AC-CPC.

Predicts next-frame tokens + death probability given past frames + actions.
Uses Rotary Position Embeddings (RoPE) instead of absolute positional encoding
for better relative position awareness and length generalization.
AC-CPC (Action-Conditioned Contrastive Predictive Coding) adds a contrastive
loss that predicts future hidden states conditioned on actions (TWISTER).

Sequence format (K context frames):
    [f0 (36 tokens)] [a0 (1 token)] ... [fK-1 (36)] [aK-1 (1)] [target (36)]

Block-causal attention: bidirectional within each frame/action block, causal across.
Target frame uses causal masking within. GPT-shift: position t-1 predicts token t.

References:
    - IRIS (Micheli et al., ICLR 2023): block-causal attention on VQ tokens
    - TWISTER (Burchert et al., ICLR 2025): AC-CPC contrastive loss
    - Su et al., 2021: RoFormer / Rotary Position Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tokens per frame (6x6 VQ-VAE grid)
TOKENS_PER_FRAME = 36


def apply_rope(x, cos, sin):
    """Apply rotary position embedding.

    Args:
        x: (B, n_heads, T, head_dim)
        cos: (T, head_dim // 2)
        sin: (T, head_dim // 2)
    Returns:
        Rotated tensor, same shape as x.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class WorldModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        n_actions: int = 2,
        n_levels: int = 8,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        context_frames: int = 4,
        dropout: float = 0.1,
        cpc_dim: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.context_frames = context_frames
        self.tokens_per_frame = TOKENS_PER_FRAME

        # Sequence length: K * (36 + 1) + 36
        self.seq_len = context_frames * (TOKENS_PER_FRAME + 1) + TOKENS_PER_FRAME

        # Embeddings (no absolute positional — using RoPE instead)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.action_embed = nn.Embedding(n_actions, embed_dim)
        self.level_embed = nn.Embedding(n_levels, embed_dim)

        # RoPE precomputed frequencies
        head_dim = embed_dim // n_heads
        rope_cos, rope_sin = self._precompute_rope(head_dim, self.seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.death_head = nn.Linear(embed_dim, 1)

        # AC-CPC: contrastive prediction of future hidden states
        self.cpc_dim = cpc_dim
        self.cpc_target_proj = nn.Linear(embed_dim, cpc_dim)
        # One predictor per horizon step (1..K)
        self.cpc_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim + embed_dim, cpc_dim),
                nn.GELU(),
                nn.Linear(cpc_dim, cpc_dim),
            )
            for _ in range(context_frames)
        ])

        # Block-causal attention mask (bidirectional within blocks, causal across)
        self.register_buffer("attn_mask", self._build_mask())

        self._init_weights()

    @staticmethod
    def _precompute_rope(dim, max_len, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)  # (max_len, dim//2)
        return torch.cos(angles), torch.sin(angles)

    def _build_mask(self):
        """Build hybrid attention mask.

        Context frames: block-causal (bidirectional within frame, causal across).
        Target frame: causal within the block (position t sees target tokens 0..t-1
        only, plus all context). This prevents cheating via bidirectional attention.

        Returns:
            mask: (seq_len, seq_len) bool — True = blocked.
        """
        S = self.seq_len
        K = self.context_frames
        TPF = TOKENS_PER_FRAME
        ctx_end = K * (TPF + 1)  # start of target block

        # Assign block index to each position
        block_idx = torch.zeros(S, dtype=torch.long)
        pos = 0
        for i in range(K):
            block_idx[pos:pos + TPF] = 2 * i        # frame i
            pos += TPF
            block_idx[pos] = 2 * i + 1               # action i
            pos += 1
        block_idx[pos:] = 2 * K                      # target frame

        # Block-causal: query can attend to same or earlier blocks
        mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)

        # Override target block: causal within (not bidirectional)
        target_size = TPF
        for i in range(target_size):
            for j in range(i + 1, target_size):
                mask[ctx_end + i, ctx_end + j] = True

        return mask

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _compute_cpc_loss(self, x, actions, temperature=0.1):
        """Compute AC-CPC contrastive loss over context frame hidden states.

        For each pair (source_step t, target_step t+k), predicts the hidden
        representation at t+k conditioned on actions t..t+k-1.

        Args:
            x: (B, seq_len, D) — transformer output after ln_f.
            actions: (B, K) long — action indices for context frames.
            temperature: InfoNCE temperature.

        Returns:
            cpc_loss: scalar tensor.
        """
        K = self.context_frames
        TPF = TOKENS_PER_FRAME
        B = x.size(0)

        if B < 2:
            return torch.tensor(0.0, device=x.device)

        # Action positions in sequence: after each frame's 36 tokens
        action_positions = [i * (TPF + 1) + TPF for i in range(K)]

        # Extract hidden states at action positions + target summary (last pos)
        h_steps = [x[:, pos] for pos in action_positions]
        h_steps.append(x[:, -1])

        # Project targets to CPC space (stop gradient on targets)
        z_targets = [F.normalize(self.cpc_target_proj(h.detach()), dim=-1)
                     for h in h_steps]

        # Action embeddings for conditioning
        act_embeds = self.action_embed(actions)  # (B, K, D)

        total_loss = 0.0
        n_pairs = 0

        for step_idx, k in enumerate(range(1, K + 1)):
            predictor = self.cpc_predictors[step_idx]

            for t in range(K + 1 - k):
                h_src = h_steps[t]
                end = min(t + k, K)
                if end <= t:
                    continue
                act_ctx = act_embeds[:, t:end].mean(dim=1)

                z_pred = predictor(torch.cat([h_src, act_ctx], dim=-1))
                z_pred = F.normalize(z_pred, dim=-1)
                z_pos = z_targets[t + k]

                # InfoNCE: (B, B) similarity, diagonal = positives
                sim = torch.mm(z_pred, z_pos.t()) / temperature
                labels = torch.arange(B, device=sim.device)
                total_loss += F.cross_entropy(sim, labels)
                n_pairs += 1

        return total_loss / max(n_pairs, 1)

    def forward(self, frame_tokens, actions, level_ids=None):
        """Forward pass.

        Args:
            frame_tokens: (B, K+1, 36) long — K context frames + 1 target frame.
            actions: (B, K) long — action for each context frame.
            level_ids: (B,) long — level index (0-based). None = no conditioning.

        Returns:
            logits: (B, 36, vocab_size) — predictions for the target frame tokens.
            death_logit: (B, 1) — death prediction logit (raw, pre-sigmoid).
            cpc_loss: scalar — AC-CPC contrastive loss (0 during eval).
        """
        B = frame_tokens.size(0)
        K = self.context_frames

        # Build interleaved sequence: [f0 a0 f1 a1 ... fK-1 aK-1 fK]
        parts = []
        for i in range(K):
            parts.append(self.token_embed(frame_tokens[:, i]))  # (B, 36, D)
            act = self.action_embed(actions[:, i])               # (B, D)
            parts.append(act.unsqueeze(1))                       # (B, 1, D)
        parts.append(self.token_embed(frame_tokens[:, K]))       # (B, 36, D) target

        x = torch.cat(parts, dim=1)  # (B, seq_len, D)

        # Level conditioning (no absolute pos embed — RoPE handles position)
        if level_ids is not None:
            x = x + self.level_embed(level_ids).unsqueeze(1)
        x = self.embed_drop(x)

        # Transformer blocks with RoPE
        for block in self.blocks:
            x = block(x, self.attn_mask, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)

        # GPT-style shift: predict target token t from position t-1.
        predict_positions = x[:, -(TOKENS_PER_FRAME + 1):-1]  # (B, 36, D)
        logits = self.head(predict_positions)  # (B, 36, vocab_size)

        # Death prediction from last position
        death_logit = self.death_head(x[:, -1])  # (B, 1)

        # AC-CPC loss (training only)
        if self.training:
            cpc_loss = self._compute_cpc_loss(x, actions)
        else:
            cpc_loss = torch.tensor(0.0, device=x.device)

        return logits, death_logit, cpc_loss

    @torch.no_grad()
    def predict_next_frame(self, frame_tokens, actions, level_ids=None):
        """Predict next frame tokens autoregressively.

        Args:
            frame_tokens: (B, K, 36) long — K context frames.
            actions: (B, K) long — actions for context frames.
            level_ids: (B,) long — level index (0-based). None = no conditioning.

        Returns:
            predicted: (B, 36) long — predicted next frame tokens.
            death_prob: (B,) float — probability of death at predicted frame.
        """
        B = frame_tokens.size(0)
        predicted = torch.zeros(B, TOKENS_PER_FRAME, dtype=torch.long,
                                device=frame_tokens.device)

        for t in range(TOKENS_PER_FRAME):
            full_frames = torch.cat([
                frame_tokens,
                predicted.unsqueeze(1),
            ], dim=1)

            logits, death_logit, _ = self.forward(full_frames, actions, level_ids)
            predicted[:, t] = logits[:, t].argmax(dim=-1)

        death_prob = torch.sigmoid(death_logit).squeeze(-1)
        return predicted, death_prob


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.ln1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, rope_cos, rope_sin):
        B, T, D = x.shape
        h = self.ln1(x)

        # QKV projection → (B, T, 3, n_heads, head_dim)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, n_heads, T, head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        h = (attn @ v).transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(h)

        x = x + self.resid_drop(h)
        x = x + self.mlp(self.ln2(x))
        return x
