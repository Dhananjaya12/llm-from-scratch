# from __future__ import annotations

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from edge_cloud_llm.utils import top_k_top_p_filtering

# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.scale = nn.Parameter(torch.ones(dim))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape: (..., dim)
#         rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
#         x_norm = x / rms
#         return self.scale * x_norm
    
# class CausalSelfAttention(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
#         super().__init__()
#         assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

#         self.n_head = n_head
#         self.d_head = n_embd // n_head

#         self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
#         self.proj = nn.Linear(n_embd, n_embd, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x shape: (B, T, C)
#         B = batch size
#         T = sequence length
#         C = embedding dimension
#         """
#         B, T, C = x.shape

#         qkv = self.qkv(x)
#         qkv = qkv.view(B, T, 3, self.n_head, self.d_head)

#         q, k, v = qkv.unbind(dim=2)

#         q = q.transpose(1, 2)  # (B, n_head, T, d_head)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         y = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=None,
#             dropout_p=self.dropout.p if self.training else 0.0,
#             is_causal=True,
#         )

#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.proj(y)
#         return y


# class FeedForward(nn.Module):
#     def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd, mult * n_embd),
#             nn.GELU(),
#             nn.Linear(mult * n_embd, n_embd),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# class Block(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, dropout: float):
#         super().__init__()
#         # self.ln1 = nn.LayerNorm(n_embd)
#         self.ln1 = RMSNorm(n_embd)
#         self.attn = CausalSelfAttention(n_embd, n_head, dropout)
#         # self.ln2 = nn.LayerNorm(n_embd)
#         self.ln2 = RMSNorm(n_embd)
#         self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.attn(self.ln1(x))
#         x = x + self.ffn(self.ln2(x))
#         return x


# class GPT(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         block_size: int,
#         n_layer: int = 4,
#         n_head: int = 4,
#         n_embd: int = 256,
#         dropout: float = 0.0,
#     ):
#         super().__init__()

#         self.block_size = block_size

#         self.tok_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.drop = nn.Dropout(dropout)

#         self.blocks = nn.ModuleList(
#             [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
#         )

#         # self.ln_f = nn.LayerNorm(n_embd)
#         self.ln_f = RMSNorm(n_embd)
#         self.head = nn.Linear(n_embd, vocab_size, bias=False)

#         self.apply(self._init_weights)

#     def _init_weights(self, module: nn.Module) -> None:
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(
#         self,
#         idx: torch.Tensor,
#         targets: torch.Tensor | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor | None]:
#         """
#         idx shape: (B, T)
#         targets shape: (B, T), optional
#         """
#         B, T = idx.shape
#         assert T <= self.block_size, "Sequence length exceeds block_size"

#         pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

#         x = self.tok_emb(idx) + self.pos_emb(pos)
#         x = self.drop(x)

#         for block in self.blocks:
#             x = block(x)

#         x = self.ln_f(x)
#         logits = self.head(x)

#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)),
#                 targets.view(-1),
#             )

#         return logits, loss

#     @torch.no_grad()
#     def generate(
#         self,
#         idx: torch.Tensor,
#         max_new_tokens: int = 100,
#         temperature: float = 1.0,
#         top_k: int | None = 50,
#         top_p: float | None = None,
#     ) -> torch.Tensor:
#         self.eval()

#         for _ in range(max_new_tokens):
#             idx_cond = idx[:, -self.block_size:]
#             logits, _ = self(idx_cond)

#             logits = logits[:, -1, :] / max(temperature, 1e-6)
#             logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

#             probs = torch.softmax(logits, dim=-1)
#             next_id = torch.multinomial(probs, num_samples=1)

#             idx = torch.cat([idx, next_id], dim=1)

#         return idx



from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_cloud_llm.utils import top_k_top_p_filtering


# ---------------------------------------------------------
# 1. RMSNorm
# ---------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ---------------------------------------------------------
# 2. RoPE helpers
# ---------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension in half and rotates:
    [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, positions: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.get_cos_sin(positions, q.device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


# ---------------------------------------------------------
# 3. Causal Self-Attention with RoPE + KV cache
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        window_size: int | None = None,
        attention_sink_tokens: int = 0,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.d_head = n_embd // n_head
        if self.d_head % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE.")

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.d_head, base=rope_base)

        # Optional inference-time cache trimming
        self.window_size = window_size
        self.attention_sink_tokens = attention_sink_tokens

    def _apply_cache_policy(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optional rolling cache policy for inference.

        Keeps:
        - first `attention_sink_tokens`
        - last `window_size` tokens

        Only trims when window_size is set and total cache is longer than allowed.
        """
        if self.window_size is None:
            return k, v

        total_len = k.size(-2)
        sink = max(0, self.attention_sink_tokens)

        if total_len <= sink + self.window_size:
            return k, v

        if sink == 0:
            return k[..., -self.window_size :, :], v[..., -self.window_size :, :]

        sink_k = k[..., :sink, :]
        sink_v = v[..., :sink, :]

        recent_k = k[..., -self.window_size :, :]
        recent_v = v[..., -self.window_size :, :]

        k = torch.cat([sink_k, recent_k], dim=-2)
        v = torch.cat([sink_v, recent_v], dim=-2)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        x: (B, T, C)
        positions: (T,) absolute positions for the current chunk
        past_kv:
            k_cache, v_cache shaped (B, n_head, T_past, d_head)

        Returns:
            y: (B, T, C)
            new_kv if use_cache else None
        """
        B, T, C = x.shape

        qkv = self.qkv(x)                                  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head) # (B, T, 3, H, Dh)
        q, k, v = qkv.unbind(dim=2)                       # each (B, T, H, Dh)

        q = q.transpose(1, 2)  # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to current q and k
        q, k = self.rope.apply_rotary(q, k, positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            k_all = torch.cat([past_k, k], dim=-2)
            v_all = torch.cat([past_v, v], dim=-2)
        else:
            k_all = k
            v_all = v

        # Optional inference-time rolling cache / attention sink behavior
        if use_cache:
            k_all, v_all = self._apply_cache_policy(k_all, v_all)

        # Training path: full causal attention over the whole chunk
        if past_kv is None:
            y = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # In cached generation, q is only the new chunk.
            # It is allowed to attend to all cached keys and current keys.
            y = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        new_kv = (k_all, v_all) if use_cache else None
        return y, new_kv


# ---------------------------------------------------------
# 4. SwiGLU FeedForward
# ---------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = mult * n_embd

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2( silu(w1(x)) * w3(x) )
        gated = F.silu(self.w1(x)) * self.w3(x)
        out = self.w2(gated)
        return self.dropout(out)


# ---------------------------------------------------------
# 5. Transformer Block
# ---------------------------------------------------------
class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        rope_base: float = 10000.0,
        window_size: int | None = None,
        attention_sink_tokens: int = 0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            rope_base=rope_base,
            window_size=window_size,
            attention_sink_tokens=attention_sink_tokens,
        )
        self.ln2 = RMSNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_kv = self.attn(
            self.ln1(x),
            positions=positions,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


# ---------------------------------------------------------
# 6. GPT Model (RoPE means no learned position embedding now)
# ---------------------------------------------------------
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        window_size: int | None = None,
        attention_sink_tokens: int = 0,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_layer = n_layer

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    rope_base=rope_base,
                    window_size=window_size,
                    attention_sink_tokens=attention_sink_tokens,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        """
        idx: (B, T)
        targets: optional (B, T)
        kv_caches: list of length n_layer, each item is None or (k, v)
        use_cache: if True, return updated kv caches

        Returns:
            logits, loss, new_kv_caches
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        if kv_caches is None:
            kv_caches = [None] * self.n_layer

        # Determine absolute positions for current chunk
        if kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(-2)
        else:
            past_len = 0

        positions = torch.arange(past_len, past_len + T, device=idx.device)

        x = self.tok_emb(idx)
        x = self.drop(x)

        new_kv_caches = [] if use_cache else None

        for block, past_kv in zip(self.blocks, kv_caches):
            x, new_kv = block(
                x,
                positions=positions,
                past_kv=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()

        if idx.size(1) == 0:
            raise ValueError("Prompt cannot be empty.")

        # Keep prompt within model context window
        idx = idx[:, -self.block_size :]

        if use_cache:
            # First pass: process the prompt and build caches
            logits, _, kv_caches = self(idx, kv_caches=None, use_cache=True)
            next_logits = logits[:, -1, :]
        else:
            kv_caches = None
            next_logits = None

        for _ in range(max_new_tokens):
            if use_cache:
                logits = next_logits / max(temperature, 1e-6)
            else:
                idx_cond = idx[:, -self.block_size :]
                logits, _, _ = self(idx_cond, kv_caches=None, use_cache=False)
                logits = logits[:, -1, :] / max(temperature, 1e-6)

            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if use_cache:
                # Only feed the new token when using cache
                next_logits, _, kv_caches = self(
                    next_id,
                    kv_caches=kv_caches,
                    use_cache=True,
                )
                next_logits = next_logits[:, -1, :]

        return idx