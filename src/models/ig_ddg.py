"""IG-DDG model implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    w = mask.unsqueeze(-1)
    num = (x * w).sum(dim=dim)
    den = w.sum(dim=dim).clamp_min(eps)
    return num / den


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * mult),
            nn.GELU(),
            nn.Linear(d * mult, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InterfaceMessageBlock(nn.Module):
    def __init__(self, d: int, edge_dim: int, use_inter_chain_attention: bool = True, use_film: bool = True) -> None:
        super().__init__()
        self.d = d
        self.use_inter_chain_attention = use_inter_chain_attention
        self.use_film = use_film

        self.msg_mlp = nn.Sequential(
            nn.Linear(d * 2 + edge_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.inter_proj = nn.Linear(d * 2, d)
        self.film = nn.Linear(d, d * 2)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)

    def _aggregate_edges(
        self,
        h_b: torch.Tensor,
        edge_index_b: torch.Tensor,
        edge_feat_b: torch.Tensor,
        edge_mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = edge_mask_b > 0.5
        if valid.sum() == 0:
            z = torch.zeros_like(h_b)
            deg = torch.zeros((h_b.shape[0],), device=h_b.device, dtype=h_b.dtype)
            return z, deg

        ei = edge_index_b[:, valid].long()
        ef = edge_feat_b[valid]
        src = ei[0]
        dst = ei[1]
        ok = (src >= 0) & (dst >= 0) & (src < h_b.shape[0]) & (dst < h_b.shape[0])
        src = src[ok]
        dst = dst[ok]
        ef = ef[ok]
        if src.numel() == 0:
            z = torch.zeros_like(h_b)
            deg = torch.zeros((h_b.shape[0],), device=h_b.device, dtype=h_b.dtype)
            return z, deg

        msg_in = torch.cat([h_b[src], h_b[dst], ef], dim=-1)
        msg = self.msg_mlp(msg_in)

        agg = torch.zeros_like(h_b)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((h_b.shape[0],), device=h_b.device, dtype=h_b.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h_b.dtype))
        return agg, deg

    def forward(
        self,
        h: torch.Tensor,
        chain_ids: torch.Tensor,
        mutation_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        # h: [B, N, d]
        B, N, _ = h.shape
        out = []
        for b in range(B):
            h_b = h[b]
            nmask = node_mask[b] > 0.5
            agg, deg = self._aggregate_edges(h_b, edge_index[b], edge_feat[b], edge_mask[b])
            h_new = h_b + agg / torch.sqrt(deg.unsqueeze(-1) + 1.0)

            if self.use_inter_chain_attention:
                chain = chain_ids[b]
                for c in torch.unique(chain[nmask]):
                    idx = (chain == c) & nmask
                    opp = (chain != c) & nmask
                    if idx.any() and opp.any():
                        opp_mean = h_new[opp].mean(dim=0, keepdim=True)
                        h_new[idx] = h_new[idx] + self.inter_proj(
                            torch.cat([h_new[idx], opp_mean.expand(idx.sum(), -1)], dim=-1)
                        )

            if self.use_film:
                m = mutation_mask[b] * node_mask[b]
                if m.sum() > 0:
                    cond = masked_mean(h_new.unsqueeze(0), m.unsqueeze(0), dim=1).squeeze(0)
                else:
                    cond = masked_mean(h_new.unsqueeze(0), node_mask[b].unsqueeze(0), dim=1).squeeze(0)
                gamma, beta = self.film(cond).chunk(2, dim=-1)
                h_new = h_new * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)

            h_new = self.norm1(h_new)
            h_new = h_new + self.ffn(self.norm2(h_new))
            h_new = h_new * node_mask[b].unsqueeze(-1)
            out.append(h_new)

        return torch.stack(out, dim=0)


@dataclass
class IGDDGConfig:
    aa_vocab: int = 21
    esm_dim: int = 1280
    scalar_dim: int = 6
    atom_dim: int = 8
    edge_dim: int = 25
    hidden_dim: int = 256
    num_layers: int = 8
    cath_classes: int = 32
    foldx_dim: int = 32
    use_foldx_branch: bool = True
    use_inter_chain_attention: bool = True
    use_film: bool = True


class PatchEncoder(nn.Module):
    def __init__(self, cfg: IGDDGConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.aa_emb = nn.Embedding(cfg.aa_vocab + 1, 64)
        self.esm_proj = nn.Linear(cfg.esm_dim, 128)
        self.scalar_proj = nn.Linear(cfg.scalar_dim, 32)
        self.atom_proj = nn.Linear(cfg.atom_dim, 32)
        self.mut_proj = nn.Linear(1, 16)
        self.input_proj = nn.Linear(64 + 128 + 32 + 32 + 16, d)

        self.blocks = nn.ModuleList(
            [
                InterfaceMessageBlock(
                    d=d,
                    edge_dim=cfg.edge_dim,
                    use_inter_chain_attention=cfg.use_inter_chain_attention,
                    use_film=cfg.use_film,
                )
                for _ in range(cfg.num_layers)
            ]
        )

    def forward(self, patch: dict[str, torch.Tensor]) -> torch.Tensor:
        aa = patch["aa_ids"].long()
        esm = patch["esm"].float()
        scalars = patch["scalars"].float()
        atom = patch["atom_summary"].float()
        m = patch["mutation_mask"].float().unsqueeze(-1)

        h = torch.cat(
            [
                self.aa_emb(aa),
                self.esm_proj(esm),
                self.scalar_proj(scalars),
                self.atom_proj(atom),
                self.mut_proj(m),
            ],
            dim=-1,
        )
        h = self.input_proj(h)

        for blk in self.blocks:
            h = blk(
                h=h,
                chain_ids=patch["chain_ids"],
                mutation_mask=patch["mutation_mask"],
                edge_index=patch["edge_index"],
                edge_feat=patch["edge_feat"],
                edge_mask=patch["edge_mask"],
                node_mask=patch["node_mask"],
            )
        return h


class IGDDG(nn.Module):
    def __init__(self, cfg: IGDDGConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or IGDDGConfig()
        d = self.cfg.hidden_dim

        self.encoder = PatchEncoder(self.cfg)

        self.delta_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 2),
        )
        self.sign_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.cath_head = nn.Linear(d, self.cfg.cath_classes)

        self.use_foldx_branch = self.cfg.use_foldx_branch
        if self.use_foldx_branch:
            self.foldx_mlp = nn.Sequential(
                nn.Linear(self.cfg.foldx_dim, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1),
            )
            self.foldx_gate = nn.Sequential(
                nn.Linear(d + 1, d // 2),
                nn.GELU(),
                nn.Linear(d // 2, 1),
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        wt_patch = batch["wt_patch"]
        mt_patch = batch["mt_patch"]

        wt_h = self.encoder(wt_patch)
        mt_h = self.encoder(mt_patch)

        delta = mt_h - wt_h
        pool_mask = torch.clamp(mt_patch["mutation_mask"] + mt_patch["first_shell_mask"], 0.0, 1.0)
        pooled = masked_mean(delta, pool_mask, dim=1)

        pred = self.delta_head(pooled)
        ddg_mean = pred[:, 0]
        ddg_logvar = pred[:, 1].clamp(min=-8.0, max=8.0)
        sign_logit = self.sign_head(pooled).squeeze(-1)

        if self.use_foldx_branch and batch.get("foldx") is not None:
            foldx = batch["foldx"].float()
            if foldx.shape[-1] != self.cfg.foldx_dim:
                if foldx.shape[-1] > self.cfg.foldx_dim:
                    foldx = foldx[:, : self.cfg.foldx_dim]
                else:
                    pad = torch.zeros(
                        (foldx.shape[0], self.cfg.foldx_dim - foldx.shape[-1]),
                        device=foldx.device,
                        dtype=foldx.dtype,
                    )
                    foldx = torch.cat([foldx, pad], dim=-1)
            fx_scalar = self.foldx_mlp(foldx).squeeze(-1)
            gate = torch.sigmoid(self.foldx_gate(torch.cat([pooled, fx_scalar.unsqueeze(-1)], dim=-1))).squeeze(-1)
            fmask = batch.get("foldx_mask")
            if fmask is None:
                ddg_mean = gate * ddg_mean + (1.0 - gate) * fx_scalar
            else:
                fmask = fmask.float()
                fused = gate * ddg_mean + (1.0 - gate) * fx_scalar
                ddg_mean = fmask * fused + (1.0 - fmask) * ddg_mean
        else:
            gate = torch.zeros_like(ddg_mean)

        wt_pool = masked_mean(wt_h, torch.clamp(wt_patch["interface_mask"] + wt_patch["mutation_mask"], 0.0, 1.0), dim=1)
        cath_logit = self.cath_head(wt_pool)

        return {
            "ddg_mean": ddg_mean,
            "ddg_logvar": ddg_logvar,
            "sign_logit": sign_logit,
            "aux_outputs": {
                "cath_logit": cath_logit,
                "wt_pooled": wt_pool,
                "delta_pooled": pooled,
                "foldx_gate": gate,
            },
        }
