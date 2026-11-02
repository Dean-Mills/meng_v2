"""
PG-GAT-H (hierarchical PG-GAT) — PEng.

Attention in three levels, matching the three layers of the MEng headline:

  1. Joint level   — PG-GAT layer(s) over the kNN joint graph (SAGATLayer,
                     unchanged: type-pair bias, skeleton-relative position
                     encoding, same-type repulsion head).
  2. Bone level    — nodes are CANDIDATE bones: for every joint, its m nearest
                     detections of each skeletally adjacent type. Built from
                     positions and types only, never from person labels (the
                     TriGAT construction used GT persons; this does not).
                     Bones attend over the line graph: bones sharing a joint,
                     with an edge flag for "competing" (same bone type, i.e.
                     alternative candidates at that joint) vs "chain"
                     (different types, consecutive along the limb).
  3. Triplet level — nodes are pairs of chain bones meeting at a pivot joint,
                     i.e. candidate A-B-C triplets carrying TriGAT's
                     articulation features (angle at the pivot, bone-length
                     ratio). Kept to the top-r bones per (pivot, bone type)
                     by bone validity. Triplets attend over triplets sharing
                     a bone (same competing / chain flag).

Top-down fusion replaces TriGAT's voting: every joint adds the
validity-weighted mean of its incident bone and triplet states to its own
state, then post-fusion PG-GAT layer(s) and the projection produce the usual
per-joint embedding. Grouping read-out (k-means / COP-KM / THA, K selection)
is therefore unchanged.

Bone and triplet validity logits get an optional auxiliary BCE
(label = all endpoints belong to the same person) through `aux_loss`.

Drop-in for SAGATEmbedding: same forward(data) -> [N, output_dim].
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from config import PGGATHierConfig
from sa_gat import (
    COCO_SKELETON, LIMBS, NUM_JOINT_TYPES, NUM_CATEGORIES,
    CAT_SAME_TYPE, CAT_SKELETAL_NEIGHBOR, CAT_SAME_LIMB, CAT_CROSS_BODY,
    SAGATLayer,
)

# The 19-edge official COCO skeleton adds eye-eye and ear-shoulder.
COCO_SKELETON_19 = COCO_SKELETON + [(1, 2), (3, 5), (4, 6)]
SKELETONS = {"coco16": COCO_SKELETON, "coco19": COCO_SKELETON_19}

# sa_gat uses the unreachable-pair sentinel (8) as the hop normaliser; keep it
# fixed so the coco16 tables reproduce sa_gat's edge features exactly.
HOP_SENTINEL = 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton tables
# ─────────────────────────────────────────────────────────────────────────────

def skeleton_tables(skeleton: List[Tuple[int, int]]) -> dict:
    """Joint-pair lookups for the PG-GAT edge features plus bone / triplet types."""
    J = NUM_JOINT_TYPES
    adj = [set() for _ in range(J)]
    for a, b in skeleton:
        adj[a].add(b)
        adj[b].add(a)

    skel_nb = torch.zeros(J, J, dtype=torch.bool)
    for a, b in skeleton:
        skel_nb[a, b] = skel_nb[b, a] = True

    same_limb = torch.zeros(J, J, dtype=torch.bool)
    for limb in LIMBS:
        for a in limb:
            for b in limb:
                if a != b and not skel_nb[a, b]:
                    same_limb[a, b] = True

    hops = torch.full((J, J), HOP_SENTINEL)
    for src in range(J):
        dist = {src: 0}
        queue = [src]
        for node in queue:
            for nb in sorted(adj[node]):
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        for dst, d in dist.items():
            hops[src, dst] = float(d)

    # Bone type = index of the skeleton edge; ends stored in table orientation.
    bone_type = torch.full((J, J), -1, dtype=torch.long)
    for e, (a, b) in enumerate(skeleton):
        bone_type[a, b] = bone_type[b, a] = e

    # Triplet type for an (ordered) pair of bone types meeting at a pivot.
    E = len(skeleton)
    triplet_type = torch.full((E, E), -1, dtype=torch.long)
    n_trip = 0
    for B in range(J):
        nbs = sorted(adj[B])
        for i, A in enumerate(nbs):
            for C in nbs[i + 1:]:
                e1, e2 = int(bone_type[A, B]), int(bone_type[B, C])
                triplet_type[e1, e2] = triplet_type[e2, e1] = n_trip
                n_trip += 1

    return {
        "skel_nb": skel_nb, "same_limb": same_limb, "hops": hops,
        "bone_type": bone_type, "triplet_type": triplet_type,
        "n_bone_types": E, "n_triplet_types": n_trip,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def co_incidence_pairs(container: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    """
    Ordered pairs of distinct items that share a container.

    Args:
        container: [P] container id per incidence
        item:      [P] item id per incidence
    Returns:
        [2, M] (item_i, item_j), i != j
    """
    device = item.device
    if container.numel() < 2:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    order = torch.argsort(container, stable=True)
    c, it = container[order], item[order]
    _, counts = torch.unique_consecutive(c, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    sq = counts * counts
    total = int(sq.sum())
    g = torch.repeat_interleave(torch.arange(counts.numel(), device=device), sq)
    p = torch.arange(total, device=device) - torch.repeat_interleave(torch.cumsum(sq, 0) - sq, sq)
    s = counts[g]
    i = starts[g] + p // s
    j = starts[g] + p % s
    keep = i != j
    return torch.stack([it[i[keep]], it[j[keep]]])


def rank_within_groups(group: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Rank (0 = highest score) of each entry within its group."""
    order = torch.argsort(-score, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    g_sorted = group[order]
    _, counts = torch.unique_consecutive(g_sorted, return_counts=True)
    starts = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    rank = torch.empty_like(group)
    rank[order] = torch.arange(group.numel(), device=group.device) - starts
    return rank


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def add_same_type_edges(edge_index: torch.Tensor, jt: torch.Tensor, n_types: int,
                        batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Union of edge_index with every ordered pair of distinct same-type joints of one graph.

    The kNN-16 joint graph holds only ~22 % of the other people's same-type joints at 9+ people
    (diagnostics 2026-10-26), so the repulsion heads (same-type subset of these edges) miss most of
    the joints they must separate from. Pairs never cross graphs of a batch.
    """
    n = jt.numel()
    group = jt if batch is None else batch * n_types + jt
    same = group.unsqueeze(0) == group.unsqueeze(1)
    same.fill_diagonal_(False)
    s, d = same.nonzero(as_tuple=True)
    key = torch.unique(torch.cat([edge_index[0] * n + edge_index[1], s * n + d]))
    return torch.stack([key // n, key % n])


class PGGATHierEmbedding(nn.Module):
    """Hierarchical PG-GAT embedding network (drop-in for SAGATEmbedding)."""

    def __init__(self, config: PGGATHierConfig):
        super().__init__()
        self.config = c = config
        assert c.num_pre_joint_layers >= 1, "need a joint level before the bone level"
        self.skeleton = SKELETONS[c.skeleton]
        tables = skeleton_tables(self.skeleton)
        self.n_bone_types = tables["n_bone_types"]
        self.n_triplet_types = tables["n_triplet_types"]
        for name in ("skel_nb", "same_limb", "hops", "bone_type", "triplet_type"):
            self.register_buffer(name, tables[name])

        D = c.hidden_dim * c.num_heads   # joint state width (concatenated heads)
        H = c.hier_dim                   # bone / triplet state width
        assert H % c.num_heads == 0
        te = c.type_embedding_dim

        def joint_layer(in_dim: int, concat: bool) -> SAGATLayer:
            return SAGATLayer(
                in_dim=in_dim, out_dim=c.hidden_dim, num_heads=c.num_heads,
                dropout=c.dropout,
                use_type_pair_attention=c.use_type_pair_attention,
                use_position_encoding=c.use_position_encoding,
                use_repulsion_heads=c.use_repulsion_heads,
                n_repulsion_heads=c.n_repulsion_heads,
                concat=concat,
            )

        def norm(dim: int) -> nn.Module:
            return nn.LayerNorm(dim) if c.use_layer_norm else nn.Identity()

        self.joint_embedding = nn.Embedding(c.num_joint_types, c.joint_embedding_dim)

        # Level 1 — joints
        self.pre_layers = nn.ModuleList(
            [joint_layer(c.input_dim if i == 0 else D, True) for i in range(c.num_pre_joint_layers)]
        )
        self.pre_norms = nn.ModuleList([norm(D) for _ in range(c.num_pre_joint_layers)])

        # Level 2 — bones
        self.bone_type_emb = nn.Embedding(self.n_bone_types, te)
        self.bone_encoder = nn.Sequential(nn.Linear(2 * D + 3 + te, H), nn.ELU())
        self.bone_gat = GATv2Conv(H, H // c.num_heads, heads=c.num_heads, concat=True,
                                  dropout=c.dropout, edge_dim=2)
        self.bone_norm = norm(H)
        self.bone_score = nn.Linear(H, 1)
        self.bone_to_joint = nn.Linear(H, D)

        # Level 3 — triplets
        if c.use_triplet_level:
            self.trip_type_emb = nn.Embedding(self.n_triplet_types, te)
            self.trip_encoder = nn.Sequential(nn.Linear(2 * H + 3 + te, H), nn.ELU())
            self.trip_gat = GATv2Conv(H, H // c.num_heads, heads=c.num_heads, concat=True,
                                      dropout=c.dropout, edge_dim=2)
            self.trip_norm = norm(H)
            self.trip_score = nn.Linear(H, 1)
            self.trip_to_joint = nn.Linear(H, D)

        self.fuse_norm = norm(D)

        # Image evidence (optional, PEng #3): frozen backbone, sampled along bones
        # (and optionally at joints). Output layers start at zero, so a warm start
        # from a positions-only checkpoint reproduces it exactly.
        self.needs_image = c.image_backbone is not None
        if self.needs_image:
            from image_evidence import FrozenBackbone
            self.backbone = FrozenBackbone(c.image_backbone)
            Fi, S = c.image_feat_dim, c.bone_image_samples
            self.image_proj = nn.Conv2d(self.backbone.out_channels, Fi, kernel_size=1)
            self.bone_image_encoder = nn.Sequential(nn.Linear(S * Fi, H), nn.ELU(), nn.Linear(H, H))
            nn.init.zeros_(self.bone_image_encoder[-1].weight)
            nn.init.zeros_(self.bone_image_encoder[-1].bias)
            self.register_buffer("bone_t", torch.linspace(0.0, 1.0, S))
            if c.joint_image_features:
                self.joint_image_encoder = nn.Sequential(nn.Linear(Fi, D), nn.ELU(), nn.Linear(D, D))
                nn.init.zeros_(self.joint_image_encoder[-1].weight)
                nn.init.zeros_(self.joint_image_encoder[-1].bias)

        # Read-out — joints again
        n_post = c.num_post_joint_layers
        self.post_layers = nn.ModuleList([joint_layer(D, i < n_post - 1) for i in range(n_post)])
        self.post_norms = nn.ModuleList(
            [norm(D if i < n_post - 1 else c.hidden_dim) for i in range(n_post)]
        )
        self.projection = nn.Linear(c.hidden_dim if n_post > 0 else D, c.output_dim)
        self.final_norm = norm(c.output_dim)
        self.act = nn.ELU()
        self.dropout_layer = nn.Dropout(c.dropout)

        self._aux: Optional[dict] = None

    # ── joint-level edge features (as SAGATEmbedding, skeleton-parameterised) ──

    def _joint_edge_features(self, edge_index, jt, pos):
        src, dst = edge_index
        ts, td = jt[src], jt[dst]
        same_type = ts == td
        feats = []
        if self.config.use_type_pair_attention:
            cat = torch.full_like(ts, CAT_CROSS_BODY)
            cat[self.same_limb[ts, td]] = CAT_SAME_LIMB
            cat[self.skel_nb[ts, td]] = CAT_SKELETAL_NEIGHBOR
            cat[same_type] = CAT_SAME_TYPE
            feats.append(F.one_hot(cat, NUM_CATEGORIES).float())
        if self.config.use_position_encoding:
            spatial = (pos[src] - pos[dst]).norm(dim=1, keepdim=True)
            hop = (self.hops[ts, td] / HOP_SENTINEL).unsqueeze(1)
            feats.append(torch.cat([spatial, hop, same_type.float().unsqueeze(1)], dim=1))
        edge_features = torch.cat(feats, dim=1) if feats else None
        same_type_mask = same_type if self.config.use_repulsion_heads else None
        return edge_features, same_type_mask

    # ── candidate construction (no person labels) ─────────────────────────────

    @torch.no_grad()
    def bone_candidates(self, pos: torch.Tensor, jt: torch.Tensor):
        """Each joint's m nearest detections of every skeletally adjacent type.

        Returns (src, dst, bone_type); src has the first joint type of the
        skeleton edge, dst the second; pairs are unique.
        """
        n = pos.size(0)
        m = self.config.bone_candidates
        dist = torch.cdist(pos, pos)
        srcs, dsts = [], []
        for a, b in self.skeleton:
            ia = (jt == a).nonzero(as_tuple=True)[0]
            ib = (jt == b).nonzero(as_tuple=True)[0]
            if ia.numel() == 0 or ib.numel() == 0:
                continue
            d = dist[ia][:, ib]
            k1 = min(m, ib.numel())
            j1 = d.topk(k1, dim=1, largest=False).indices
            srcs.append(ia.repeat_interleave(k1))
            dsts.append(ib[j1.reshape(-1)])
            k2 = min(m, ia.numel())
            i2 = d.topk(k2, dim=0, largest=False).indices
            srcs.append(ia[i2.reshape(-1)])
            dsts.append(ib.repeat(k2))
        if not srcs:
            empty = torch.zeros(0, dtype=torch.long, device=pos.device)
            return empty, empty, empty
        key = torch.unique(torch.cat(srcs) * n + torch.cat(dsts))
        src, dst = key // n, key % n
        return src, dst, self.bone_type[jt[src], jt[dst]]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, data: Data) -> torch.Tensor:
        c = self.config
        jt = data.joint_types
        pos = data.x[:, :2]
        edge_index = data.edge_index
        if c.same_type_edges:
            edge_index = add_same_type_edges(edge_index, jt, c.num_joint_types, getattr(data, "batch", None))
        x = torch.cat([data.x, self.joint_embedding(jt)], dim=-1)
        jef, st_mask = self._joint_edge_features(edge_index, jt, pos)

        # Level 1 — joints
        h = x
        for layer, nrm in zip(self.pre_layers, self.pre_norms):
            h = self.dropout_layer(self.act(nrm(layer(h, edge_index, jef, st_mask))))

        fmap = None
        if self.needs_image:
            from image_evidence import sample_points
            assert getattr(data, "image", None) is not None, \
                "this PG-GAT-H variant needs graph.image (512 letterbox frame; see image_evidence.attach_images)"
            fmap = self.image_proj(self.backbone(data.image))
            if c.joint_image_features:
                h = h + self.joint_image_encoder(sample_points(fmap, pos))

        # Level 2 — bones
        src, dst, btype = self.bone_candidates(pos, jt)
        B = src.numel()
        self._aux = {"bone_src": src, "bone_dst": dst}
        fused = h
        if B > 0:
            dp = pos[dst] - pos[src]
            g = self.bone_encoder(torch.cat(
                [h[src], h[dst], dp, dp.norm(dim=1, keepdim=True), self.bone_type_emb(btype)], dim=1))
            if fmap is not None:
                t = self.bone_t.view(1, -1, 1)
                along = pos[src].unsqueeze(1) * (1.0 - t) + pos[dst].unsqueeze(1) * t   # [B, S, 2]
                g = g + self.bone_image_encoder(sample_points(fmap, along).flatten(1))
            bone_ids = torch.arange(B, device=h.device)
            inc_node = torch.cat([src, dst])
            inc_bone = torch.cat([bone_ids, bone_ids])
            bpairs = co_incidence_pairs(inc_node, inc_bone)
            competing = (btype[bpairs[0]] == btype[bpairs[1]]).float()
            battr = torch.stack([competing, 1.0 - competing], dim=1)
            g = self.act(self.bone_norm(g + self.bone_gat(g, bpairs, edge_attr=battr)))
            s_bone = self.bone_score(g).squeeze(-1)
            self._aux["bone_logits"] = s_bone

            w = torch.sigmoid(s_bone)
            msg = self.bone_to_joint(g) * w.unsqueeze(1)
            num = torch.zeros_like(h).index_add_(0, inc_node, torch.cat([msg, msg]))
            den = torch.zeros(h.size(0), device=h.device).index_add_(0, inc_node, torch.cat([w, w]))
            fused = fused + num / (den.unsqueeze(1) + 1e-6)

            # Level 3 — triplets (chain bone pairs meeting at a pivot)
            if c.use_triplet_level:
                fused = fused + self._triplet_level(g, s_bone, src, dst, btype, bpairs, pos, h)

        h = self.dropout_layer(self.act(self.fuse_norm(fused)))

        # Read-out — joints
        for i, (layer, nrm) in enumerate(zip(self.post_layers, self.post_norms)):
            h = self.act(nrm(layer(h, edge_index, jef, st_mask)))
            if i < len(self.post_layers) - 1:
                h = self.dropout_layer(h)

        emb = self.final_norm(self.projection(h))
        if c.l2_normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def _triplet_level(self, g, s_bone, src, dst, btype, bpairs, pos, h):
        c = self.config
        b1, b2 = bpairs
        chain = btype[b1] < btype[b2]           # distinct types, one orientation
        b1, b2 = b1[chain], b2[chain]
        if b1.numel() == 0:
            return torch.zeros_like(h)
        # pivot = the joint the two bones share; outer joints A (from b1), C (from b2)
        piv = torch.where((src[b1] == src[b2]) | (src[b1] == dst[b2]), src[b1], dst[b1])
        A = torch.where(src[b1] == piv, dst[b1], src[b1])
        C = torch.where(src[b2] == piv, dst[b2], src[b2])

        if c.triplet_top_r > 0:
            B = src.numel()
            inc_node = torch.cat([src, dst])
            inc_bone = torch.cat([torch.arange(B, device=g.device)] * 2)
            group = inc_node * self.n_bone_types + btype[inc_bone]
            rank = rank_within_groups(group, s_bone.detach()[inc_bone])

            def rank_at(bone, node):
                return torch.where(src[bone] == node, rank[bone], rank[bone + B])

            keep = (rank_at(b1, piv) < c.triplet_top_r) & (rank_at(b2, piv) < c.triplet_top_r)
            b1, b2, piv, A, C = b1[keep], b2[keep], piv[keep], A[keep], C[keep]
            if b1.numel() == 0:
                return torch.zeros_like(h)

        T = b1.numel()
        u, v = pos[A] - pos[piv], pos[C] - pos[piv]
        lu, lv = u.norm(dim=1).clamp(min=1e-6), v.norm(dim=1).clamp(min=1e-6)
        cos = ((u * v).sum(dim=1) / (lu * lv)).clamp(-1.0, 1.0)
        sin = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / (lu * lv)
        geo = torch.stack([cos, sin, torch.log(lu / lv)], dim=1)
        ttype = self.triplet_type[btype[b1], btype[b2]]
        t = self.trip_encoder(torch.cat([g[b1], g[b2], geo, self.trip_type_emb(ttype)], dim=1))

        trip_ids = torch.arange(T, device=g.device)
        tpairs = co_incidence_pairs(torch.cat([b1, b2]), torch.cat([trip_ids, trip_ids]))
        competing = (ttype[tpairs[0]] == ttype[tpairs[1]]).float()
        tattr = torch.stack([competing, 1.0 - competing], dim=1)
        t = self.act(self.trip_norm(t + self.trip_gat(t, tpairs, edge_attr=tattr)))
        s_trip = self.trip_score(t).squeeze(-1)
        self._aux.update({"trip_logits": s_trip, "trip_nodes": torch.stack([A, piv, C], dim=1)})

        w = torch.sigmoid(s_trip)
        msg = self.trip_to_joint(t) * w.unsqueeze(1)
        inc_node = torch.cat([A, piv, C])
        num = torch.zeros_like(h).index_add_(0, inc_node, torch.cat([msg, msg, msg]))
        den = torch.zeros(h.size(0), device=h.device).index_add_(0, inc_node, torch.cat([w, w, w]))
        return num / (den.unsqueeze(1) + 1e-6)

    # ── auxiliary supervision ─────────────────────────────────────────────────

    def aux_loss(self, person_labels: torch.Tensor) -> torch.Tensor:
        """BCE on bone / triplet validity from the last forward (label = same person)."""
        c = self.config
        loss = torch.zeros((), device=person_labels.device)
        a = self._aux or {}
        if c.aux_bone_weight > 0 and a.get("bone_logits") is not None:
            y = (person_labels[a["bone_src"]] == person_labels[a["bone_dst"]]).float()
            loss = loss + c.aux_bone_weight * F.binary_cross_entropy_with_logits(a["bone_logits"], y)
        if c.aux_triplet_weight > 0 and a.get("trip_logits") is not None:
            p = person_labels[a["trip_nodes"]]
            y = ((p[:, 0] == p[:, 1]) & (p[:, 1] == p[:, 2])).float()
            loss = loss + c.aux_triplet_weight * F.binary_cross_entropy_with_logits(a["trip_logits"], y)
        return loss
