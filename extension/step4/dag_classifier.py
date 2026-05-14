"""
Extension Step 4 — DAGNN Classifier for Task Graph Realization
==============================================================
Classifies whether a task graph realization (built by B3) represents a correct
or incorrect recipe execution.

Architecture (faithful to Thost & Chen, ICLR 2021):
    text_feats + vis_feats  →  NodeFusionProjector  →  x (N, 256)
        → Linear(256 → hidden_dim) + ReLU                  [input projection, h^0]
        → DAGNNConv × L                                     [attention + GRU, h^1..h^L]
        → concat(h^0..h^L) per node  →  MaxPool on targets [readout, Eq. 8]
        → Linear(hidden_dim*(L+1) → 1)                     [classifier]
        → BCEWithLogitsLoss

Key differences from a plain GCN / previous implementation:
  - Topological processing order: h_u^l is available when computing h_v^l (u predecessor of v)
  - Attention aggregation (Eq. 5-6): weighted sum with additive query-key scores
  - GRU combine (Eq. 7): message is the hidden state, past node feature is the input
  - Readout only on target nodes (no successors), concat all layers (Eq. 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

from extension.step3.realization_builder import NodeFusionProjector


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def topological_sort(edge_index: torch.Tensor, num_nodes: int) -> list[int]:
    """
    Returns node indices in topological order (sources first).
    edge_index[0] = src, edge_index[1] = dst  (src → dst).
    Falls back to natural order if a cycle is detected (should not happen on DAGs).
    """
    in_degree = [0] * num_nodes
    adj: list[list[int]] = [[] for _ in range(num_nodes)]

    if edge_index.numel() > 0:
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj[u].append(v)
            in_degree[v] += 1

    queue = [n for n in range(num_nodes) if in_degree[n] == 0]
    order: list[int] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != num_nodes:
        return list(range(num_nodes))
    return order


def target_nodes(edge_index: torch.Tensor, num_nodes: int) -> list[int]:
    """
    Returns indices of nodes with no outgoing edges (no successors).
    These are the 'target' nodes T in the DAGNN readout (Eq. 8): they have
    digested information from the entire graph via the topological chain.
    """
    has_successor = set()
    if edge_index.numel() > 0:
        for u in edge_index[0].tolist():
            has_successor.add(u)
    return [v for v in range(num_nodes) if v not in has_successor]


# ---------------------------------------------------------------------------
# DAGNNConv — one DAGNN layer (Eq. 5-7)
# ---------------------------------------------------------------------------

class DAGNNConv(nn.Module):
    """
    One DAGNN propagation layer (Thost & Chen, ICLR 2021, Eq. 5-7).

    For each node v, processed after all its predecessors P(v):

      Attention (Eq. 5-6):
        score_u  = w1 · h_v^{l-1}  +  w2 · h_u^l        for each u in P(v)
        alpha_u  = softmax(score_u) over u in P(v)
        m_v^l    = sum_u( alpha_u * h_u^l )              [weighted message]

      Combine (Eq. 7):
        h_v^l = GRUCell( input = h_v^{l-1},  hx = m_v^l )

    Source nodes (no predecessors): m_v^l = 0  →  h_v^l = GRUCell(h_v^{l-1}, 0)

    Note on GRU role inversion vs standard usage:
        Standard GRU: input = new data, hx = memory
        DAGNN:        input = past node repr (h_v^{l-1}), hx = aggregated predecessors (m_v^l)
        This means the "memory" tracks what the predecessors communicated, while the
        "input" is what this node was before being updated — a natural fit for DAGs.

    Args:
        hidden_dim: feature dimension (same for input and output — all layers operate
                    in the same space after the initial projection in DAGClassifier)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Additive attention parameters (one scalar per node pair, per layer)
        self.w1 = nn.Parameter(torch.empty(hidden_dim))  # query: dot with h_v^{l-1}
        self.w2 = nn.Parameter(torch.empty(hidden_dim))  # key:   dot with h_u^l
        # GRU cell: input and hidden have the same dimension
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.unsqueeze(0))
        nn.init.xavier_uniform_(self.w2.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,           # (N, hidden_dim) — node features at layer l-1
        edge_index: torch.Tensor,  # (2, E) — directed edges src → dst
        topo_order: list[int],     # nodes in topological order (precomputed)
    ) -> torch.Tensor:             # (N, hidden_dim) — updated node features at layer l
        N = x.size(0)

        # Build predecessor list: preds[v] = [u1, u2, ...] such that u_i → v
        preds: dict[int, list[int]] = {v: [] for v in range(N)}
        if edge_index.numel() > 0:
            for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                preds[v].append(u)

        # Store outputs in a dict (not a pre-allocated tensor) so that each
        # node's grad_fn is preserved for backprop through the predecessor chain.
        node_out: dict[int, torch.Tensor] = {}

        for v in topo_order:
            x_v = x[v]  # h_v^{l-1}, shape (hidden_dim,)

            if preds[v]:
                # Stack current-layer representations of predecessors
                # h_u^l is already in node_out[u] because u comes before v
                pred_h = torch.stack([node_out[u] for u in preds[v]], dim=0)  # (K, D)

                # Additive attention scores (Eq. 6)
                # w1 · h_v^{l-1}: scalar, broadcast over all K predecessors
                # w2 · h_u^l:     (K,) vector
                scores = (self.w1 * x_v).sum() + pred_h @ self.w2  # (K,)
                alpha = F.softmax(scores, dim=0)                    # (K,)

                # Weighted message (Eq. 5)
                m_v = (alpha.unsqueeze(1) * pred_h).sum(dim=0)     # (D,)
            else:
                # Source node: no predecessors → zero message
                m_v = x.new_zeros(self.hidden_dim)

            # GRU combine (Eq. 7): input = h_v^{l-1}, hidden state = m_v
            # GRUCell expects (batch=1, dim) tensors
            h_v = self.gru(x_v.unsqueeze(0), m_v.unsqueeze(0)).squeeze(0)  # (D,)
            node_out[v] = h_v

        return torch.stack([node_out[v] for v in range(N)], dim=0)  # (N, hidden_dim)


# ---------------------------------------------------------------------------
# DAGClassifier — full model
# ---------------------------------------------------------------------------

class DAGClassifier(nn.Module):
    """
    Graph-level binary classifier for task graph realizations.

    The NodeFusionProjector (Linear 512→256, from extension.step3.realization_builder)
    is imported and optimised end-to-end with the GNN.

    Args:
        in_channels:  raw node feature dim after fusion (256 from EgoVLP, post-projector)
        hidden_dim:   hidden dim for all DAGNN layers (default 128)
        num_layers:   number of DAGNNConv layers (default 2)
        dropout:      dropout before the classifier head (default 0.5)

    Forward signature (compatible with PyG DataLoader batching):
        logits = model(text_feats, vis_feats, matched_mask, edge_index, batch, ptr)

    where `ptr` is the cumulative node-count array produced by DataLoader (used to
    extract per-graph edge_index slices for topological sort).
    """

    def __init__(
        self,
        in_channels: int  = 256,
        hidden_dim:  int  = 128,
        num_layers:  int  = 2,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout

        # NodeFusionProjector from extension.step3: fuses text (256) + visual (256) → 256
        self.fusion = NodeFusionProjector(in_channels)

        # Input projection: 256 → hidden_dim  (produces h^0)
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # L DAGNN layers, all hidden_dim → hidden_dim
        self.convs = nn.ModuleList(
            [DAGNNConv(hidden_dim) for _ in range(num_layers)]
        )

        # Readout: MaxPool over target nodes, concat h^0..h^L
        # → vector of size hidden_dim * (num_layers + 1)
        readout_dim = hidden_dim * (num_layers + 1)
        self.head = nn.Linear(readout_dim, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        text_feats:   torch.Tensor,  # (N_total, 256) — EgoVLP text embeddings
        vis_feats:    torch.Tensor,  # (N_total, 256) — visual embeddings (0 if unmatched)
        matched_mask: torch.Tensor,  # (N_total,) bool — True where visual info exists
        edge_index:   torch.Tensor,  # (2, E_total)    — all edges in the batch
        batch:        torch.Tensor,  # (N_total,)       — node-to-graph assignment
        ptr:          torch.Tensor,  # (B+1,)           — cumulative node counts per graph
    ) -> torch.Tensor:               # (B, 1)           — one logit per graph
        # --- NodeFusionProjector (from extension.step3) ---
        # Matched nodes: concat([text, vis]) → 256; unmatched: concat([text, zeros]) → 256
        x = F.relu(self.fusion(text_feats, vis_feats))      # (N_total, 256)

        # --- Input projection → h^0 ---
        x = F.relu(self.input_proj(x))  # (N_total, hidden_dim)

        # --- DAGNN layers with multi-layer readout ---
        # We collect the node features after each layer (including h^0) for the readout.
        layer_feats: list[torch.Tensor] = [x]  # h^0

        for conv in self.convs:
            x = self._dagnn_batched(conv, x, edge_index, batch, ptr)
            layer_feats.append(x)  # h^1, h^2, ...

        # --- Readout (Eq. 8): concat all layers, MaxPool over target nodes ---
        # Concat h^0..h^L per node → (N_total, hidden_dim*(L+1))
        node_repr = torch.cat(layer_feats, dim=-1)  # (N_total, readout_dim)

        # MaxPool over target nodes only (per graph)
        graph_emb = self._target_maxpool(node_repr, edge_index, batch, ptr)  # (B, readout_dim)

        # --- Classifier ---
        graph_emb = F.dropout(graph_emb, p=self.dropout, training=self.training)
        return self.head(graph_emb)  # (B, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dagnn_batched(
        self,
        conv:       DAGNNConv,
        x:          torch.Tensor,  # (N_total, F)
        edge_index: torch.Tensor,  # (2, E_total)
        batch:      torch.Tensor,  # (N_total,)
        ptr:        torch.Tensor,  # (B+1,)
    ) -> torch.Tensor:
        """
        Apply one DAGNNConv to a batch of graphs.

        PyG batches graphs by offsetting node indices, so we split by graph,
        compute topological sort per graph, apply DAGNN, then re-stack.
        """
        B = ptr.size(0) - 1
        out_parts: list[torch.Tensor] = []

        for g in range(B):
            lo = ptr[g].item()      # first node index of graph g (global)
            hi = ptr[g + 1].item()  # first node index of graph g+1 (global)
            N_g = hi - lo

            x_g = x[lo:hi]  # (N_g, F) — local node features

            # Extract and re-index edges for graph g
            if edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                mask = (src >= lo) & (src < hi)   # edges within graph g
                local_ei = torch.stack([
                    src[mask] - lo,
                    dst[mask] - lo,
                ], dim=0)                          # (2, E_g), 0-indexed
            else:
                local_ei = torch.zeros(2, 0, dtype=torch.long, device=x.device)

            topo = topological_sort(local_ei, N_g)
            out_parts.append(conv(x_g, local_ei, topo))  # (N_g, hidden_dim)

        return torch.cat(out_parts, dim=0)  # (N_total, hidden_dim)

    def _target_maxpool(
        self,
        node_repr:  torch.Tensor,  # (N_total, D)
        edge_index: torch.Tensor,  # (2, E_total)
        batch:      torch.Tensor,  # (N_total,)
        ptr:        torch.Tensor,  # (B+1,)
    ) -> torch.Tensor:             # (B, D)
        """
        For each graph, identify its target nodes (no outgoing edges) and
        MaxPool their representations.  Falls back to global MaxPool if all
        nodes happen to have successors (degenerate case).
        """
        B = ptr.size(0) - 1
        D = node_repr.size(1)
        graph_embs: list[torch.Tensor] = []

        for g in range(B):
            lo = ptr[g].item()
            hi = ptr[g + 1].item()
            N_g = hi - lo

            if edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                mask = (src >= lo) & (src < hi)
                local_src = (src[mask] - lo).tolist()
            else:
                local_src = []

            has_successor = set(local_src)
            tgt = [v for v in range(N_g) if v not in has_successor]

            if not tgt:  # fallback: all nodes
                tgt = list(range(N_g))

            tgt_repr = node_repr[lo + torch.tensor(tgt, device=node_repr.device)]  # (|T|, D)
            graph_embs.append(tgt_repr.max(dim=0).values)  # (D,)

        return torch.stack(graph_embs, dim=0)  # (B, D)
