"""
Extension Step 3 — Task Graph Loader
=====================================
Loads CaptainCook4D task graphs from the annotations submodule and exposes
them as TaskGraph objects: a list of step texts (nodes) and DAG edges, with
START/END sentinel nodes removed and indices remapped to 0..N-1.

Task graph JSON format (annotations/task_graphs/<recipe>.json):
  {
    "steps": {"0": "START", "1": "Crack two eggs ...", ..., "15": "END"},
    "edges": [[0, 3], [3, 7], ...]   # integer [from, to] pairs
  }

Usage:
  graphs = load_all_task_graphs(
      graphs_dir  = f"{REPO_DIR}/annotations/task_graphs",
      annotations = f"{REPO_DIR}/annotations/annotation_json/complete_step_annotations.json",
  )
  g = graphs[activity_id]      # TaskGraph for recipe with that id
  print(g.nodes)               # ["Crack two eggs ...", "Whisk eggs ...", ...]
  print(g.edges)               # [(0, 1), (1, 2), ...]
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TaskGraph:
    """Represents one recipe task graph with START/END nodes removed."""

    activity_id:   int
    activity_name: str
    nodes:         List[str]               # step text descriptions, 0-indexed
    edges:         List[Tuple[int, int]]   # DAG edges in remapped indices
    # original JSON node IDs corresponding to each entry in `nodes` (for debug)
    original_ids:  List[int] = field(default_factory=list)

    def __repr__(self):
        return (
            f"TaskGraph(activity={self.activity_id} '{self.activity_name}', "
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )


def _activity_name_to_filename(activity_name: str) -> str:
    """
    Convert activity_name from annotations to the task graph filename stem.
    e.g. "Microwave Egg Sandwich" -> "microwaveeggsandwich"
    """
    return activity_name.lower().replace(" ", "")


def load_task_graph(graphs_dir: str, activity_id: int, activity_name: str) -> TaskGraph:
    """
    Load and parse a single task graph JSON file.

    START and END sentinel nodes are filtered out. Remaining node IDs are
    remapped to contiguous indices 0..N-1 so they can be used directly as
    GNN node indices.

    Args:
        graphs_dir:    path to annotations/task_graphs/ folder
        activity_id:   integer recipe ID (used only for labelling)
        activity_name: human-readable name (used to find the JSON file)

    Returns:
        TaskGraph with cleaned nodes and remapped edges.

    Raises:
        FileNotFoundError if the JSON for this recipe does not exist.
    """
    filename = _activity_name_to_filename(activity_name) + ".json"
    path = os.path.join(graphs_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Task graph not found for '{activity_name}' (tried: {path})"
        )

    with open(path) as f:
        data = json.load(f)

    raw_steps: Dict[str, str] = data["steps"]   # {"0": "START", "1": "text", ...}
    raw_edges: List[List[int]] = data["edges"]   # [[from, to], ...]

    # Filter out sentinel nodes (START / END) — they carry no textual meaning
    # and cannot be encoded by the text encoder in a meaningful way.
    sentinel_ids = {
        int(k) for k, v in raw_steps.items()
        if v.strip().upper() in ("START", "END")
    }

    # Build contiguous re-indexing: old_id -> new_idx (only for kept nodes)
    kept = [(int(k), v) for k, v in raw_steps.items() if int(k) not in sentinel_ids]
    kept.sort(key=lambda x: x[0])   # sort by original ID for deterministic order

    old_to_new: Dict[int, int] = {old_id: new_idx for new_idx, (old_id, _) in enumerate(kept)}
    nodes       = [text for _, text in kept]
    original_ids = [old_id for old_id, _ in kept]

    # Remap edges: skip any edge touching a sentinel node
    edges: List[Tuple[int, int]] = []
    for src, dst in raw_edges:
        if src in sentinel_ids or dst in sentinel_ids:
            continue
        if src not in old_to_new or dst not in old_to_new:
            continue   # defensive: skip unknown node IDs
        edges.append((old_to_new[src], old_to_new[dst]))

    return TaskGraph(
        activity_id=activity_id,
        activity_name=activity_name,
        nodes=nodes,
        edges=edges,
        original_ids=original_ids,
    )


def load_all_task_graphs(graphs_dir: str, annotations_path: str) -> Dict[int, TaskGraph]:
    """
    Load task graphs for all recipes present in complete_step_annotations.json.

    Reads the annotations to discover the (activity_id, activity_name) pairs,
    then loads each corresponding task graph JSON.  Recipes whose JSON is
    missing are skipped with a warning (so the pipeline stays robust).

    Args:
        graphs_dir:       path to annotations/task_graphs/
        annotations_path: path to annotations/annotation_json/complete_step_annotations.json

    Returns:
        Dict mapping activity_id (int) -> TaskGraph.
    """
    with open(annotations_path) as f:
        data = json.load(f)

    # Collect unique (activity_id, activity_name) pairs from all recordings
    recipe_map: Dict[int, str] = {}
    for info in data.values():
        aid   = info["activity_id"]
        aname = info["activity_name"]
        if aid not in recipe_map:
            recipe_map[aid] = aname

    task_graphs: Dict[int, TaskGraph] = {}
    n_missing = 0

    for activity_id, activity_name in sorted(recipe_map.items()):
        try:
            tg = load_task_graph(graphs_dir, activity_id, activity_name)
            task_graphs[activity_id] = tg
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            n_missing += 1

    print(
        f"Loaded {len(task_graphs)} task graphs "
        f"({n_missing} missing) from {graphs_dir}"
    )
    return task_graphs


def print_graph_stats(task_graphs: Dict[int, "TaskGraph"]) -> None:
    """
    Print per-recipe graph statistics and highlight isolated nodes.

    An isolated node has no incoming and no outgoing edges after START/END
    removal.  These nodes receive no message passing in GNN layers and
    contribute only their initial features to the readout — important to
    know before choosing the GNN architecture in Step 4.
    """
    total_isolated = 0

    print(f"\n{'act':>4}  {'recipe':<32}  {'nodes':>5}  {'edges':>5}  {'isolated':>8}")
    print("-" * 62)

    for aid, g in sorted(task_graphs.items()):
        connected = set()
        for src, dst in g.edges:
            connected.add(src)
            connected.add(dst)
        isolated = [i for i in range(len(g.nodes)) if i not in connected]
        total_isolated += len(isolated)

        flag = "  ⚠" if isolated else ""
        print(
            f"{aid:>4}  {g.activity_name:<32}  {len(g.nodes):>5}  "
            f"{len(g.edges):>5}  {len(isolated):>8}{flag}"
        )
        if isolated:
            for i in isolated:
                print(f"       └─ node {i}: '{g.nodes[i][:55]}'")

    print("-" * 62)
    print(
        f"      {'TOTAL':<32}  "
        f"{sum(len(g.nodes) for g in task_graphs.values()):>5}  "
        f"{sum(len(g.edges) for g in task_graphs.values()):>5}  "
        f"{total_isolated:>8}"
    )
