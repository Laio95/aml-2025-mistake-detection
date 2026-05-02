"""
build_step_embeddings.py
========================
Step 1 Extension — Recipe Step Localization (output stage).

Reads the detections produced by eval.py (eval_results.pkl) and computes
one step-level embedding per detected step by mean-pooling the EgoVLP
features that fall within the (t_start, t_end) boundaries of each detection.

Pipeline position:
    train.py  →  eval.py --saveonly --all_splits
                    └─ eval_results.pkl
                            │
                    build_step_embeddings.py  ← this script
                            │
                    {recording_id}_step_embeddings.npz  (one per video)
                            │
                    Step 12: Task Verification (Transformer + LOO)
                    Step 13: Hungarian Matching (task graph nodes)

Output format per recording_id ({recording_id}_step_embeddings.npz):
    step_embeddings  (N, 256) float32 — mean-pooled EgoVLP embedding per step
    step_intervals   (N, 2)   float32 — [[t_start, t_end], ...] in seconds
    step_ids         (N,)     int32   — ActionFormer label_id (0-352)
    step_scores      (N,)     float32 — ActionFormer confidence score

Feature-to-time mapping (1 feature = 1 second):
    i_start = floor(t_start)          # first feature index included
    i_end   = min(ceil(t_end), T)     # first feature index excluded
    embedding = features[i_start:i_end].mean(axis=0)  → shape (256,)
"""

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────────────────────────────────────

def load_egovlp_features(feat_folder: str, recording_id: str,
                          file_suffix: str = '_360p',
                          file_ext: str = '.npz') -> np.ndarray | None:
    """
    Load EgoVLP features for a given recording_id.

    Expected filename: {recording_id}{file_suffix}.mp4_1s_1s{file_ext}
    Example:  1_7_360p.mp4_1s_1s.npz

    Returns:
        np.ndarray of shape (T, 256) float32, or None if file not found.
    """
    filename = f"{recording_id}{file_suffix}.mp4_1s_1s{file_ext}"
    path = Path(feat_folder) / filename

    if not path.exists():
        print(f"  [WARN] Feature file not found: {path}")
        return None

    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())

    # Try common key names in order of priority
    if 'video_features' in keys:
        feats = data['video_features']
    elif 'arr_0' in keys:
        feats = data['arr_0']
    else:
        feats = data[keys[0]]

    return feats.astype(np.float32)  # (T, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Mean pooling
# ─────────────────────────────────────────────────────────────────────────────

def mean_pool_segment(features: np.ndarray,
                      t_start: float,
                      t_end: float) -> np.ndarray:
    """
    Mean-pool EgoVLP features within [t_start, t_end] seconds.

    Mapping: 1 feature = 1 second (feat_stride=30 at 29.97fps).
        i_start = floor(t_start)
        i_end   = min(ceil(t_end), T)   [exclusive upper bound]

    If the resulting slice is empty (degenerate segment), falls back to
    the single nearest feature.

    Returns:
        np.ndarray of shape (256,) float32.
    """
    T = features.shape[0]
    i_start = max(0, min(int(math.floor(t_start)), T - 1))
    i_end   = max(i_start + 1, min(int(math.ceil(t_end)), T))
    return features[i_start:i_end].mean(axis=0)  # (256,)


# ─────────────────────────────────────────────────────────────────────────────
# Detection loading
# ─────────────────────────────────────────────────────────────────────────────

def load_detections(pkl_path: str,
                    score_threshold: float = 0.001
                    ) -> dict[str, list[dict]]:
    """
    Load ActionFormer detections from eval_results.pkl.

    The .pkl produced by valid_one_epoch (actionformer_release) has format:
        {
          "video-id": List[str],
          "t-start":  List[float],
          "t-end":    List[float],
          "label":    List[int],   # step_id (0-352)
          "score":    List[float],
        }

    Returns:
        Dict {recording_id: [{"t_start", "t_end", "label_id", "score"}, ...]}
        sorted by t_start within each video.
    """
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    detections: dict[str, list[dict]] = {}
    n_total, n_kept = 0, 0

    for vid, ts, te, lab, sc in zip(
        raw['video-id'], raw['t-start'], raw['t-end'],
        raw['label'],    raw['score']
    ):
        n_total += 1
        if float(sc) < score_threshold:
            continue
        n_kept += 1
        detections.setdefault(str(vid), []).append({
            't_start' : float(ts),
            't_end'   : float(te),
            'label_id': int(lab),
            'score'   : float(sc),
        })

    for vid in detections:
        detections[vid].sort(key=lambda d: d['t_start'])

    print(f"Loaded {n_kept}/{n_total} detections (score >= {score_threshold}) "
          f"across {len(detections)} videos.")
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────────────────────────────────────

def build_step_embeddings(
    pkl_path:        str,
    feat_folder:     str,
    output_dir:      str,
    file_suffix:     str   = '_360p',
    file_ext:        str   = '.npz',
    score_threshold: float = 0.001,
) -> dict[str, str | None]:
    """
    Build and save step-level embeddings for all videos in eval_results.pkl.

    Args:
        pkl_path:        path to eval_results.pkl from eval.py --saveonly
        feat_folder:     directory containing EgoVLP .npz feature files
        output_dir:      where to save {recording_id}_step_embeddings.npz
        file_suffix:     suffix in feature filename (default: '_360p')
        file_ext:        feature file extension (default: '.npz')
        score_threshold: discard detections with score below this value

    Returns:
        Dict {recording_id: output_path | None}
    """
    detections = load_detections(pkl_path, score_threshold)
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str | None] = {}
    total = len(detections)

    for i, (recording_id, dets) in enumerate(detections.items()):

        # Load EgoVLP features (T, 256)
        feats = load_egovlp_features(feat_folder, recording_id, file_suffix, file_ext)
        if feats is None:
            results[recording_id] = None
            continue

        if not dets:
            results[recording_id] = None
            continue

        # Compute one embedding per detection via mean pooling
        embeddings = np.stack([
            mean_pool_segment(feats, d['t_start'], d['t_end'])
            for d in dets
        ])                                                           # (N, 256)

        intervals = np.array(
            [[d['t_start'], d['t_end']] for d in dets],
            dtype=np.float32
        )                                                            # (N, 2)

        step_ids = np.array(
            [d['label_id'] for d in dets], dtype=np.int32
        )                                                            # (N,)

        step_scores = np.array(
            [d['score'] for d in dets], dtype=np.float32
        )                                                            # (N,)

        out_path = out_dir / f"{recording_id}_step_embeddings.npz"
        np.savez(
            out_path,
            step_embeddings = embeddings,    # (N, 256) — input for steps 12/13
            step_intervals  = intervals,     # (N, 2)   — [t_start, t_end] seconds
            step_ids        = step_ids,      # (N,)     — ActionFormer label_id
            step_scores     = step_scores,   # (N,)     — confidence score
            recording_id    = np.array(recording_id),
        )
        results[recording_id] = str(out_path)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {recording_id}: "
                  f"{embeddings.shape[0]} steps → {out_path.name}")

    ok = sum(v is not None for v in results.values())
    print(f"\nDone. Saved {ok}/{total} step embedding files in '{out_dir}'.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────────────────

def verify_output(output_dir: str, n_samples: int = 3) -> None:
    """
    Quick sanity check: load a few .npz files and print their shapes.
    """
    out_dir = Path(output_dir)
    files   = sorted(out_dir.glob('*_step_embeddings.npz'))

    if not files:
        print("No output files found.")
        return

    print(f"\nVerification ({min(n_samples, len(files))} samples):")
    for path in files[:n_samples]:
        data = np.load(path, allow_pickle=True)
        embs = data['step_embeddings']
        ivs  = data['step_intervals']
        print(f"  {path.name}")
        print(f"    step_embeddings : {embs.shape}  "
              f"(mean norm: {np.linalg.norm(embs, axis=1).mean():.3f})")
        print(f"    step_intervals  : {ivs.shape}  "
              f"(t range: [{ivs[:,0].min():.1f}s – {ivs[:,1].max():.1f}s])")


################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build step-level EgoVLP embeddings from ActionFormer detections.'
    )
    parser.add_argument('--pkl', type=str, required=True,
                        help='path to eval_results.pkl from eval.py --saveonly')
    parser.add_argument('--feat_folder', type=str, required=True,
                        help='directory with EgoVLP .npz feature files '
                             '(e.g. .../features_egovlp_num_frames_16)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='where to save {recording_id}_step_embeddings.npz files')
    parser.add_argument('--file_suffix', type=str, default='_360p',
                        help='suffix in feature filename (default: _360p)')
    parser.add_argument('--file_ext', type=str, default='.npz',
                        help='feature file extension (default: .npz)')
    parser.add_argument('--score_threshold', type=float, default=0.001,
                        help='minimum confidence score for detections (default: 0.001)')
    parser.add_argument('--verify', action='store_true',
                        help='print a quick sanity check on the output files')

    args = parser.parse_args()

    results = build_step_embeddings(
        pkl_path        = args.pkl,
        feat_folder     = args.feat_folder,
        output_dir      = args.output_dir,
        file_suffix     = args.file_suffix,
        file_ext        = args.file_ext,
        score_threshold = args.score_threshold,
    )

    if args.verify:
        verify_output(args.output_dir)
