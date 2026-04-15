"""
EgoVLP segment feature extractor for CaptainCook4D.

Adapts the feature-extraction pipeline proposed by CaptainCook4D
(feature_extractors/frame_features/extract_features.py) to use the
EgoVLP FrozenInTime video encoder as the backbone.

Output layout mirrors the Omnivore/SlowFast convention used by
dataloader/CaptainCookStepDataset.py:
    <output_features_path>/<recording_id>_360p.mp4_1s_1s.npz
where arr_0 is an [N, 256] float32 numpy array of per-1s features.

Usage (Colab):
    python extract_egovlp_features.py \
        --egovlp_repo /content/egovlp \
        --egovlp_ckpt /content/drive/MyDrive/AML_Project/models/egovlp.pth \
        --videos_dir /content/drive/MyDrive/AML_Project/CaptainCook4D/captain_cook_4d_gopro_resized_extracted \
        --output_dir /content/drive/MyDrive/AML_Project/CaptainCook4D/features/egovlp \
        --num_frames 4 --fps 30
"""

import argparse
import os
import sys
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo


def build_egovlp_model(egovlp_repo: str, ckpt_path: str, num_frames: int, device: torch.device):
    """Build FrozenInTime and load the EgoVLP checkpoint manually.

    We bypass FrozenInTime's built-in checkpoint loading because it uses
    os.environ['LOCAL_RANK'], which is not set in Colab.
    """
    if egovlp_repo not in sys.path:
        sys.path.insert(0, egovlp_repo)
    # pretrained/ must exist and contain jx_vit_base_p16_224-80ecf9dd.pth
    os.chdir(egovlp_repo)

    from model.model import FrozenInTime  # noqa: E402
    from utils.util import state_dict_data_parallel_fix  # noqa: E402

    video_params = {
        "model": "SpaceTimeTransformer",
        "arch_config": "base_patch16_224",
        "num_frames": num_frames,
        "pretrained": True,
        "time_init": "zeros",
    }
    text_params = {
        "model": "distilbert-base-uncased",
        "pretrained": True,
        "input": "text",
    }

    model = FrozenInTime(
        video_params=video_params,
        text_params=text_params,
        projection_dim=256,
        load_checkpoint=None,  # load manually below
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
    # Inflate temporal embeds if shapes differ
    new_state_dict = model._inflate_positional_embeds(new_state_dict)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logging.info(f"Loaded EgoVLP checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.eval().to(device)
    return model


def preprocess_clip(clip: torch.Tensor, num_frames: int, image_size: int = 224) -> torch.Tensor:
    """Take a pytorchvideo clip tensor [C, T, H, W] (0-255 uint8/float)
    and return an EgoVLP video input [1, T, C, H, W], normalized to [0,1]
    and ImageNet-normalized, resized to (image_size, image_size).
    """
    if clip.dtype != torch.float32:
        clip = clip.float()
    # clip: [C, T, H, W]
    C, T, H, W = clip.shape
    # Uniformly sample num_frames along T
    if T >= num_frames:
        idxs = torch.linspace(0, T - 1, num_frames).long()
    else:
        # Pad by repeating last frame
        idxs = torch.cat([torch.arange(T), torch.full((num_frames - T,), T - 1)]).long()
    clip = clip[:, idxs, :, :]  # [C, num_frames, H, W]

    # Normalize 0-255 -> 0-1
    clip = clip / 255.0
    # Resize spatial dims to image_size x image_size
    # F.interpolate expects [N, C, H, W]; reshape temporally
    clip = clip.permute(1, 0, 2, 3)  # [T, C, H, W]
    clip = F.interpolate(clip, size=(image_size, image_size), mode="bilinear", align_corners=False)
    # Standard ImageNet mean/std (EgoVLP inherits ViT-B normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    clip = (clip - mean) / std  # [T, C, H, W]
    clip = clip.unsqueeze(0)  # [1, T, C, H, W]
    return clip


@torch.no_grad()
def extract_video_features(model, video_path: str, output_npz_path: str,
                           num_frames: int, fps: int, device: torch.device):
    segment_size = 1.0  # 1-second segments (matches CaptainCook4D convention)

    if os.path.exists(output_npz_path):
        logging.info(f"Skipping existing: {output_npz_path}")
        return

    video = EncodedVideo.from_path(video_path)
    video_duration = float(video.duration)
    segment_end = max(video_duration - segment_size + 1, 1)
    segment_features = []

    for start_time in tqdm(np.arange(0, segment_end, segment_size),
                           desc=f"Segments of {os.path.basename(video_path)}"):
        end_time = min(start_time + segment_size, video_duration)
        if end_time - start_time < 0.04:
            continue
        try:
            clip_data = video.get_clip(start_sec=float(start_time), end_sec=float(end_time))
        except Exception as e:
            logging.warning(f"get_clip failed at {start_time}-{end_time}: {e}")
            continue
        clip = clip_data["video"]  # [C, T, H, W]
        if clip is None or clip.shape[1] == 0:
            continue
        video_tensor = preprocess_clip(clip, num_frames=num_frames).to(device)
        emb = model.compute_video(video_tensor)  # [1, 256]
        segment_features.append(emb.squeeze(0).cpu().numpy().astype(np.float32))

    if not segment_features:
        logging.warning(f"No features extracted for {video_path}")
        return

    features = np.stack(segment_features, axis=0)  # [N, 256]
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez(output_npz_path, features)
    logging.info(f"Saved {features.shape} -> {output_npz_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--egovlp_repo", type=str, default="/content/egovlp")
    parser.add_argument("--egovlp_ckpt", type=str, required=True,
                        help="Path to egovlp.pth checkpoint")
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Directory containing <recording_id>_360p.mp4 files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save <recording_id>_360p.mp4_1s_1s.npz files")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video_ext", type=str, default=".mp4")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model BEFORE chdir side-effect rebinds cwd
    cwd_before = os.getcwd()
    model = build_egovlp_model(args.egovlp_repo, args.egovlp_ckpt, args.num_frames, device)
    os.chdir(cwd_before)

    os.makedirs(args.output_dir, exist_ok=True)
    video_files = sorted(f for f in os.listdir(args.videos_dir) if f.endswith(args.video_ext))
    logging.info(f"Found {len(video_files)} videos in {args.videos_dir}")

    for video_name in tqdm(video_files, desc="Videos"):
        video_path = os.path.join(args.videos_dir, video_name)
        # Match CaptainCookStepDataset convention: <recording_id>_360p.mp4_1s_1s.npz
        out_name = f"{video_name}_1s_1s.npz"
        output_npz_path = os.path.join(args.output_dir, out_name)
        try:
            extract_video_features(model, video_path, output_npz_path,
                                   args.num_frames, args.fps, device)
        except Exception as e:
            logging.error(f"Failed on {video_name}: {e}")


if __name__ == "__main__":
    main()