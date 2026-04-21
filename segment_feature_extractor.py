import argparse
import datetime
import glob
import os
import sys
import numpy as np
import torch
# torchvision >=0.16 removed functional_tensor; patch for pytorchvideo compatibility
import torchvision.transforms.functional as _F_compat
sys.modules.setdefault("torchvision.transforms.functional_tensor", _F_compat)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import torchvision.transforms as T
import concurrent.futures
import logging
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from natsort import natsorted

try:
    # decord is optional: faster video decoding via NVDEC (GPU) or its CPU fallback.
    # get_batch() returns decord NDArray; .asnumpy() or torch conversion handled per call.
    from decord import VideoReader, cpu, gpu as decord_gpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="omnivore",
                        help="Backbone: omnivore | slowfast | x3d | 3dresnet | egovlp")
    parser.add_argument("--egovlp_repo", type=str, default="/content/egovlp",
                        help="Path to cloned EgoVLP repo (required for backbone=egovlp)")
    parser.add_argument("--egovlp_ckpt", type=str, default=None,
                        help="Path to egovlp.pth checkpoint (required for backbone=egovlp)")
    parser.add_argument("--use_decord", action="store_true", default=False,
                        help="Use decord for faster video decoding (requires pip install decord)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Clips per GPU forward pass when --use_decord is set (default: 8)")
    parser.add_argument("--videos_dir", type=str,
                        default="/data/rohith/captain_cook/data/gopro/resolution_360p",
                        help="Directory containing input .mp4 files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory where .npz feature files will be saved")
    return parser.parse_args()


def _build_egovlp_model(egovlp_repo: str, ckpt_path: str, device: torch.device):
    """Load FrozenInTime (EgoVLP) with manual checkpoint fixing.

    torch.hub is not used because EgoVLP requires sys.path injection and
    os.chdir so FrozenInTime can find its local ViT-B/16 weights at
    pretrained/jx_vit_base_p16_224-80ecf9dd.pth.
    """
    if egovlp_repo not in sys.path:
        sys.path.insert(0, egovlp_repo)

    cwd = os.getcwd()
    os.chdir(egovlp_repo)
    from model.model import FrozenInTime                 
    from utils.util import state_dict_data_parallel_fix  

    model = FrozenInTime(
        video_params={
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": 16,
            "pretrained": True,
            "time_init": "zeros",
        },
        text_params={
            "model": "distilbert-base-uncased",
            "pretrained": True,
            "input": "text",
        },
        projection_dim=256,
        load_checkpoint=None,
    )
    os.chdir(cwd)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
    new_state_dict = model._inflate_positional_embeds(new_state_dict)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    logging.info(f"EgoVLP loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")

    return model.eval().to(device)


def _infer_batch(clips, feature_extractor, method, device):
    """Run a single batched forward pass on a list of transformed clip tensors.

    Each clip is already in [C, T, H, W] format (output of ApplyTransformToKey).
    Returns a list of [1, D] numpy arrays, one per input clip.

    slowfast and omnivore have multi-input formats that require per-clip inference;
    they fall back to sequential calls rather than true batching.
    """
    with torch.no_grad():
        if method == "egovlp":
            # [B, C, T, H, W] → [B, T, C, H, W] as expected by compute_video
            batch = torch.stack(clips).permute(0, 2, 1, 3, 4).to(device)
            out = feature_extractor.compute_video(batch)  # [B, D]
        elif method in ("x3d", "3dresnet"):
            batch = torch.stack(clips).to(device)  # [B, C, T, H, W]
            out = feature_extractor(batch)          # [B, D]
        else:
            # slowfast (two-pathway list) and omnivore (multi-crop list) are not batchable here
            results = []
            for clip in clips:
                if method == "slowfast":
                    inp = [p.to(device)[None, ...] for p in clip]
                else:  # omnivore
                    inp = clip[0][None, ...].to(device)
                results.append(feature_extractor(inp).cpu().numpy())
            return results
    return [out[i : i + 1].cpu().numpy() for i in range(len(clips))]


# Video Processing
class VideoProcessor:
    def __init__(self, method, feature_extractor, video_transform, use_decord=False, batch_size=8):
        self.method = method
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform
        self.use_decord = use_decord
        self.batch_size = batch_size

        self.fps = 30
        self.num_frames_per_feature = 30

    def process_video(self, video_name, video_directory_path, output_features_path):
        segment_size = self.fps / self.num_frames_per_feature
        video_path = os.path.join(video_directory_path, f"{video_name}.mp4" if "mp4" not in video_name else video_name)

        output_file_path = os.path.join(output_features_path, video_name.replace('_224', ''))

        if os.path.exists(f"{output_file_path}_{int(segment_size)}s_{int(1)}s.npz"):
            logger.info(f"Skipping video: {video_name}")
            return

        os.makedirs(output_features_path, exist_ok=True)

        stride = 1
        video_features = []

        if self.use_decord:
            # GPU decode (NVDEC): frames land in VRAM directly, no CPU→GPU copy.
            # Falls back to CPU decode if NVDEC is unavailable (e.g. no CUDA driver).
            try:
                vr = VideoReader(video_path, ctx=decord_gpu(0))
            except Exception:
                vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            total_frames = len(vr)
            video_duration = total_frames / video_fps

            logger.info(f"video: {video_name} video_duration: {video_duration} s")
            segment_end = max(video_duration - segment_size + 1, 1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch_clips = []
            for start_time in tqdm(np.arange(0, segment_end, segment_size),
                                   desc=f"Processing video segments for video {video_name}"):
                end_time = min(start_time + segment_size, video_duration)
                if end_time - start_time < 0.04:
                    continue

                start_frame = int(start_time * video_fps)
                end_frame = min(int(end_time * video_fps), total_frames - 1)
                if end_frame <= start_frame:
                    continue

                # get_batch returns decord NDArray [T, H, W, C] uint8; convert to torch [C, T, H, W]
                frames = vr.get_batch(list(range(start_frame, end_frame)))
                segment_video_inputs = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2)

                # apply transforms (CPU-bound); result is [C, T, H, W]
                transformed = self.video_transform({"video": segment_video_inputs, "audio": None})
                batch_clips.append(transformed["video"])

                if len(batch_clips) == self.batch_size:
                    video_features.extend(_infer_batch(batch_clips, self.feature_extractor, self.method, device))
                    batch_clips = []

            # flush remaining clips that didn't fill a full batch
            if batch_clips:
                video_features.extend(_infer_batch(batch_clips, self.feature_extractor, self.method, device))
        else:    # old method not optimized 
            video = EncodedVideo.from_path(video_path)
            video_duration = video.duration

            logger.info(f"video: {video_name} video_duration: {video_duration} s")
            segment_end = max(video_duration - segment_size + 1, 1)

            for start_time in tqdm(np.arange(0, segment_end, segment_size),
                                   desc=f"Processing video segments for video {video_name}"):
                end_time = min(start_time + segment_size, video_duration)
                if end_time - start_time < 0.04:
                    continue

                video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
                segment_video_inputs = video_data["video"]

                segment_features = extract_features(
                    video_data_raw=segment_video_inputs,
                    feature_extractor=self.feature_extractor,
                    transforms_to_apply=self.video_transform,
                    method=self.method
                )
                video_features.append(segment_features)

        video_features = np.vstack(video_features)
        np.savez(f"{output_file_path}_{int(segment_size)}s_{int(stride)}s.npz", video_features)
        logger.info(f"Finished extraction and saving video: {video_name} video_features: {video_features.shape}")


# Feature Extraction
def extract_features(video_data_raw, feature_extractor, transforms_to_apply, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_data_for_transform = {"video": video_data_raw, "audio": None}
    video_data = transforms_to_apply(video_data_for_transform)
    video_inputs = video_data["video"]
    if method in ["omnivore"]:
        video_input = video_inputs[0][None, ...].to(device)
    elif method == "slowfast":
        video_input = [i.to(device)[None, ...] for i in video_inputs]
    elif method == "x3d":
        video_input = video_inputs.unsqueeze(0).to(device)
    elif method == "3dresnet":
        video_input = video_inputs.unsqueeze(0).to(device)
    elif method == "egovlp":
        # ApplyTransformToKey output: [C, T, H, W] → [1, T, C, H, W]
        video_input = video_inputs.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    with torch.no_grad():
        if method == "egovlp":
            features = feature_extractor.compute_video(video_input)  # [1, 256]
        else:
            features = feature_extractor(video_input)
    return features.cpu().numpy()


# Model Initialization
def get_video_transformation(name):
    if name == "omnivore":
        from omnivore_transforms import SpatialCrop, TemporalCrop  # noqa: E402
        num_frames = 32
        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                T.Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                NormalizeVideo(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                TemporalCrop(frames_per_clip=32, stride=40),
                SpatialCrop(crop_size=224, num_crops=3),
            ]
        )
    elif name == "slowfast":
        slowfast_alpha = 4
        num_frames = 32
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        class PackPathway(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                # Perform temporal sampling from the fast pathway.
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
                    ).long(),
                )
                frame_list = [slow_pathway, fast_pathway]
                return frame_list

        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size),
                PackPathway(),
            ]
        )
    elif name == "x3d":
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        model_transform_params = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            },
        }
        # Taking x3d_m as the model
        transform_params = model_transform_params["x3d_m"]
        video_transform = Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(
                        transform_params["crop_size"],
                        transform_params["crop_size"],
                    )
                ),
            ]
        )
    elif name == "3dresnet":
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        )
    elif name == "egovlp":
        side_size = 224
        crop_size = 224
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        num_frames = 16
        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(side_size),
                CenterCropVideo(crop_size),  
            ]
        )
    return ApplyTransformToKey(key="video", transform=video_transform)


def get_feature_extractor(name, device="cuda", egovlp_repo=None, egovlp_ckpt=None):
    if name == "egovlp":
        if not egovlp_repo or not egovlp_ckpt:
            raise ValueError("egovlp_repo and egovlp_ckpt are required for backbone=egovlp")
        return _build_egovlp_model(egovlp_repo, egovlp_ckpt, torch.device(device))
    if name == "omnivore":
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
        model.heads = torch.nn.Identity()
    elif name == "slowfast":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "x3d":
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "3dresnet":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        model.heads = torch.nn.Identity()

    feature_extractor = model
    feature_extractor = feature_extractor.to(device)
    feature_extractor = feature_extractor.eval()
    return feature_extractor


#def main_hololens(is_sequential=False):
    # hololens_directory_path = "/data/rohith/captain_cook/data/hololens/"
    # output_features_path = f"/data/rohith/captain_cook/features/hololens/segments/{method}/"

    # video_transform = get_video_transformation(method)
    # feature_extractor = get_feature_extractor(method)

    # processor = VideoProcessor(method, feature_extractor, video_transform)

    # if not is_sequential:
    #     num_threads = 10
    #     with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
    #         for recording_id in os.listdir(hololens_directory_path):
    #             video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
    #             executor.submit(processor.process_video, recording_id, video_file_path, output_features_path)
    # else:
    #     for recording_id in os.listdir(hololens_directory_path):
    #         video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
    #         processor.process_video(recording_id, video_file_path, output_features_path)


# Main
def main():
    video_files_path = args.videos_dir
    output_features_path = args.output_dir or f"/data/rohith/captain_cook/features/gopro/segments/{method}/"

    video_transform = get_video_transformation(method)
    feature_extractor = get_feature_extractor(
        method,
        egovlp_repo=args.egovlp_repo,
        egovlp_ckpt=args.egovlp_ckpt,
    )

    if args.use_decord and not _DECORD_AVAILABLE:
        raise RuntimeError("--use_decord requires decord: pip install decord")

    processor = VideoProcessor(method, feature_extractor, video_transform, use_decord=args.use_decord, batch_size=args.batch_size)

    mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]

    num_threads = 1
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    lambda file: processor.process_video(file, video_files_path, output_features_path), mp4_files
                ), total=len(mp4_files)
            )
        )


if __name__ == "__main__":
    args = parse_arguments()
    method = args.backbone

    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, f"{method}.log")
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    logger = logging.getLogger(__name__)

    # main_hololens(is_sequential=False)
    main()
