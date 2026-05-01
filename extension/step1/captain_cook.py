import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

# Import from the actionformer_release submodule (relative path to the submodule)
from actionformer_release.libs.datasets.datasets import register_dataset
from actionformer_release.libs.datasets.data_utils import truncate_feats


@register_dataset("captaincook_dataset")
class CaptainCookDataset(Dataset):
    """
    Custom CaptainCook4D dataset for pre-extracted EgoVLP features.

    Expected feature structure on disk (flat, no subfolders):
        {feat_folder}/{recording_id}_360p.mp4_1s_1s.npz

    Concrete example:
        /content/drive/.../features_egovlp_num_frames_16/1_7_360p.mp4_1s_1s.npz

    Each .npz file contains an array of shape (T, 256):
        T = number of temporal segments (≈ video_duration_in_seconds)
        256 = EgoVLP embedding dimension

    Differences compared to the original captain_cook.py from CaptainCook4D:
        - No subfolder structure: files are flat in the feat_folder
        - Filename constructed as: {id}_360p.mp4_1s_1s.npz (not video_features.npy)
        - No forced override of file_ext based on the backbone
        - Loading .npz with automatic search for the correct array key
        - Removed unused imports (EncodedVideo, pandas, F)
    """

    def __init__(
            self,
            is_training,      # True during training, False during eval/inference
            split,            # list of subsets to load, e.g. ["training", "validation"]
            feat_folder,      # path to the folder with flat .npz files
            json_file,        # path to the annotations JSON in ActivityNet format
            feat_stride,      # temporal stride in frames (e.g. 30 -> 1s at 29.97fps)
            num_frames,       # frames per feature (e.g. 16)
            default_fps,      # default fps (29.97 for CaptainCook4D)
            downsample_rate,  # feature downsampling rate (1 = none)
            max_seq_len,      # max sequence length for the model (e.g. 4096)
            trunc_thresh,     # threshold to truncate segments at boundaries (e.g. 0.3)
            crop_ratio,       # random crop range in training, e.g. [0.9, 1.0]
            input_dim,        # embedding dimension (256 for EgoVLP)
            num_classes,      # total classes (353 unique steps in CaptainCook4D)
            file_prefix,      # file name prefix (None -> '')
            file_suffix,      # file name suffix (e.g. '_360p')
            file_ext,         # file extension (e.g. '.npz')
            force_upsampling, # force upsample to max_seq_len
            backbone,         # backbone name (e.g. 'egovlp')
            division_type,    # split type (e.g. 'recordings')
            videos_type,      # video type filter: 'all', 'normal', 'error'
    ):
        # ── Path validation ──────────────────────────────────────────────────
        assert os.path.exists(feat_folder), \
            f"feat_folder not found: {feat_folder}"
        assert os.path.exists(json_file), \
            f"json_file not found: {json_file}"
        assert isinstance(split, (tuple, list)), \
            f"split must be a list or tuple, received: {type(split)}"
        assert crop_ratio is None or len(crop_ratio) == 2

        # ── File attributes ────────────────────────────────────────────────────
        self.feat_folder  = feat_folder
        self.file_prefix  = file_prefix if file_prefix is not None else ''
        self.file_suffix  = file_suffix if file_suffix is not None else '_360p'
        self.file_ext     = file_ext if file_ext is not None else '.npz'
        self.json_file    = json_file

        # ── Mode and split ──────────────────────────────────────────────────
        self.split       = split
        self.is_training = is_training

        # ── Backbone and dataset metadata ───────────────────────────────────────
        self.backbone      = backbone
        self.division_type = division_type
        self.videos_type   = videos_type

        # ── Temporal parameters ───────────────────────────────────────────────
        self.feat_stride     = feat_stride      # stride in frames
        self.num_frames      = num_frames       # frames per feature (16 for EgoVLP 1s)
        self.input_dim       = input_dim        # 256
        self.default_fps     = default_fps      # 29.97
        self.downsample_rate = downsample_rate  # 1
        self.max_seq_len     = max_seq_len      # 4096
        self.trunc_thresh    = trunc_thresh     # 0.3
        self.num_classes     = num_classes      # 353
        self.crop_ratio      = crop_ratio       # [0.9, 1.0]
        self.label_dict      = None             # populated by _load_json_db

        # ── Load the database ────────────────────────────────────────────────
        dict_db, label_dict = self._load_json_db(self.json_file, videos_type)
        assert len(label_dict) <= num_classes, \
            f"label_dict ({len(label_dict)}) exceeds num_classes ({num_classes})"

        self.data_list  = dict_db
        self.label_dict = label_dict

        # ── Dataset attributes (used by train.py) ─────────────────────────
        empty_label_ids = self._find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name'   : 'captaincook_egovlp',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': empty_label_ids,
        }

        print(f"[CaptainCookDataset] Loaded {len(self.data_list)} videos "
              f"for split {split} | videos_type='{videos_type}'")

    # ──────────────────────────────────────────────────────────────────────────
    # Public methods required by train.py
    # ──────────────────────────────────────────────────────────────────────────

    def get_attributes(self):
        """Returns the dataset attributes (called by train.py)."""
        return self.db_attributes

    def __len__(self):
        return len(self.data_list)

    # ──────────────────────────────────────────────────────────────────────────
    # Feature path construction (customized for flat files)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_feat_path(self, recording_id: str) -> str:
        """
        Constructs the full path of the .npz file for a given recording_id.

        Format: {feat_folder}/{prefix}{recording_id}{suffix}.mp4_1s_1s{ext}

        Examples:
            recording_id = "1_7"
            prefix       = ''
            suffix       = '_360p'
            ext          = '.npz'
            -> .../features_egovlp_num_frames_16/1_7_360p.mp4_1s_1s.npz
        """
        filename = f"{self.file_prefix}{recording_id}{self.file_suffix}.mp4_1s_1s{self.file_ext}"
        return os.path.join(self.feat_folder, filename)

    # ──────────────────────────────────────────────────────────────────────────
    # Database loading from JSON
    # ──────────────────────────────────────────────────────────────────────────

    def _load_json_db(self, json_file: str, videos_type: str = 'all'):
        """
        Loads the annotations JSON and filters the videos based on:
          1. Subset (training / validation / test)
          2. Existence of the .npz file on disk
          3. Video type (all / normal / error)

        Returns:
            dict_db:    tuple of dictionaries, one for each valid video
            label_dict: map {step_name: step_id} built on all videos
        """
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        json_db = json_data['database']

        # ── First pass: construct label_dict from ALL videos ─────────────
        # (not just those of the current split, to have a complete map)
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # ── Determine normal/error filter ─────────────────────────────────────
        # CaptainCook4D recording_id convention:
        #   second field (after "_") in [1-25, 100-125]  -> NORMAL execution
        #   second field in [26-99, 126-200]            -> execution with ERROR
        if videos_type == 'all':
            skip_normal, skip_error = False, False
        elif videos_type == 'error':
            skip_normal, skip_error = True, False
        elif videos_type == 'normal':
            skip_normal, skip_error = False, True
        else:
            raise ValueError(
                f"invalid videos_type: '{videos_type}'. "
                "Use 'all', 'normal' or 'error'."
            )

        # ── Second pass: filter and build dict_db ───────────────────────
        dict_db = tuple()
        n_missing_feat = 0

        for key, value in json_db.items():

            # Filter 1: subset
            if value['subset'].lower() not in self.split:
                continue

            # Filter 2: feature file existence
            feat_path = self._build_feat_path(key)
            if not os.path.exists(feat_path):
                n_missing_feat += 1
                continue

            # Filter 3: video type (normal vs error)
            try:
                error_id = int(key.split("_")[1])
            except (IndexError, ValueError):
                # recording_id with unexpected format -> include just to be safe
                error_id = 0

            is_normal = (1 <= error_id <= 25) or (100 <= error_id <= 125)
            is_error  = (26 <= error_id <= 99) or (126 <= error_id <= 200)

            if skip_normal and is_normal:
                continue
            if skip_error and is_error:
                continue

            # ── Temporal metadata ────────────────────────────────────────────
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = float(value['fps'])
            else:
                raise ValueError(f"FPS not available for '{key}'")

            duration = float(value['duration']) if 'duration' in value else 1e8

            # ── Annotations (segments and labels) ────────────────────────────────
            if value.get('annotations'):
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])             # [t_start, t_end] in seconds
                    labels.append([label_dict[act['label']]])   # [[step_id]]

                segments = np.asarray(segments, dtype=np.float32)          # (N, 2)
                labels   = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)  # (N,)
            else:
                segments = None
                labels   = None

            dict_db += ({
                'id'      : key,
                'fps'     : fps,
                'duration': duration,
                'segments': segments,   # (N, 2) in seconds
                'labels'  : labels,     # (N,)
            },)

        if n_missing_feat > 0:
            print(f"[CaptainCookDataset] ⚠ {n_missing_feat} videos skipped: "
                  f".npz file not found in '{self.feat_folder}'")

        return dict_db, label_dict

    # ──────────────────────────────────────────────────────────────────────────
    # Empty classes (steps without samples in the loaded subset)
    # ──────────────────────────────────────────────────────────────────────────

    def _find_empty_cls(self, label_dict: dict, num_classes: int):
        """
        Returns the list of step_ids that do not appear in any video
        of the loaded subset. Used by ActionFormer to initialize the
        prior of the classification head.
        """
        if len(label_dict) == num_classes:
            return []
        present_ids = set(label_dict.values())
        return [i for i in range(num_classes) if i not in present_ids]

    # ──────────────────────────────────────────────────────────────────────────
    # __getitem__ — returns a single sample
    # ──────────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        # ── Load features from disk ───────────────────────────────────────────
        feat_path = self._build_feat_path(video_item['id'])

        # The .npz file can have the key 'arr_0' (numpy default)
        # or a key with the same name as the array.
        # Let's try the most common keys in order.
        data = np.load(feat_path, allow_pickle=False)
        keys = list(data.keys())

        if 'video_features' in keys:
            feats = data['video_features'].astype(np.float32)
        elif 'arr_0' in keys:
            feats = data['arr_0'].astype(np.float32)
        else:
            # Fallback: take the first available array
            feats = data[keys[0]].astype(np.float32)

        # feats shape: (T, 256) — T temporal segments, 256 EgoVLP dim

        # ── Temporal downsampling ────────────────────────────────────────────
        # downsample_rate=1 -> no effect; >1 -> subsample the sequence
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate

        # feat_offset: half a clip in feature units, used to center segments.
        # With num_frames=16 and feat_stride=30: offset = 0.5 * 16/30 ≈ 0.267 features
        # Each feature represents a window of num_frames frames: its temporal center
        # is half-window, not at the beginning.
        feat_offset = 0.5 * self.num_frames / feat_stride

        # (T, C) -> (C, T): ActionFormer expects channels-first
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # ── Convert segments: seconds -> feature indices ────────────────────
        # Formula: idx_feature = t_sec * fps / feat_stride - feat_offset
        # Example: t=10.0s, fps=29.97, feat_stride=30, feat_offset=0.267
        #          -> 10.0 * 29.97 / 30 - 0.267 = 9.73 feature indices
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments = None
            labels   = None

        # ── Output dict ───────────────────────────────────────────────────────
        data_dict = {
            'video_id'       : video_item['id'],
            'feats'          : feats,        # (C, T) = (256, T)
            'segments'       : segments,     # (N, 2) in feature units
            'labels'         : labels,       # (N,)
            'fps'            : video_item['fps'],
            'duration'       : video_item['duration'],
            'feat_stride'    : feat_stride,
            'feat_num_frames': self.num_frames,
        }

        # During training: truncate the sequence if it exceeds max_seq_len
        # (random crop in the range crop_ratio × max_seq_len)
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict,
                self.max_seq_len,
                self.trunc_thresh,
                feat_offset,
                self.crop_ratio,
            )

        return data_dict
