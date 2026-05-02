# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# actionformer core
from actionformer_release.libs.core import load_config
from actionformer_release.libs.modeling import make_meta_arch
from actionformer_release.libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

# This import triggers @register_dataset("captaincook_dataset") as a side
# effect, inserting CaptainCookDataset into the actionformer registry.
import captain_cook

from actionformer_release.libs.datasets import make_dataset, make_data_loader


################################################################################
def main(args):
    """Run ActionFormer inference and save detections to eval_results.pkl."""

    if not os.path.isfile(args.config):
        raise ValueError(f"Config file does not exist: {args.config}")
    cfg = load_config(args.config)
    assert len(cfg['val_split']) > 0, "val_split must be specified in the YAML."

    # Same pattern as train.py: CLI args take priority over YAML values.
    cfg['dataset']['backbone']      = args.backbone
    cfg['dataset']['division_type'] = args.division_type
    cfg['dataset']['num_frames']    = args.num_frames
    cfg['dataset']['feat_stride']   = args.stride
    cfg['dataset']['videos_type']   = args.videos_type
    if args.feat_folder != 'features':
        cfg['dataset']['feat_folder'] = args.feat_folder

    # Apply dataset defaults
    cfg['dataset'].setdefault('force_upsampling', False)
    cfg['dataset'].setdefault('downsample_rate', 1)
    cfg['dataset'].setdefault('file_prefix', '')
    cfg['dataset'].setdefault('file_suffix', '_360p')
    cfg['dataset'].setdefault('file_ext', '.npz')

    # Apply top-level cfg defaults (same as train.py)
    cfg.setdefault('init_rand_seed', 1234567891)
    cfg.setdefault('model_name', 'LocPointTransformer')
    
    if torch.cuda.is_available():
        cfg['devices'] = list(range(torch.cuda.device_count()))
    else:
        cfg['devices'] = [0]

    cfg['opt'].setdefault('warmup_epochs', 5)
    cfg['loader'].setdefault('num_workers', 2)

    # ext_score_file is optional; default to None if not present in YAML
    cfg['test_cfg'].setdefault('ext_score_file', None)

    # ── 5. Set model input_dim ────────────────────────────────────────────────
    # Original eval.py hardcodes input_dim per backbone but omits egovlp.
    # Strategy: known backbones keep their hardcoded value for reproducibility;
    # egovlp (and any unknown backbone) reads input_dim from cfg['dataset'],
    # which is 256 as set in the YAML.
    backbone_input_dims = {
        'omnivore': 1024,
        'videomae': 400,
        '3dresnet': 400,
        'slowfast' : 400,
        'x3d'     : 400,
        'egovlp'  : 256
    }
    if args.backbone in backbone_input_dims:
        cfg['model']['input_dim'] = backbone_input_dims[args.backbone]
    else:
        cfg['model']['input_dim'] = cfg['dataset']['input_dim']

    # Reconstruct checkpoint folder path
    backbone       = args.backbone
    division_type  = args.division_type
    output_folder_name = f"{backbone}_{division_type}"

    # For omnivore only: append segment size (e.g. "_3s") to match train.py
    if backbone == 'omnivore':
        seg_size = int(cfg['dataset']['num_frames'] / cfg['dataset']['default_fps'])
        output_folder_name += f"_{seg_size}s"
        feat_folder = cfg['dataset']['feat_folder']
        if 'sub' in feat_folder:
            sub_sample_size_str = os.path.basename(feat_folder).split('_')[-1]
            output_folder_name += f"_{sub_sample_size_str}"
    # For egovlp: output_folder_name = "egovlp_recordings" — no suffix needed.

    dataset_name = cfg['dataset_name']
    ckpt_base = os.path.join(
        cfg['output_folder'], dataset_name,
        output_folder_name + '_' + str(args.ckpt)
    )
    print(f"Checkpoint base path: {ckpt_base}")

    # Resolve the actual .pth.tar file
    if ".pth.tar" in ckpt_base:
        assert os.path.isfile(ckpt_base), f"Checkpoint file not found: {ckpt_base}"
        ckpt_file = ckpt_base
    else:
        assert os.path.isdir(ckpt_base), f"Checkpoint folder not found: {ckpt_base}"
        if args.epoch > 0:
            ckpt_file = os.path.join(ckpt_base, f'epoch_{args.epoch:03d}.pth.tar')
        else:
            ckpt_files = sorted(glob.glob(os.path.join(ckpt_base, '*.pth.tar')))
            assert ckpt_files, f"No .pth.tar files found in {ckpt_base}"
            ckpt_file = ckpt_files[-1]  # use latest epoch
        assert os.path.exists(ckpt_file), f"Checkpoint file not found: {ckpt_file}"

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    # val_split is used for inference; for step 1 we want ALL videos
    # (training + validation + test) to build step embeddings for the
    # full dataset. Override val_split to include all subsets if --all_splits.
    if args.all_splits:
        cfg['val_split'] = ['training', 'validation', 'test']

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print(f"Loading checkpoint: {ckpt_file}")
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # Always load EMA model weights (more stable than the regular model)
    print("Loading EMA model weights...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # etup evaluator or saveonly output
    det_eval, output_file = None, None

    if args.saveonly:
        # Save raw detections to .pkl for downstream
        # Output: {ckpt_dir}/eval_results.pkl
        output_file = os.path.join(os.path.dirname(ckpt_file), 'eval_results.pkl')
        print(f"--saveonly mode: detections will be saved to {output_file}")
    else:
        # Compute mAP metrics (useful for benchmarking/report)
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )

    """5. Test the model"""
    print(f"\nRunning inference with model '{cfg['model_name']}' ...")
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
    )
    end = time.time()
    print(f"All done! Total time: {end - start:.2f} sec")

    if output_file:
        print(f"\nDetections saved to: {output_file}")
        print("Next step: run build_step_embeddings.py to compute step-level embeddings.")


################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run ActionFormer inference on CaptainCook4D with EgoVLP features'
    )
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='run name used in train.py --output (e.g. "reproduce")')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch to load (-1 = latest, default: -1)')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output detections per video (default: -1 = no limit)')
    parser.add_argument('--saveonly', action='store_true',
                        help='save detections to .pkl without computing mAP '
                             '(always use this for step 1 pipeline)')
    parser.add_argument('--all_splits', action='store_true',
                        help='run inference on train+val+test (needed for extension step 1: '
                             'we need step embeddings for ALL videos)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')

    # Feature / dataset options (mirrors train.py CLI)
    parser.add_argument('--backbone', default='egovlp', type=str,
                        choices=['omnivore', '3dresnet', 'videomae', 'slowfast', 'x3d', 'egovlp'],
                        help='feature backbone (default: egovlp)')
    parser.add_argument('--division_type', default='recordings', type=str,
                        choices=['recordings', 'person', 'environment', 'recipes'],
                        help='data split type (default: recordings)')
    parser.add_argument('--feat_folder', default='features', type=str,
                        help='override feat_folder from YAML (default: use YAML value)')
    parser.add_argument('--num_frames', default=30, type=int,
                        help='frames per feature window (default: 30 = 1s at 29.97fps)')
    parser.add_argument('--stride', default=30, type=int,
                        help='stride in frames between consecutive features (default: 30)')
    parser.add_argument('--videos_type', default='all', type=str,
                        choices=['all', 'normal', 'error'],
                        help='which videos to process (default: all)')

    args = parser.parse_args()
    main(args)
