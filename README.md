# AML 2025/2026 - Mistake Detection Project

## Team

| Name | Matricola |
|------|-----------|
| Dario Lupo | s336550 |
| Alberto Giunti | s336374 |
| Marco Laiolo | s283816 |
| Angelo Rosario Modica | s344983 |

## Notebooks

The experiments are organized as numbered notebooks inside the `notebooks/` directory. Each notebook corresponds to a specific step of the pipeline and should be run in order within each group:

- **A\*** — Error recognition track (baseline reproduction, LSTM, EgoVLP extraction and training, backbone comparison)
- **B\*** — Step localization and graph-based fusion track

> **Note**: Some notebooks rely on code from git submodules (e.g. feature extractors, ActionFormer). Make sure to initialize all submodules before running them:
> ```
> git submodule update --init --recursive
> ```

## Environment Setup

The notebooks are designed to run on **Google Colab**. A personal Google Drive must be mounted as storage for datasets, features, and checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')
```

All paths to data and resources in the notebooks reference a `DRIVE_ROOT` variable (default: `/content/drive/MyDrive/AML_Project`). Update this variable at the top of each notebook to match your own Drive structure.

### Required data

- **Pre-extracted features**: the CaptainCook4D pre-extracted features (1s segments), expected at `{DRIVE_ROOT}/CaptainCook4D/features`.
- **Videos**: the CaptainCook4D GoPro resized dataset (`captain_cook_4d_gopro_resized_extracted`), expected at `{DRIVE_ROOT}/CaptainCook4D/captain_cook_4d_gopro_resized_extracted`.
- **EgoVLP checkpoint**: the pretrained weights `EgoVLP_PT_BEST.pth` (available from the [EgoVLP release page](https://github.com/showlab/EgoVLP)), expected at `{DRIVE_ROOT}/models/EgoVLP_PT_BEST.pth`.

## Acknowledgements

This project builds on repositories from the CaptainCook4D release.

- **Error Recognition**: https://github.com/CaptainCook4D/error_recognition
- **Features Extraction**: https://github.com/CaptainCook4D/feature_extractors
- **Multi-Step Localization** : https://github.com/CaptainCook4D/multi_step_localization
- **ActionFormer**: https://github.com/happyharrycn/actionformer_release
- **EgoVLP**: https://github.com/showlab/EgoVLP

