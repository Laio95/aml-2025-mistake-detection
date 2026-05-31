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

All paths to data and resources in the notebooks refer to the mounted Drive structure.

## Acknowledgements

This project builds on repositories from the CaptainCook4D release.

- **Error Recognition**: https://github.com/CaptainCook4D/error_recognition
- **Features Extraction**: https://github.com/CaptainCook4D/feature_extractors
- **ActionFormer**: https://github.com/happyharrycn/actionformer_release
- **EgoVLP**: https://github.com/showlab/EgoVLP
