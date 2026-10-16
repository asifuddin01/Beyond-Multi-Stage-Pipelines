# Beyond Multi-Stage Pipelines

**A label-correlation ensemble for multi-label retinal fundus disease classification**

Md Asif Uddin, Jannatul Ferdous, Asif Islam, Zobaer Mahmud Zisan, Promit Ghosh,
Shah Ahafik Arman, Niloy Farhan, Rafeed Rahman — Department of Computer Science and
Engineering, BRAC University, Dhaka, Bangladesh.

This repository holds the code, the released test-set predictions and the paper
source for a controlled comparison of three architectures on one 39-target label
space built from RFMiD-1, RFMiD-2 and ODIR-5K:

| | Architecture | Composition |
|---|---|---|
| A | Three-stage cascade | binary screening gate → 7-way group router → six group-specific classifiers (8 models) |
| B | Flat classifier | one Swin-V2-B, linear head, 39 independent sigmoids |
| C | Label-correlation model | Swin-V2-B + 39 learned label queries + 2-layer transformer decoder |
| D | **Best configuration** | B and C, 4-view test-time augmentation, blended 0.45 / 0.55, per-class thresholds |

All models share the backbone (Swin-V2-B, ImageNet-22k → 1k, 384 px), the
preprocessing, the loss (BCE with per-class `pos_weight`) and the optimiser, so the
differences are attributable to the architecture.

## Main results (2,454-image test split, 39 targets)

| Configuration | Exact | Jaccard | micro-F1 | macro-F1 | macro-AUC | Detection |
|---|---|---|---|---|---|---|
| Flat baseline | 0.679 | 0.723 | 0.664 | 0.443 | 0.943 | 0.621 |
| Three-stage pipeline | 0.687 | 0.728 | 0.649 | 0.392 | 0.747\* | 0.594 |
| Label-correlation + TTA | 0.676 | 0.730 | 0.656 | 0.442 | 0.967 | **0.689** |
| **Flat + label-corr. ensemble + TTA** | **0.694** | **0.740** | **0.678** | **0.456** | **0.972** | 0.638 |

\* Depressed by the cascade's hard-gated zero probabilities, not by weak features.
The full nine-row ablation is in [`results/ablation/ablation_table.csv`](results/ablation/ablation_table.csv).

In the cascade, the two coarse gates lose 34.2 % of all true disease instances
(18.6 % at Stage 1, 15.6 % at Stage 2) while the fine-grained Stage-3 classifiers
lose only 6.7 %.

## Repository layout

```
src/fundus/          shared library
  labels.py          39-target label space, disease groups, pooled G_Rare
  config.py          paths (from $FUNDUS_DATA_ROOT) and per-model hyper-parameters
  data.py            retina-square crop, transforms, datasets, source-balanced negatives
  models.py          Swin-V2-B classifier, LabelCorrelationNet, GeM head
  losses.py          BCE / focal / asymmetric loss
  engine.py          resumable AMP training loop, evaluation
  metrics.py         the single scoring routine used for every ablation row
  inference.py       test-time augmentation, ensembling, three-stage cascade, loss cascade
notebooks/           run in order
  01_stage1_screening.ipynb
  02_stage2_group_router.ipynb
  03_stage3_group_classifiers.ipynb
  04_pipeline_test_evaluation.ipynb
  05_flat_baseline.ipynb
  06_label_correlation.ipynb
  07_tta_ensemble_ablation.ipynb
scripts/
  score_predictions.py   re-score the released predictions against the test labels
results/
  final_ensemble/    test-set predictions of the best configuration (binary + probabilities)
  ablation/          nine-configuration ablation table
  pipeline/          cascade summary, loss cascade, per-disease table, Stage-2/3 validation
  flat_baseline/     flat model per-disease table and summary
paper/               LaTeX source (IEEEtran), PDF and figure sources of the paper
scripts/make_paper_figures.py   regenerate the paper's data figures from results/
data/README.md       expected dataset layout
```

## Setup

```bash
git clone https://github.com/asifuddin01/Beyond-Multi-Stage-Pipelines.git
cd Beyond-Multi-Stage-Pipelines
pip install -r requirements.txt      # PyTorch: install the build for your CUDA first
pip install -e .
export FUNDUS_DATA_ROOT=/path/to/merged_dataset
export FUNDUS_RUN_ROOT=runs
```

Experiments were run on a single NVIDIA GPU with batch size 4 and gradient
accumulation (effective batch 8; 16 for the label-correlation model). One epoch of
the flat or label-correlation model over 8,520 images takes about 23 minutes.

## Data

Only public datasets are used: RFMiD 1.0, RFMiD 2.0 and ODIR-5K. They are not
redistributed here; download them from their official sources and merge them into
the layout described in [`data/README.md`](data/README.md). Each source keeps its
official train/validation/test split.

## Reproducing the paper

Run the notebooks in order. Notebooks 01–03 train the cascade, 04 evaluates it on
the test split, 05–06 train the flat and label-correlation models, and 07 builds
every row of the ablation, selects the blend weight on validation and exports the
final predictions.

To check the released predictions without training anything:

```bash
python scripts/score_predictions.py \
  --gt $FUNDUS_DATA_ROOT/test/merged_test.csv \
  --prob results/final_ensemble/test_prob_ensemble_flat_lc_w055_tta.csv \
  --per-disease results/final_ensemble/test_per_disease_ensemble_flat_lc.csv
# add --subset mured15 or --subset rfmid26 for the literature-comparison subsets
```

The cleaned code fixes two inconsistencies of the original research notebooks:
Stage-2 routing thresholds are now always read from the same run as the Stage-2
checkpoint, and each Stage-3 model is calibrated on the same checkpoint (best
validation macro-AP) that the cascade loads. A retrained cascade may therefore
differ slightly from the numbers reported in the paper, which come from the
original runs.

## Citation

```bibtex
@misc{uddin2026beyond,
  title  = {Beyond Multi-Stage Pipelines: A Label-Correlation Ensemble for
            Multi-Label Retinal Fundus Disease Classification},
  author = {Uddin, Md Asif and Ferdous, Jannatul and Islam, Asif and Zisan, Zobaer Mahmud
            and Ghosh, Promit and Arman, Shah Ahafik and Farhan, Niloy and Rahman, Rafeed},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The datasets
remain under the terms of their original providers. This software is a research
prototype and is not a medical device.

Contact: Md Asif Uddin, md.asif.uddin@g.bracu.ac.bd
