# Dataset layout

The study merges three public fundus datasets and keeps each one's official split:

| Source | Train | Val | Test | Native labels |
|---|---|---|---|---|
| ODIR-5K (per-eye images) | 6,092 | 861 | 1,642 | 8 categories + diagnostic keywords |
| RFMiD 1.0 | 1,920 | 640 | 640 | 45 conditions |
| RFMiD 2.0 | 508 | 177 | 172 | 50 conditions |
| **Merged** | **8,520** | **1,678** | **2,454** | → 39 scored targets |

```
$FUNDUS_DATA_ROOT/
    train/merged_train.csv   train/images/
    val/merged_val.csv       val/images/
    test/merged_test.csv     test/images/
```

## CSV format

One row per image:

| column | meaning |
|---|---|
| `image` | identifier **without** file extension; must match an image file stem in `images/` |
| `Normal` | 1 = healthy, 0 = any pathology |
| one column per disease code | 0/1, e.g. `DR`, `ARMD`, `MH`, `TSLN`, ... and the rare codes `VH`, `PLQ`, ... |

Group columns, the pooled `G_Rare` column, the `unhealthy` flag and the source are
derived in code (`fundus.data.load_split`). Missing disease columns are treated as 0.

## Harmonisation rules used in the paper

1. **ODIR-5K** is converted from patient-level records to per-eye images; each eye's
   labels come from that eye's diagnostic keywords.
2. **Healthy flags** are unified into `Normal` (RFMiD 1.0's `Disease_Risk` is inverted;
   RFMiD 2.0 uses `WNL`; ODIR-5K uses `N`).
3. **ODIR "Other"** is removed: it is a catch-all, not a definable pathology.
4. **Namespacing**: identifiers are prefixed by source (`rfmid_1_…`, `rfmid_2_…`;
   ODIR keeps `<id>_left` / `<id>_right`) to prevent collisions; `fundus.labels.source_of`
   relies on these prefixes.
5. **Rare pooling**: the 21 classes with five or fewer training images are pooled into
   `G_Rare` (see `fundus.labels.RARE`).
6. The merged test split is sealed: it is read only by notebooks 04, 05 (final
   evaluation) and 07.
