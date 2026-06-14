```bash
PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python innovation/evaluate_three_class.py --model resnet18 --checkpoint results/checkpoints/resnet18_3class_seed0.pt --seed 0
[data] Validation split is missing classes ['VIRUS']. Re-sampling a balanced validation set from training data.

Saved metrics: results\metrics\resnet18_3class_seed0.json

==================================================
  3-CLASS EVALUATION — resnet18
==================================================
  Accuracy       : 0.7692
  F1 (macro)     : 0.7527
  Precision (macro): 0.7893
  Recall (macro) : 0.7672
  AUROC (OvR)    : 0.9396
  Inference      : 12.840 ms/image

  Per-class F1:
    NORMAL      : 0.7263
    BACTERIA    : 0.8593
    VIRUS       : 0.6726

  Confusion Matrix (rows=actual, cols=predicted):
                NORMAL    BACTERIA       VIRUS
      NORMAL         134          32          68
    BACTERIA           1         232           9
       VIRUS           0          34         114
```

```bash
PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python innovation/evaluate_three_class.py --model resnet18 --checkpoint results/checkpoints/resnet18_3class_seed1.pt --seed 1
[data] Validation split is missing classes ['VIRUS']. Re-sampling a balanced validation set from training data.

Saved metrics: results\metrics\resnet18_3class_seed1.json

==================================================
  3-CLASS EVALUATION — resnet18
==================================================
  Accuracy       : 0.7644
  F1 (macro)     : 0.7561
  Precision (macro): 0.8117
  Recall (macro) : 0.7813
  AUROC (OvR)    : 0.9143
  Inference      : 12.757 ms/image

  Per-class F1:
    NORMAL      : 0.7143
    BACTERIA    : 0.9002
    VIRUS       : 0.6538

  Confusion Matrix (rows=actual, cols=predicted):
                NORMAL    BACTERIA       VIRUS
      NORMAL         130           4         100
    BACTERIA           0         212          30
       VIRUS           0          13         135
```

```bash
PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python innovation/evaluate_three_class.py --model resnet18 --checkpoint results/checkpoints/resnet18_3class_seed2.pt --seed 2
[data] Validation split is missing classes ['VIRUS']. Re-sampling a balanced validation set from training data.

Saved metrics: results\metrics\resnet18_3class_seed2.json

==================================================
  3-CLASS EVALUATION — resnet18
==================================================
  Accuracy       : 0.7740
  F1 (macro)     : 0.7566
  Precision (macro): 0.7974
  Recall (macro) : 0.7797
  AUROC (OvR)    : 0.9450
  Inference      : 12.335 ms/image

  Per-class F1:
    NORMAL      : 0.6961
    BACTERIA    : 0.8996
    VIRUS       : 0.6739

  Confusion Matrix (rows=actual, cols=predicted):
                NORMAL    BACTERIA       VIRUS
      NORMAL         126          19          89
    BACTERIA           2         233           7
       VIRUS           0          24         124
```

```bash
PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python innovation/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed0.pt --seed 0
[data] Validation split is missing classes ['VIRUS']. Re-sampling a balanced validation set from training data.

Saved metrics: results\metrics\densenet121_3class_seed0.json

==================================================
  3-CLASS EVALUATION — densenet121
==================================================
  Accuracy       : 0.8189
  F1 (macro)     : 0.8073
  Precision (macro): 0.8237
  Recall (macro) : 0.8207
  AUROC (OvR)    : 0.9517
  Inference      : 15.562 ms/image

  Per-class F1:
    NORMAL      : 0.8040
    BACTERIA    : 0.9076
    VIRUS       : 0.7102

  Confusion Matrix (rows=actual, cols=predicted):
                NORMAL    BACTERIA       VIRUS
      NORMAL         160           8          66
    BACTERIA           3         226          13
       VIRUS           1          22         125
```

```bash
PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python innovation/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed1.pt --seed 1
[data] Validation split is missing classes ['VIRUS']. Re-sampling a balanced validation set from training data.

Saved metrics: results\metrics\densenet121_3class_seed1.json

==================================================
  3-CLASS EVALUATION — densenet121
==================================================
  Accuracy       : 0.8397
  F1 (macro)     : 0.8295
  Precision (macro): 0.8458
  Recall (macro) : 0.8457
  AUROC (OvR)    : 0.9632
  Inference      : 15.161 ms/image

  Per-class F1:
    NORMAL      : 0.8101
    BACTERIA    : 0.9185
    VIRUS       : 0.7600

  Confusion Matrix (rows=actual, cols=predicted):
                NORMAL    BACTERIA       VIRUS
      NORMAL         160          15          59
    BACTERIA           1         231          10
       VIRUS           0          15         133
```

```bash
PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python innovation/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed2.pt --seed 2
[data] Validation split is missing classes ['VIRUS']. Re-sampling a balanced validation set from training data.

Saved metrics: results\metrics\densenet121_3class_seed2.json

==================================================
  3-CLASS EVALUATION — densenet121
==================================================
  Accuracy       : 0.7821
  F1 (macro)     : 0.7674
  Precision (macro): 0.8006
  Recall (macro) : 0.7852
  AUROC (OvR)    : 0.9526
  Inference      : 14.524 ms/image

  Per-class F1:
    NORMAL      : 0.7243
    BACTERIA    : 0.8705
    VIRUS       : 0.7072

  Confusion Matrix (rows=actual, cols=predicted):
                NORMAL    BACTERIA       VIRUS
      NORMAL         134          33          67
    BACTERIA           2         232           8
       VIRUS           0          26         122
```