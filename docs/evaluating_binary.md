```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model custom_cnn --checkpoint results/checkpoints/custom_cnn_seed0.pt --seed 0
Saved metrics: results\metrics\custom_cnn_seed0.json
accuracy=0.8109 precision=0.8542 recall=0.8410 f1=0.8475 auroc=0.8892395353933815
inference_time_ms_per_image=18.561
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model custom_cnn --checkpoint results/checkpoints/custom_cnn_seed1.pt --seed 1
Saved metrics: results\metrics\custom_cnn_seed1.json
accuracy=0.6522 precision=0.6439 recall=0.9923 f1=0.7810 auroc=0.8689129958360727
inference_time_ms_per_image=12.259
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model custom_cnn --checkpoint results/checkpoints/custom_cnn_seed2.pt --seed 2
Saved metrics: results\metrics\custom_cnn_seed2.json
accuracy=0.8061 precision=0.8064 recall=0.9077 f1=0.8540 auroc=0.8996712689020383
inference_time_ms_per_image=13.817
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed0.pt --seed 0
Saved metrics: results\metrics\resnet18_seed0.json
accuracy=0.8301 precision=0.7886 recall=0.9949 f1=0.8798 auroc=0.9544926583388122
inference_time_ms_per_image=13.025
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed1.pt --seed 1
Saved metrics: results\metrics\resnet18_seed1.json
accuracy=0.8109 precision=0.7709 recall=0.9923 f1=0.8677 auroc=0.9456169186938417
inference_time_ms_per_image=12.910
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed2.pt --seed 2
Saved metrics: results\metrics\resnet18_seed2.json
accuracy=0.8478 precision=0.8118 recall=0.9846 f1=0.8899 auroc=0.9473044049967126
inference_time_ms_per_image=14.364
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt --seed 0
Saved metrics: results\metrics\densenet121_seed0.json
accuracy=0.8462 precision=0.8050 recall=0.9949 f1=0.8899 auroc=0.9653298268682884
inference_time_ms_per_image=13.507
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed1.pt --seed 1
Saved metrics: results\metrics\densenet121_seed1.json
accuracy=0.8574 precision=0.8155 recall=0.9974 f1=0.8973 auroc=0.9734440061363139
inference_time_ms_per_image=13.031
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed2.pt --seed 2
Saved metrics: results\metrics\densenet121_seed2.json
accuracy=0.8189 precision=0.7764 recall=0.9974 f1=0.8732 auroc=0.9605413105413105
inference_time_ms_per_image=14.727
```