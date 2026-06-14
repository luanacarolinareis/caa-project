```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_baseline.py --config configs/default.yaml --seed 0
Training custom_cnn on cuda with seed 0
baseline epoch 001/20 train_loss=0.4614 val_loss=0.6678 val_f1=0.6957                                                                                    
baseline epoch 002/20 train_loss=0.3398 val_loss=1.1549 val_f1=0.6667                                                                                    
baseline epoch 003/20 train_loss=0.3054 val_loss=1.0313 val_f1=0.7273                                                                                    
baseline epoch 004/20 train_loss=0.2878 val_loss=0.5781 val_f1=0.8421                                                                                    
baseline epoch 005/20 train_loss=0.2788 val_loss=0.7634 val_f1=0.7273                                                                                    
baseline epoch 006/20 train_loss=0.2716 val_loss=0.7481 val_f1=0.7273                                                                                    
baseline epoch 007/20 train_loss=0.2540 val_loss=0.8811 val_f1=0.7273                                                                                    
baseline epoch 008/20 train_loss=0.2492 val_loss=1.8089 val_f1=0.6667                                                                                    
baseline epoch 009/20 train_loss=0.2359 val_loss=1.1438 val_f1=0.6957                                                                                    
Early stopping at epoch 9; best epoch was 4.
Saved checkpoint: results\checkpoints\custom_cnn_seed0.pt
Saved history: results\metrics\custom_cnn_seed0_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_baseline.py --config configs/default.yaml --seed 1
Training custom_cnn on cuda with seed 1
baseline epoch 001/20 train_loss=0.5107 val_loss=0.7687 val_f1=0.6667                                                                                    
baseline epoch 002/20 train_loss=0.3858 val_loss=0.8266 val_f1=0.6667                                                                                    
baseline epoch 003/20 train_loss=0.3304 val_loss=0.9342 val_f1=0.6667                                                                                    
baseline epoch 004/20 train_loss=0.3016 val_loss=0.8822 val_f1=0.6957                                                                                    
baseline epoch 005/20 train_loss=0.2892 val_loss=0.7773 val_f1=0.7273                                                                                    
baseline epoch 006/20 train_loss=0.2759 val_loss=0.8068 val_f1=0.6957                                                                                    
Early stopping at epoch 6; best epoch was 1.
Saved checkpoint: results\checkpoints\custom_cnn_seed1.pt
Saved history: results\metrics\custom_cnn_seed1_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_baseline.py --config configs/default.yaml --seed 2
Training custom_cnn on cuda with seed 2
baseline epoch 001/20 train_loss=0.4547 val_loss=0.9658 val_f1=0.6667                                                                                    
baseline epoch 002/20 train_loss=0.3369 val_loss=0.9188 val_f1=0.6667                                                                                    
baseline epoch 003/20 train_loss=0.3032 val_loss=0.7285 val_f1=0.6957                                                                                    
baseline epoch 004/20 train_loss=0.2826 val_loss=0.7574 val_f1=0.7273                                                                                    
baseline epoch 005/20 train_loss=0.2693 val_loss=1.2576 val_f1=0.7273                                                                                    
baseline epoch 006/20 train_loss=0.2657 val_loss=0.7996 val_f1=0.6957                                                                                    
baseline epoch 007/20 train_loss=0.2589 val_loss=0.6519 val_f1=0.7273                                                                                    
baseline epoch 008/20 train_loss=0.2408 val_loss=1.2973 val_f1=0.6667                                                                                    
baseline epoch 009/20 train_loss=0.2359 val_loss=1.2746 val_f1=0.6957                                                                                    
baseline epoch 010/20 train_loss=0.2307 val_loss=1.3810 val_f1=0.6667                                                                                    
baseline epoch 011/20 train_loss=0.2291 val_loss=1.0904 val_f1=0.6667                                                                                    
baseline epoch 012/20 train_loss=0.2143 val_loss=0.9641 val_f1=0.7273                                                                                    
Early stopping at epoch 12; best epoch was 7.
Saved checkpoint: results\checkpoints\custom_cnn_seed2.pt
Saved history: results\metrics\custom_cnn_seed2_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 0
Model       : resnet18
Config      : configs\default.yaml
Seed        : 0
Pretrained  : True
Head epochs : 5
Max epochs  : 30
Checkpoints : results\checkpoints
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\Carolina/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 44.7M/44.7M [00:06<00:00, 6.76MB/s]

Model instantiated: resnet18
Device      : cuda
Phase 1 -> head only, trainable params: 513
head_only epoch 001/5 train_loss=0.5360 val_loss=0.7039 val_f1=0.6667                                                                                    
head_only epoch 002/5 train_loss=0.4257 val_loss=0.5503 val_f1=0.7273                                                                                    
head_only epoch 003/5 train_loss=0.3589 val_loss=0.5016 val_f1=0.7619                                                                                    
head_only epoch 004/5 train_loss=0.3123 val_loss=0.4309 val_f1=0.8000                                                                                    
head_only epoch 005/5 train_loss=0.2807 val_loss=0.3941 val_f1=0.8000                                                                                    
Phase 2 -> full fine-tune, trainable params: 11,177,025
fine_tune epoch 001/30 train_loss=0.0928 val_loss=0.7375 val_f1=0.7273                                                                                   
fine_tune epoch 002/30 train_loss=0.0619 val_loss=0.6710 val_f1=0.7619                                                                                   
fine_tune epoch 003/30 train_loss=0.0411 val_loss=0.2218 val_f1=0.8889                                                                                   
fine_tune epoch 004/30 train_loss=0.0358 val_loss=0.5273 val_f1=0.8000                                                                                   
fine_tune epoch 005/30 train_loss=0.0321 val_loss=0.3507 val_f1=0.8421                                                                                   
fine_tune epoch 006/30 train_loss=0.0269 val_loss=0.4520 val_f1=0.8000                                                                                   
fine_tune epoch 007/30 train_loss=0.0255 val_loss=0.6863 val_f1=0.7273                                                                                   
fine_tune epoch 008/30 train_loss=0.0126 val_loss=0.1278 val_f1=0.9412                                                                                   
fine_tune epoch 009/30 train_loss=0.0101 val_loss=0.1453 val_f1=0.9412                                                                                   
fine_tune epoch 010/30 train_loss=0.0073 val_loss=0.3858 val_f1=0.8421                                                                                   
fine_tune epoch 011/30 train_loss=0.0110 val_loss=0.2362 val_f1=0.8889                                                                                   
fine_tune epoch 012/30 train_loss=0.0041 val_loss=0.1565 val_f1=0.9412                                                                                   
fine_tune epoch 013/30 train_loss=0.0041 val_loss=0.3391 val_f1=0.8889                                                                                   
Early stopping at epoch 13; best epoch was 8.

Saved checkpoint: results\checkpoints\resnet18_seed0.pt
Saved histories: results\metrics\resnet18_seed0_head_only_history.json, results\metrics\resnet18_seed0_fine_tune_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 1
Model       : resnet18
Config      : configs\default.yaml
Seed        : 1
Pretrained  : True
Head epochs : 5
Max epochs  : 30
Checkpoints : results\checkpoints

Model instantiated: resnet18
Device      : cuda
Phase 1 -> head only, trainable params: 513
head_only epoch 001/5 train_loss=0.4977 val_loss=0.5992 val_f1=0.6667                                                                                    
head_only epoch 002/5 train_loss=0.3982 val_loss=0.4953 val_f1=0.7778                                                                                    
head_only epoch 003/5 train_loss=0.3401 val_loss=0.4463 val_f1=0.8750                                                                                    
head_only epoch 004/5 train_loss=0.3038 val_loss=0.4093 val_f1=0.9333                                                                                    
head_only epoch 005/5 train_loss=0.2793 val_loss=0.3846 val_f1=0.9333                                                                                    
Phase 2 -> full fine-tune, trainable params: 11,177,025
fine_tune epoch 001/30 train_loss=0.0939 val_loss=0.4732 val_f1=0.7619                                                                                   
fine_tune epoch 002/30 train_loss=0.0610 val_loss=0.9139 val_f1=0.7619                                                                                   
fine_tune epoch 003/30 train_loss=0.0393 val_loss=0.6564 val_f1=0.7619                                                                                   
fine_tune epoch 004/30 train_loss=0.0334 val_loss=0.3292 val_f1=0.8889                                                                                   
fine_tune epoch 005/30 train_loss=0.0276 val_loss=0.1117 val_f1=0.9412                                                                                   
fine_tune epoch 006/30 train_loss=0.0288 val_loss=0.2230 val_f1=0.8421                                                                                   
fine_tune epoch 007/30 train_loss=0.0254 val_loss=0.9779 val_f1=0.7273                                                                                   
fine_tune epoch 008/30 train_loss=0.0203 val_loss=0.3606 val_f1=0.8889                                                                                   
fine_tune epoch 009/30 train_loss=0.0185 val_loss=0.8414 val_f1=0.8000                                                                                   
fine_tune epoch 010/30 train_loss=0.0072 val_loss=0.1693 val_f1=0.9412                                                                                   
Early stopping at epoch 10; best epoch was 5.

Saved checkpoint: results\checkpoints\resnet18_seed1.pt
Saved histories: results\metrics\resnet18_seed1_head_only_history.json, results\metrics\resnet18_seed1_fine_tune_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 2
Model       : resnet18
Config      : configs\default.yaml
Seed        : 2
Pretrained  : True
Head epochs : 5
Max epochs  : 30
Checkpoints : results\checkpoints

Model instantiated: resnet18
Device      : cuda
Phase 1 -> head only, trainable params: 513
head_only epoch 001/5 train_loss=0.5732 val_loss=0.7602 val_f1=0.6667                                                                                    
head_only epoch 002/5 train_loss=0.4484 val_loss=0.5893 val_f1=0.6364                                                                                    
head_only epoch 003/5 train_loss=0.3737 val_loss=0.5127 val_f1=0.7000                                                                                    
head_only epoch 004/5 train_loss=0.3331 val_loss=0.4637 val_f1=0.7778                                                                                    
head_only epoch 005/5 train_loss=0.2990 val_loss=0.4151 val_f1=0.7778                                                                                    
Phase 2 -> full fine-tune, trainable params: 11,177,025
fine_tune epoch 001/30 train_loss=0.0959 val_loss=0.1836 val_f1=0.9412                                                                                   
fine_tune epoch 002/30 train_loss=0.0570 val_loss=0.3296 val_f1=0.8889                                                                                   
fine_tune epoch 003/30 train_loss=0.0407 val_loss=0.0716 val_f1=1.0000                                                                                   
fine_tune epoch 004/30 train_loss=0.0373 val_loss=0.6362 val_f1=0.7273                                                                                   
fine_tune epoch 005/30 train_loss=0.0239 val_loss=0.3754 val_f1=0.8421                                                                                   
fine_tune epoch 006/30 train_loss=0.0292 val_loss=0.0334 val_f1=1.0000                                                                                   
fine_tune epoch 007/30 train_loss=0.0196 val_loss=0.1710 val_f1=0.8889                                                                                   
fine_tune epoch 008/30 train_loss=0.0244 val_loss=0.5063 val_f1=0.7619                                                                                   
fine_tune epoch 009/30 train_loss=0.0167 val_loss=0.2171 val_f1=0.9412                                                                                   
fine_tune epoch 010/30 train_loss=0.0164 val_loss=0.1456 val_f1=1.0000                                                                                   
fine_tune epoch 011/30 train_loss=0.0068 val_loss=0.1856 val_f1=0.9412                                                                                   
Early stopping at epoch 11; best epoch was 6.

Saved checkpoint: results\checkpoints\resnet18_seed2.pt
Saved histories: results\metrics\resnet18_seed2_head_only_history.json, results\metrics\resnet18_seed2_fine_tune_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0
Model       : densenet121
Config      : configs\default.yaml
Seed        : 0
Pretrained  : True
Head epochs : 5
Max epochs  : 30
Checkpoints : results\checkpoints

Model instantiated: densenet121
Device      : cuda
Phase 1 -> head only, trainable params: 1,025
head_only epoch 001/5 train_loss=0.5471 val_loss=0.6587 val_f1=0.6667                                                                                    
head_only epoch 002/5 train_loss=0.4610 val_loss=0.5712 val_f1=0.6957                                                                                    
head_only epoch 003/5 train_loss=0.3988 val_loss=0.5192 val_f1=0.7619                                                                                    
head_only epoch 004/5 train_loss=0.3542 val_loss=0.4760 val_f1=0.8000                                                                                    
head_only epoch 005/5 train_loss=0.3193 val_loss=0.4257 val_f1=0.8421                                                                                    
Phase 2 -> full fine-tune, trainable params: 6,954,881
fine_tune epoch 001/30 train_loss=0.0993 val_loss=0.1354 val_f1=0.9412                                                                                   
fine_tune epoch 002/30 train_loss=0.0580 val_loss=0.2017 val_f1=0.8889                                                                                   
fine_tune epoch 003/30 train_loss=0.0335 val_loss=0.2798 val_f1=0.8421                                                                                   
fine_tune epoch 004/30 train_loss=0.0343 val_loss=0.3645 val_f1=0.8421                                                                                   
fine_tune epoch 005/30 train_loss=0.0261 val_loss=0.1166 val_f1=0.9412                                                                                   
fine_tune epoch 006/30 train_loss=0.0185 val_loss=0.0888 val_f1=1.0000                                                                                   
fine_tune epoch 007/30 train_loss=0.0161 val_loss=0.3575 val_f1=0.8889                                                                                   
fine_tune epoch 008/30 train_loss=0.0209 val_loss=0.4860 val_f1=0.8421                                                                                   
fine_tune epoch 009/30 train_loss=0.0089 val_loss=0.1332 val_f1=0.9412                                                                                   
fine_tune epoch 010/30 train_loss=0.0134 val_loss=0.0475 val_f1=1.0000                                                                                   
fine_tune epoch 011/30 train_loss=0.0141 val_loss=0.2422 val_f1=0.8889                                                                                   
fine_tune epoch 012/30 train_loss=0.0117 val_loss=0.0355 val_f1=1.0000                                                                                   
fine_tune epoch 013/30 train_loss=0.0059 val_loss=0.2760 val_f1=0.8889                                                                                   
fine_tune epoch 014/30 train_loss=0.0089 val_loss=0.2261 val_f1=0.8889                                                                                   
fine_tune epoch 015/30 train_loss=0.0087 val_loss=0.0722 val_f1=0.9412                                                                                   
fine_tune epoch 016/30 train_loss=0.0077 val_loss=0.1266 val_f1=0.9412                                                                                   
fine_tune epoch 017/30 train_loss=0.0038 val_loss=0.0229 val_f1=1.0000                                                                                   
fine_tune epoch 018/30 train_loss=0.0017 val_loss=0.0155 val_f1=1.0000                                                                                   
fine_tune epoch 019/30 train_loss=0.0045 val_loss=0.0390 val_f1=1.0000                                                                                   
fine_tune epoch 020/30 train_loss=0.0017 val_loss=0.1070 val_f1=0.9412                                                                                   
fine_tune epoch 021/30 train_loss=0.0007 val_loss=0.0888 val_f1=0.9412                                                                                   
fine_tune epoch 022/30 train_loss=0.0017 val_loss=0.0451 val_f1=1.0000                                                                                   
fine_tune epoch 023/30 train_loss=0.0007 val_loss=0.0706 val_f1=1.0000                                                                                   
Early stopping at epoch 23; best epoch was 18.

Saved checkpoint: results\checkpoints\densenet121_seed0.pt
Saved histories: results\metrics\densenet121_seed0_head_only_history.json, results\metrics\densenet121_seed0_fine_tune_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 1
Model       : densenet121
Config      : configs\default.yaml
Seed        : 1
Pretrained  : True
Head epochs : 5
Max epochs  : 30
Checkpoints : results\checkpoints

Model instantiated: densenet121
Device      : cuda
Phase 1 -> head only, trainable params: 1,025
head_only epoch 001/5 train_loss=0.5371 val_loss=0.7204 val_f1=0.6667                                                                                    
head_only epoch 002/5 train_loss=0.4491 val_loss=0.6227 val_f1=0.6667                                                                                    
head_only epoch 003/5 train_loss=0.3905 val_loss=0.5686 val_f1=0.6957                                                                                    
head_only epoch 004/5 train_loss=0.3465 val_loss=0.5044 val_f1=0.7619                                                                                    
head_only epoch 005/5 train_loss=0.3122 val_loss=0.4821 val_f1=0.7619                                                                                    
Phase 2 -> full fine-tune, trainable params: 6,954,881
fine_tune epoch 001/30 train_loss=0.0891 val_loss=0.3041 val_f1=0.8421                                                                                   
fine_tune epoch 002/30 train_loss=0.0520 val_loss=0.2380 val_f1=0.8889                                                                                   
fine_tune epoch 003/30 train_loss=0.0458 val_loss=0.0590 val_f1=1.0000                                                                                   
fine_tune epoch 004/30 train_loss=0.0265 val_loss=0.3082 val_f1=0.8421                                                                                   
fine_tune epoch 005/30 train_loss=0.0247 val_loss=0.3090 val_f1=0.8421                                                                                   
fine_tune epoch 006/30 train_loss=0.0298 val_loss=0.2352 val_f1=0.8889                                                                                   
fine_tune epoch 007/30 train_loss=0.0212 val_loss=0.0329 val_f1=1.0000                                                                                   
fine_tune epoch 008/30 train_loss=0.0177 val_loss=0.1248 val_f1=1.0000                                                                                   
fine_tune epoch 009/30 train_loss=0.0115 val_loss=0.1810 val_f1=0.8889                                                                                   
fine_tune epoch 010/30 train_loss=0.0149 val_loss=0.0956 val_f1=0.9412                                                                                   
fine_tune epoch 011/30 train_loss=0.0085 val_loss=0.0631 val_f1=1.0000                                                                                   
fine_tune epoch 012/30 train_loss=0.0123 val_loss=0.0607 val_f1=1.0000                                                                                   
Early stopping at epoch 12; best epoch was 7.

Saved checkpoint: results\checkpoints\densenet121_seed1.pt
Saved histories: results\metrics\densenet121_seed1_head_only_history.json, results\metrics\densenet121_seed1_fine_tune_history.json
```

```bash
(.venv) PS C:\Users\Carolina\OneDrive - 4sb643\Ambiente de Trabalho\UA\CAA\Project> python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 2
Model       : densenet121
Config      : configs\default.yaml
Seed        : 2
Pretrained  : True
Head epochs : 5
Max epochs  : 30
Checkpoints : results\checkpoints

Model instantiated: densenet121
Device      : cuda
Phase 1 -> head only, trainable params: 1,025
head_only epoch 001/5 train_loss=0.5438 val_loss=0.7552 val_f1=0.6667                                                                                    
head_only epoch 002/5 train_loss=0.4510 val_loss=0.6604 val_f1=0.6667                                                                                    
head_only epoch 003/5 train_loss=0.3865 val_loss=0.5956 val_f1=0.6957                                                                                    
head_only epoch 004/5 train_loss=0.3413 val_loss=0.5445 val_f1=0.6957                                                                                    
head_only epoch 005/5 train_loss=0.3094 val_loss=0.4924 val_f1=0.7619                                                                                    
Phase 2 -> full fine-tune, trainable params: 6,954,881
fine_tune epoch 001/30 train_loss=0.1004 val_loss=0.5084 val_f1=0.8000                                                                                   
fine_tune epoch 002/30 train_loss=0.0515 val_loss=0.2707 val_f1=0.8421                                                                                   
fine_tune epoch 003/30 train_loss=0.0433 val_loss=0.0902 val_f1=0.9412                                                                                   
fine_tune epoch 004/30 train_loss=0.0266 val_loss=0.0589 val_f1=1.0000                                                                                   
fine_tune epoch 005/30 train_loss=0.0230 val_loss=0.0526 val_f1=1.0000                                                                                   
fine_tune epoch 006/30 train_loss=0.0196 val_loss=0.0465 val_f1=1.0000                                                                                   
fine_tune epoch 007/30 train_loss=0.0207 val_loss=0.0518 val_f1=1.0000                                                                                   
fine_tune epoch 008/30 train_loss=0.0123 val_loss=0.2477 val_f1=0.8889                                                                                   
fine_tune epoch 009/30 train_loss=0.0193 val_loss=0.0916 val_f1=1.0000                                                                                   
fine_tune epoch 010/30 train_loss=0.0131 val_loss=0.1260 val_f1=0.9412                                                                                   
fine_tune epoch 011/30 train_loss=0.0074 val_loss=0.1892 val_f1=0.9412                                                                                   
Early stopping at epoch 11; best epoch was 6.

Saved checkpoint: results\checkpoints\densenet121_seed2.pt
Saved histories: results\metrics\densenet121_seed2_head_only_history.json, results\metrics\densenet121_seed2_fine_tune_history.json
```