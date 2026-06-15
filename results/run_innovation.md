# Innovation (3-class) Training Log

Captured per-epoch training output (tqdm progress bars stripped).


## densenet121 — seed 0

| Field | Value |
|---|---|
| Task | 3-class classification (Normal / Bacteria / Virus) |
| Class weights | [1.18, 0.633, 1.187] |
| Selection | f1_macro |
| Resplit | train={'NORMAL': 1190, 'BACTERIA': 2219, 'VIRUS': 1183}, val={'NORMAL': 151, 'BACTERIA': 311, 'VIRUS': 162} |
| Source log | `innovation_run_2026-06-15_13-36-15.log` |

| Phase | Epoch | Train loss | Val loss | Val F1 |
|---|---|---|---|---|
| head_only | 1/5 | 1.0196 | 0.9534 | 0.5956 |
| head_only | 2/5 | 0.8683 | 0.8369 | 0.7091 |
| head_only | 3/5 | 0.7809 | 0.7850 | 0.6979 |
| head_only | 4/5 | 0.7270 | 0.7294 | 0.7178 |
| head_only | 5/5 | 0.6915 | 0.6942 | 0.7430 |
| fine_tune | 1/30 | 0.4931 | 0.4659 | 0.8048 |
| fine_tune | 2/30 | 0.3917 | 0.4333 | 0.7817 |
| fine_tune | 3/30 | 0.3505 | 0.4025 | 0.8397 |
| fine_tune | 4/30 | 0.3056 | 0.4258 | 0.8209 |
| fine_tune | 5/30 | 0.2745 | 0.3872 | 0.8368 |
| fine_tune | 6/30 | 0.2376 | 0.4895 | 0.8020 |
| fine_tune | 7/30 | 0.2053 | 0.5380 | 0.8299 |
| fine_tune | 8/30 | 0.1790 | 0.5257 | 0.8066 |

**Best epoch by val F1:** epoch 3 (val F1 = 0.8397)


## densenet121 — seed 1

| Field | Value |
|---|---|
| Task | 3-class classification (Normal / Bacteria / Virus) |
| Class weights | [1.185, 0.632, 1.183] |
| Selection | f1_macro |
| Resplit | train={'NORMAL': 1183, 'BACTERIA': 2216, 'VIRUS': 1185}, val={'NORMAL': 158, 'BACTERIA': 314, 'VIRUS': 160} |
| Source log | `innovation_run_2026-06-15_13-36-15.log` |

| Phase | Epoch | Train loss | Val loss | Val F1 |
|---|---|---|---|---|
| head_only | 1/5 | 1.0445 | 1.0089 | 0.4238 |
| head_only | 2/5 | 0.8872 | 0.8910 | 0.5914 |
| head_only | 3/5 | 0.7956 | 0.8014 | 0.6855 |
| head_only | 4/5 | 0.7377 | 0.7924 | 0.6635 |
| head_only | 5/5 | 0.6998 | 0.7363 | 0.6860 |
| fine_tune | 1/30 | 0.4856 | 0.4522 | 0.8053 |
| fine_tune | 2/30 | 0.3941 | 0.4291 | 0.8051 |
| fine_tune | 3/30 | 0.3419 | 0.4497 | 0.8099 |
| fine_tune | 4/30 | 0.3145 | 0.4827 | 0.8120 |
| fine_tune | 5/30 | 0.2825 | 0.4317 | 0.8088 |
| fine_tune | 6/30 | 0.2415 | 0.4684 | 0.8140 |
| fine_tune | 7/30 | 0.1741 | 0.4593 | 0.8213 |
| fine_tune | 8/30 | 0.1405 | 0.5565 | 0.8198 |
| fine_tune | 9/30 | 0.1324 | 0.5721 | 0.8233 |
| fine_tune | 10/30 | 0.0989 | 0.6807 | 0.7987 |
| fine_tune | 11/30 | 0.0703 | 0.6690 | 0.8274 |
| fine_tune | 12/30 | 0.0605 | 0.6818 | 0.8200 |
| fine_tune | 13/30 | 0.0485 | 0.7815 | 0.7988 |
| fine_tune | 14/30 | 0.0523 | 0.7144 | 0.8119 |
| fine_tune | 15/30 | 0.0427 | 0.7137 | 0.8020 |
| fine_tune | 16/30 | 0.0312 | 0.7082 | 0.8066 |

**Best epoch by val F1:** epoch 11 (val F1 = 0.8274)


## densenet121 — seed 2

| Field | Value |
|---|---|
| Task | 3-class |
| Class weights | [0.691, 1.309] |
| Selection | f1_macro |
| Resplit | train={'NORMAL': 1190, 'BACTERIA': 2219, 'VIRUS': 1183}, val={'NORMAL': 151, 'BACTERIA': 311, 'VIRUS': 162} |
| Source log | `innovation_run_2026-06-15_13-36-15.log` |

| Phase | Epoch | Train loss | Val loss | Val F1 |
|---|---|---|---|---|
| head_only | 1/5 | 1.0053 | 0.9030 | 0.6359 |
| head_only | 2/5 | 0.8657 | 0.8171 | 0.7220 |
| head_only | 3/5 | 0.7791 | 0.7565 | 0.7376 |
| head_only | 4/5 | 0.7227 | 0.7104 | 0.7338 |
| head_only | 5/5 | 0.6940 | 0.6982 | 0.7328 |
| fine_tune | 1/30 | 0.4867 | 0.4524 | 0.8076 |
| fine_tune | 2/30 | 0.4048 | 0.3839 | 0.8350 |
| fine_tune | 3/30 | 0.3457 | 0.3987 | 0.8293 |
| fine_tune | 4/30 | 0.3037 | 0.4012 | 0.8382 |
| fine_tune | 5/30 | 0.2698 | 0.4097 | 0.8191 |
| fine_tune | 6/30 | 0.2384 | 0.4148 | 0.8409 |
| fine_tune | 7/30 | 0.1765 | 0.4411 | 0.8278 |
| fine_tune | 8/30 | 0.1381 | 0.4514 | 0.8301 |
| fine_tune | 9/30 | 0.1107 | 0.5260 | 0.8328 |
| fine_tune | 10/30 | 0.1060 | 0.5478 | 0.8487 |
| fine_tune | 11/30 | 0.0762 | 0.5275 | 0.8401 |
| fine_tune | 12/30 | 0.0546 | 0.5562 | 0.8422 |
| fine_tune | 13/30 | 0.0509 | 0.5780 | 0.8488 |
| fine_tune | 14/30 | 0.0373 | 0.6445 | 0.8418 |
| fine_tune | 15/30 | 0.0320 | 0.6357 | 0.8461 |
| fine_tune | 16/30 | 0.0291 | 0.6081 | 0.8477 |
| fine_tune | 17/30 | 0.0282 | 0.6297 | 0.8368 |
| fine_tune | 18/30 | 0.0230 | 0.6478 | 0.8444 |
| head_only | 1/5 | 0.6564 | 0.6554 | 0.6050 |
| head_only | 2/5 | 0.6235 | 0.6330 | 0.6223 |
| head_only | 3/5 | 0.6054 | 0.6150 | 0.6556 |
| head_only | 4/5 | 0.5932 | 0.6138 | 0.6164 |
| head_only | 5/5 | 0.5905 | 0.5922 | 0.6950 |
| fine_tune | 1/30 | 0.5527 | 0.5475 | 0.7014 |
| fine_tune | 2/30 | 0.4917 | 0.5599 | 0.7454 |
| fine_tune | 3/30 | 0.4388 | 0.5783 | 0.7503 |
| fine_tune | 4/30 | 0.3939 | 0.5544 | 0.7290 |
| fine_tune | 5/30 | 0.3569 | 0.6425 | 0.7655 |
| fine_tune | 6/30 | 0.2783 | 0.6468 | 0.7515 |
| fine_tune | 7/30 | 0.2245 | 0.6066 | 0.7320 |
| fine_tune | 8/30 | 0.2034 | 0.6890 | 0.7403 |
| fine_tune | 9/30 | 0.1778 | 0.7426 | 0.7549 |
| fine_tune | 10/30 | 0.1364 | 0.7132 | 0.7439 |
| head_only | 1/5 | 0.6647 | 0.6506 | 0.6320 |
| head_only | 2/5 | 0.6251 | 0.6295 | 0.6622 |
| head_only | 3/5 | 0.6050 | 0.6008 | 0.6990 |
| head_only | 4/5 | 0.5886 | 0.5916 | 0.7047 |
| head_only | 5/5 | 0.5830 | 0.5847 | 0.7008 |
| fine_tune | 1/30 | 0.5482 | 0.5608 | 0.7169 |
| fine_tune | 2/30 | 0.4867 | 0.6379 | 0.6506 |
| fine_tune | 3/30 | 0.4414 | 0.6202 | 0.6660 |
| fine_tune | 4/30 | 0.3971 | 0.6627 | 0.7073 |
| fine_tune | 5/30 | 0.3418 | 0.7308 | 0.7234 |
| fine_tune | 6/30 | 0.2566 | 0.8091 | 0.7171 |
| fine_tune | 7/30 | 0.2092 | 0.8475 | 0.7261 |
| fine_tune | 8/30 | 0.1882 | 0.8012 | 0.6926 |
| fine_tune | 9/30 | 0.1577 | 0.8498 | 0.7075 |
| fine_tune | 10/30 | 0.1279 | 0.9151 | 0.7243 |
| fine_tune | 11/30 | 0.0929 | 0.9018 | 0.7147 |
| fine_tune | 12/30 | 0.0833 | 0.9428 | 0.7287 |
| fine_tune | 13/30 | 0.0732 | 1.0285 | 0.7126 |
| fine_tune | 14/30 | 0.0572 | 0.9835 | 0.7215 |
| fine_tune | 15/30 | 0.0504 | 0.9991 | 0.7360 |
| fine_tune | 16/30 | 0.0443 | 1.0658 | 0.7149 |
| fine_tune | 17/30 | 0.0387 | 1.0749 | 0.7073 |
| fine_tune | 18/30 | 0.0424 | 1.1447 | 0.7323 |
| fine_tune | 19/30 | 0.0360 | 1.0911 | 0.7204 |
| fine_tune | 20/30 | 0.0348 | 1.1550 | 0.7074 |
| head_only | 1/5 | 0.7217 | 0.6829 | 0.5643 |
| head_only | 2/5 | 0.6687 | 0.6433 | 0.6668 |
| head_only | 3/5 | 0.6306 | 0.6130 | 0.6861 |
| head_only | 4/5 | 0.6056 | 0.5953 | 0.7216 |
| head_only | 5/5 | 0.5915 | 0.5883 | 0.6866 |
| fine_tune | 1/30 | 0.5535 | 0.5161 | 0.7463 |
| fine_tune | 2/30 | 0.4952 | 0.5310 | 0.7565 |
| fine_tune | 3/30 | 0.4569 | 0.5466 | 0.7436 |
| fine_tune | 4/30 | 0.4191 | 0.5231 | 0.7730 |
| fine_tune | 5/30 | 0.3795 | 0.5742 | 0.7161 |
| fine_tune | 6/30 | 0.2919 | 0.5425 | 0.7531 |
| fine_tune | 7/30 | 0.2534 | 0.5643 | 0.7340 |
| fine_tune | 8/30 | 0.2109 | 0.5814 | 0.7362 |
| fine_tune | 9/30 | 0.1783 | 0.6432 | 0.7347 |

**Best epoch by val F1:** epoch 13 (val F1 = 0.8488)


## resnet18 — seed 0

| Field | Value |
|---|---|
| Task | 3-class classification (Normal / Bacteria / Virus) |
| Class weights | [1.18, 0.633, 1.187] |
| Selection | f1_macro |
| Resplit | train={'NORMAL': 1190, 'BACTERIA': 2219, 'VIRUS': 1183}, val={'NORMAL': 151, 'BACTERIA': 311, 'VIRUS': 162} |
| Source log | `innovation_run_2026-06-15_13-36-15.log` |

| Phase | Epoch | Train loss | Val loss | Val F1 |
|---|---|---|---|---|
| head_only | 1/5 | 1.0443 | 1.0454 | 0.4133 |
| head_only | 2/5 | 0.8599 | 0.9162 | 0.5296 |
| head_only | 3/5 | 0.7757 | 0.8707 | 0.5787 |
| head_only | 4/5 | 0.7147 | 0.8050 | 0.6374 |
| head_only | 5/5 | 0.6865 | 0.7642 | 0.6465 |
| fine_tune | 1/30 | 0.4871 | 0.4450 | 0.8043 |
| fine_tune | 2/30 | 0.4047 | 0.4316 | 0.8112 |
| fine_tune | 3/30 | 0.3609 | 0.4446 | 0.8035 |
| fine_tune | 4/30 | 0.3169 | 0.5341 | 0.7664 |
| fine_tune | 5/30 | 0.3078 | 0.4390 | 0.8273 |
| fine_tune | 6/30 | 0.2782 | 0.4251 | 0.8349 |
| fine_tune | 7/30 | 0.2480 | 0.4116 | 0.8336 |
| fine_tune | 8/30 | 0.2257 | 0.4940 | 0.8075 |
| fine_tune | 9/30 | 0.1918 | 0.4997 | 0.8210 |
| fine_tune | 10/30 | 0.1780 | 0.4966 | 0.8167 |
| fine_tune | 11/30 | 0.1585 | 0.5063 | 0.8185 |

**Best epoch by val F1:** epoch 6 (val F1 = 0.8349)


## resnet18 — seed 1

| Field | Value |
|---|---|
| Task | 3-class classification (Normal / Bacteria / Virus) |
| Class weights | [1.185, 0.632, 1.183] |
| Selection | f1_macro |
| Resplit | train={'NORMAL': 1183, 'BACTERIA': 2216, 'VIRUS': 1185}, val={'NORMAL': 158, 'BACTERIA': 314, 'VIRUS': 160} |
| Source log | `innovation_run_2026-06-15_13-36-15.log` |

| Phase | Epoch | Train loss | Val loss | Val F1 |
|---|---|---|---|---|
| head_only | 1/5 | 0.9926 | 0.8880 | 0.6061 |
| head_only | 2/5 | 0.8304 | 0.7956 | 0.6587 |
| head_only | 3/5 | 0.7625 | 0.7567 | 0.6750 |
| head_only | 4/5 | 0.7048 | 0.7568 | 0.6570 |
| head_only | 5/5 | 0.6761 | 0.7165 | 0.6943 |
| fine_tune | 1/30 | 0.4893 | 0.4432 | 0.8030 |
| fine_tune | 2/30 | 0.4139 | 0.4412 | 0.8100 |
| fine_tune | 3/30 | 0.3635 | 0.4377 | 0.8058 |
| fine_tune | 4/30 | 0.3164 | 0.5555 | 0.7569 |
| fine_tune | 5/30 | 0.2843 | 0.5254 | 0.7684 |
| fine_tune | 6/30 | 0.2512 | 0.5295 | 0.7908 |
| fine_tune | 7/30 | 0.2341 | 0.6195 | 0.7804 |

**Best epoch by val F1:** epoch 2 (val F1 = 0.8100)


## resnet18 — seed 2

| Field | Value |
|---|---|
| Task | 3-class classification (Normal / Bacteria / Virus) |
| Class weights | [1.196, 0.628, 1.176] |
| Selection | f1_macro |
| Resplit | train={'NORMAL': 1175, 'BACTERIA': 2237, 'VIRUS': 1195}, val={'NORMAL': 166, 'BACTERIA': 293, 'VIRUS': 150} |
| Source log | `innovation_run_2026-06-15_13-36-15.log` |

| Phase | Epoch | Train loss | Val loss | Val F1 |
|---|---|---|---|---|
| head_only | 1/5 | 0.9276 | 0.8948 | 0.5627 |
| head_only | 2/5 | 0.8043 | 0.8352 | 0.6074 |
| head_only | 3/5 | 0.7424 | 0.7841 | 0.6322 |
| head_only | 4/5 | 0.6920 | 0.7813 | 0.6329 |
| head_only | 5/5 | 0.6748 | 0.7623 | 0.6489 |
| fine_tune | 1/30 | 0.4938 | 0.4634 | 0.8093 |
| fine_tune | 2/30 | 0.4174 | 0.4453 | 0.7900 |
| fine_tune | 3/30 | 0.3662 | 0.4041 | 0.8215 |
| fine_tune | 4/30 | 0.3303 | 0.4168 | 0.8006 |
| fine_tune | 5/30 | 0.3063 | 0.3913 | 0.8189 |
| fine_tune | 6/30 | 0.2750 | 0.4403 | 0.8256 |
| fine_tune | 7/30 | 0.2422 | 0.4257 | 0.8271 |
| fine_tune | 8/30 | 0.2221 | 0.4238 | 0.8205 |
| fine_tune | 9/30 | 0.2034 | 0.4455 | 0.8210 |
| fine_tune | 10/30 | 0.1266 | 0.4177 | 0.8490 |
| fine_tune | 11/30 | 0.1028 | 0.4705 | 0.8519 |
| fine_tune | 12/30 | 0.0971 | 0.4607 | 0.8207 |
| fine_tune | 13/30 | 0.0753 | 0.5138 | 0.8366 |
| fine_tune | 14/30 | 0.0540 | 0.4748 | 0.8460 |
| fine_tune | 15/30 | 0.0445 | 0.4771 | 0.8544 |
| fine_tune | 16/30 | 0.0461 | 0.4799 | 0.8539 |
| fine_tune | 17/30 | 0.0354 | 0.5236 | 0.8445 |
| fine_tune | 18/30 | 0.0292 | 0.5195 | 0.8435 |
| fine_tune | 19/30 | 0.0278 | 0.5188 | 0.8416 |
| fine_tune | 20/30 | 0.0250 | 0.5236 | 0.8457 |

**Best epoch by val F1:** epoch 15 (val F1 = 0.8544)
