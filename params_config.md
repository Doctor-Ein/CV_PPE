## G1 参数组配置结论

基于现阶段实验结果，参数优先级可以排序为：

1. `weight_decay`
2. `lr0`
3. `lrf`
4. `optimizer`
5. `cos_lr`
6. `warmup_epochs`

当前阶段推荐结论如下：

- `optimizer` 推荐使用：`SGD`
- `cos_lr` 推荐保持：`False`
- `warmup_epochs` 推荐保持：`3.0`
- `lr0` 最优单点：`0.012`
- `weight_decay` 推荐区间：`0.00010 ~ 0.00022`
- `lrf` 可接受区间：`0.02 ~ 0.05`

若以 `mAP50` 为主，推荐优先考虑：

- `lr0=0.012`
- `weight_decay=0.00010`
- `lrf=0.01` 或在后续组合实验中再与 `0.02~0.05` 联合验证

若以 `mAP50-95` 为主，推荐优先考虑：

- `lr0=0.012`
- `weight_decay=0.00022`
- `lrf=0.01` 或在后续组合实验中再联合验证

这说明当前任务的优化重点应放在学习率和正则强度的窄区间精细搜索，而不是大范围更换优化策略。

---

## G2 参数组配置结论
即我们第二阶段得到的结果，分析程序输出为：

Experiment2: /home/aoya/workspace/CV_PPE/experiments/experiment2
Runs found : 12
Metric     : map50
hsv_s: recommend 1.0 (map50=0.55093)  dir=/home/aoya/workspace/CV_PPE/experiments/experiment2/hsv_scan_hsv_s_1
hsv_v: recommend 0.2 (map50=0.54437)  dir=/home/aoya/workspace/CV_PPE/experiments/experiment2/hsv_scan_hsv_v_0p2
bgr: recommend 0.0 (map50=0.57157)  dir=/home/aoya/workspace/CV_PPE/experiments/experiment2/hsv_scan_bgr_0p3
Wrote: /home/aoya/workspace/CV_PPE/experiments/experiment2/step2_recommendations.json
Wrote: /home/aoya/workspace/CV_PPE/experiments/experiment2/step2_recommendations.csv

(.venv) aoya@DESKTOP-UN42JV2:~/workspace/CV_PPE$ python step2_analyze.py  --metric map50_95
Experiment2: /home/aoya/workspace/CV_PPE/experiments/experiment2
Runs found : 12
Metric     : map50_95
hsv_s: recommend 0.35 (map50_95=0.27711)  dir=/home/aoya/workspace/CV_PPE/experiments/experiment2/hsv_scan_hsv_s_0p35-2
hsv_v: recommend 0.4 (map50_95=0.27434)  dir=/home/aoya/workspace/CV_PPE/experiments/experiment2/hsv_scan_hsv_v_0p4
bgr: recommend 0.0 (map50_95=0.27644)  dir=/home/aoya/workspace/CV_PPE/experiments/experiment2/hsv_scan_bgr_0p3
Wrote: /home/aoya/workspace/CV_PPE/experiments/experiment2/step2_recommendations.json
Wrote: /home/aoya/workspace/CV_PPE/experiments/experiment2/step2_recommendations.csv

## G3 参数组配置结论

参数 测试范围 最优值 best epoch mAP50 mAP50-95 相对基线提升
degrees 0–180 0.0 50 0.57464 0.27413 +0.00000
translate 0–1.0 0.05 58 0.56865 0.27992 +0.00579
scale 0–1.0 0.95 81 0.57829 0.28665 +0.01252
shear 0–180 2.0 49 0.58667 0.27713 +0.00300
perspective 0.0–0.001 0.0001 69 0.56199 0.27798 +0.00385
flipud 0–1.0 0.25 66 0.58743 0.28429 +0.01016
fliplr 0–1.0 0.75 66 0.58881 0.28008 +0.00595

## G4 参数组配置结论

| 参数 | 已验证/观察范围 | 推荐范围 | 说明 |
| --- | --- | --- | --- |
| mosaic | 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.75, 1.0 | 0.10-0.35 | 默认 1.0 偏强，后续组合中 0.10 表现最好 |
| close_mosaic | 0, 10, 20 | 10 | 单独调整无明显变化，保留默认 |
| mixup | 0.0, 0.05, 0.10, 0.15, 0.20 | 0.10-0.15 | 弱 mixup 有帮助，过强会下降 |
| cutmix | 0.0, 0.05, 0.10, 0.20 | 0.00-0.05 | 低强度可改善稳定性，但均值收益不稳定 |
| copy_paste | 0.0, 0.10, 0.20 | 0.00 | 当前实验未观察到收益 |
| erasing | 0.0, 0.20, 0.40 | 0.40 | 当前实验未观察到收益 |

## G5 参数组配置结论

baseline:
{"box": 7.5, "cls": 0.5, "dfl": 1.5, "label_smoothing"=0.0, "dropout": 0.0, "name": "baseline"}

联合调参数据组合：

{"box": 10.0, "cls": 1.25, "dfl": 1.5,  "name": "joint_large_box_std"}*map最优但是bias,variance比baseline要高不少
{"box": 10.0, "cls": 1.0,  "dfl": 0.5,  "name": "joint_high_precision_95"}
{"box": 5.0,  "cls": 1.25, "dfl": 1.5,  "name": "joint_small_box_cls_boost"}
{"box": 3.75, "cls": 1.0,  "dfl": 0.5,  "name": "joint_ultra_low_bias"}*综合map与bias,variance的最优 ✅ 选这个配置
{"box": 6.25, "cls": 1.0,  "dfl": 1.0,  "name": "joint_conservative_left"}