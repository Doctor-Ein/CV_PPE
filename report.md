# YOLO模型微调和性能验证

廖绪丞
学号： 524031910559
2026 年 6 月 9 日

```
本项目以Construction-PPE作为数据集，执行目标检测任务，模型选用 YOLOv8n。
本小组5人 先按物理对应上参数分组分工完成单变量与组内联合调参。而后各自以不同的方案完成最终组合，以及从各自的角度完成分析和解释。
主要以验证集mAP50_95为评判模型的指标。最终，选定的联合最优配置下，mAP50_95参考指标相比于基线模型，绝对提升+ 0.02248，相对提升 8.3 %.
并且通过参数分组，分别观察其对微调过程的影响，该过程可得出良好的可解释性：几何变形帮助处理施工场景中的视角扰动，颜色增强对光照条件的处理，框回归权重则增强了对于小目标定位的稳定性等。
```

## 1 整体执行说明

本次作业围绕“YOLOv3-v8模型的微调与性能验证”在 **Construction-PPE** 上进行。主要工
作包括：

1. 选模与数据准备：使用官方提供的yolov8n.pt作为基模型；按Ultralytics的检测数据集
    格式读取Construction-PPE，并用官方划分train/val/test = 1132/143/141进行实
    验。
2. 基线训练：在默认超参数下先训练一版基线模型，作为后续所有微调的统一比较对象。
3. 参数分组与协作：按任务分为 5 组参数（优化与收敛、色彩、几何、复合增强、损失/正
    则）完成单变量调参并记录各组最优点。
4. 联合调参：固定实验口径（epoch、imgsz、batch、seed等）下，对候选组合做分层组合试
    验，基于验证集mAP50-95选择联合最优。
5. 最终对比：在测试集上比较raw、baseline、单组（第三组中间模型）与joint最优模型的
    结果，并结合新增的 6 张新图做误检/漏检和场景适配分析。

## 2 评估指标与数据集配置

在数据选择方面，我们组选择了Construction-PPE数据集。我们认为能检测工地上工人是否佩
戴了安全帽，穿着工作服具有实际意义，可以保障工地安全。同时该数据集合涉及多个目标的


#### 检测，受不同物理参数的影响不同，相较单一检测目标更有训练难度，训练的效果更有对比性。

#### 编号与类编对应如下:

- 0: helmet
- 1: gloves
- 2: vest
- 3: boots
- 4: goggles
- 5: none
- 6: Person
- 7: no_helmet
- 8: no_goggle
- 9: no_gloves
- 10: no_boots 
在数据分割方面，我们采用了官方Construction-PPE分割：train=1132,
val=143, test=141。调参和训练只用construction-ppe-trainval.yaml，测试集不参与训
练，只用于最终汇总对比。

### 2.1 评估指标定义

每一个实验记录我们都会计算如下量，其中我们通过 **mAP50-95** 的大小最终评判模型的效果：

- **Precision** :预测框中真实阳性比例（误检越少越好）。
- **Recall** :所有真实目标中被检出的比例（漏检越少越好）。
- **mAP50** : IoU=0.5的平均精度。
- **mAP50-95** :在IoU= 0.50 : 0. 95 上取步长0.05的平均精度，作为主指标。
- **Best epoch** :取验证集mAP50:95(B)最大的epoch。
- **test** 对比指标:仅用于最终定稿，不用于超参数选择。

每组实验使用统一口径，按如下固定设置：

- 模型：yolov8n.pt
- 训练轮数： 100 个epoch
- 图像尺寸： 640
- Batch： 16
- 设备： 0 （按训练记录口径）
- 随机种子： 0 ，deterministic=True

剩余的参数我们分工先实现单变量调参获得最优值。


### 2.2 基线

#### 基线为默认参数训练（不含单参数改动）在验证集上的结果：

```
参数 Precision Recall mAP50 mAP50-
Baseline 0.69176 0.54924 0.57464 0.
```
```
表1:基线验证结果。
```
## 3 五组分工与参数划分逻辑

#### 我们按照参数的物理意义将需要找到最优值的参数划分为五组，按“环境条件敏感性”与“训

#### 练条件敏感性”建模：

- 第一组（lr0,lrf,optimizer,cos_lr,weight_decay,warmup_epochs）：优化器动力学与
    收敛行为，直接影响训练稳定性与最终泛化。
- 第二组（hsv_h,hsv_s,hsv_v,bgr）：光照、色彩与材质反射变化，反映施工现场的复杂
    照明。
- 第三组（ **degrees,translate,scale,shear,perspective,fliplr,flipud** ）：几何变化与
    视角，覆盖拍摄姿态、裁剪与尺度偏差。
- 第四组（mosaic,close_mosaic,mixup,cutmix,copy_paste,erasing）：样本组成与场景
    混合策略，模拟人群密度、遮挡与局部扰动。
- 第五组（box,cls,dfl,label_smoothing,dropout）：损失加权与模型正则，影响小目标
    定位与类别边界。

该划分的核心目标是：先测试“数据分布扰动”（色彩/几何/合成增强）与“优化器/损失”两类
因素是否同向改善，再做受控组合。

## 4 单变量调参（第三组）

这个部分我采用控制变量法：每轮只改变某一参数，其余保持与baseline一致。控制范围如下。

```
参数 测试范围 最优值 best epoch mAP50 mAP50-95 相对基线提升
degrees 0–180 0.0 50 0.57464 0.27413 +0.
translate 0–1.0 0.05 58 0.56865 0.27992 +0.
scale 0–1.0 0.95 81 0.57829 0.28665 +0.
shear 0–180 2.0 49 0.58667 0.27713 +0.
perspective 0.0–0.001 0.0001 69 0.56199 0.27798 +0.
flipud 0–1.0 0.25 66 0.58743 0.28429 +0.
fliplr 0–1.0 0.75 66 0.58881 0.28008 +0.
```
```
表2:第三组单参数扫荡的最优点（验证集）。
```

### 4.1 变化曲线

#### 4.1.1 前 3 张图

degrees、translate、scale的验证集mAP50-95扫描图按顺序放置：

```
图1:前 3 张参数扫描曲线（degrees/translate/scale）。
```
degrees在整段范围内都未提升mAP50-95，说明大角度旋转对该任务几乎无益；translate在
0.05附近有轻微上升；scale在0.95是最稳的峰值区域。

#### 4.1.2 后 4 张图

shear、perspective、flipud、fliplr的验证集mAP50-95扫描图按顺序放置：

```
图2: 后 4 张参数扫描曲线（shear/perspective/flipud/fliplr）。
```
shear在2.0附近更稳，过大值会拉低精度；perspective在0.0001左右有轻微提升；flipud
与fliplr在非极端翻转下可增益，过强翻转开始破坏场景先验。

### 4.2 参数解释与趋势

- **degrees=0** :大角度旋转在多数施工场景中不物理，轻度旋转也没有带来泛化优势。
- **translate=0.05** :小幅平移增强边缘目标鲁棒性，同时避免过量裁切导致目标切掉。
- **scale=0.95** : 缩放增强对应施工现场人车摄像机距离变化，PPE尺寸分布变化较大时收
    益显著。
- **shear=2.0** :轻微错切模拟姿态与拍摄角度偏差，过强错切会扭曲目标轮廓。
- **perspective=0.0001** :极小透视扰动提升轻微视角泛化，不会引入失真。
- **flipud=0.25** :上下翻转在物理上边界较弱，但在本验证集上有一定正则化效果。
- **fliplr=0.75** :左右翻转符合行人朝向交换场景，能增强不变性，过强（1.0）则分布失真。


## 5 联合调参

### 5.1 联合调参实验协议

#### 先前五组单变量报告后，将参数分为五组物理语义：

- G1：优化器与学习率（optimizer/lr0/lrf/cos_lr/weight_decay/warmup）
- G2：颜色扰动（hsv_h/hsv_s/hsv_v/bgr）
- G3：几何扰动（degrees/translate/scale/shear/perspective/flipud/fliplr）
- G4：复合增强（mosaic/close_mosaic/mixup/cutmix/copy_paste/erasing）
- G5：定位与分类损失权重（box/cls/dfl/dropout）

联合搜索没有做 25 的全量穷举，而采用“分层控制变量”方案。具体做法如下：

1. 固定实验口径：联合阶段全部以baseline的训练设置和训练/评估脚本为基础，只替换待
    试组合参数，避免不同训练策略混淆比较。
2. 定义候选池：先从每个组中取“单变量最优且物理上合理”的取值作为候选基点（例如G
    的degrees=0, translate=0.05, scale=0.95）。
3. 低风险优先组合：先构建G4、G3+G4、G2+G4这类可解释、冲突较小的组合，优先考察其
    是否在召回和定位上协同。
4. 扩展高阶组合：再逐步加入G5，再加入G2与G5的二者协同，最后在“稳态组”基础上
    测试G1版本（auto/SGD两类）与loss版本。
5. 排序与保留：以验证集mAP50-95为第一指标排序，mAP50与Recall作为协同指标；任
    一候选若出现召回大涨但定位明显退化，则保留观察不直接定为最优。

```
图3: Top-7联合候选在验证集上的mAP50与mAP50-95对比。
```

### 5.2 联合结果与最优模型

```
排名 Run 组组合 Precision Recall mAP50 mAP50-
1 joint_auto_all_with_loss_phys G2+G3+G4+G5 0.61791 0.58992 0.60349 0.
2 joint_all_no_loss_phys G1+G2+G3+G4 0.58821 0.63737 0.62725 0.
3 joint_g3_g4_stable_auto G3+G4 0.69168 0.52723 0.59845 0.
4 joint_g4_g5_loss_auto G4+G5 0.62159 0.58470 0.60187 0.
5 joint_g4_stable_auto G4 0.61506 0.58947 0.59952 0.
6 joint_all_with_loss_phys G1+G2+G3+G4+G5 0.62516 0.61745 0.61399 0.
7 joint_g2_g4_stable_auto G2+G4 0.66106 0.57376 0.58607 0.
```
```
表3: 联合调参主要候选（验证集）
```
最优配置 采用第一名joint_auto_all_with_loss_phys：

- mAP50:95(B) = 0. 29679
- 相对baseline提升+0. 02266

其参数详见第 1 节公式序列（含box=10.0, cls=1.0, dfl=0.5, dropout=0.0）。

#### 为什么这样选

#### • 先选G2+G3+G4主要是为了覆盖现场可观测到的光照、几何和成像复合变化，避免模型

#### 只记住单一视角。

#### • 这些参数的扰动幅度都控制在“轻扰动”范围，既能抑制过拟合，又不至于把场景语义结

#### 构改写太多。

- G5中较高的box权重（10.0）强化框定位偏好，能更紧地约束PPE小目标边界。

## 6 模型对比与新图像验证

四类模型（raw、baseline、group3-mid、joint）在新图像集上的输出如下。这里按每张图先给
出可视化，再给出对应结论，重点讨论joint的行为及其优劣。

```
图像 Raw Baseline Group3_mid Joint
data1 3 13 9 11
data2 3 9 9 10
data3 8 19 17 18
data4 4 18 14 24
data5 6 21 20 28
data6 5 17 14 15
```
```
表4: 四组模型在new_test_dataset（ 6 张图）上的框数量（conf=0.25）。
```

### 6.1 data

Raw Baseline Group3-mid Joint

```
图4: data1: 4模型并排对比。
```
现象：raw仅有 3 个框，明显存在漏检；baseline和joint在该图对主要目标有明显补齐，joint
的框数为 11 （略低于baseline的 13 ）。分析（聚焦 **joint** ）：joint在这张图中表现为“中等保
守的高召回”，补齐了多处未检测目标，但局部框数量没有baseline多，说明其对这个场景中的
小目标并未全面拉满检测头，也未出现严重误检爆炸，鲁棒性较baseline略好于baseline的过
度响应风险。

### 6.2 data

Raw Baseline Group3-mid Joint

```
图5: data2: 4模型并排对比。
```
现象：raw只有 3 个框；baseline与group3均为 9 ，joint为 10 。分析（聚焦 **joint** ）：joint在
该样本只带来小幅收益（+1），主要体现在对背景边界较复杂区域的补检能力。与data1相比，
joint在该场景没有引发明显过拟合式误报，说明该图像并未触发其不稳定行为。

### 6.3 data

Raw Baseline Group3-mid Joint

```
图6: data3: 4模型并排对比。
```
现象：raw只有 8 个框，三类微调模型检出量显著提高，baseline 19、group3 17、joint 18。分
析（聚焦 **joint** ）：相比baseline，joint保持了高召回特征却减少了 1 个框，说明它在这类目标密
集场景下有“更稳定但未过度激进”的特性。与此同时，group3仍保持较接近的检测密度，说


明几何扰动在该样本对泛化贡献更直接，joint的组合优势主要来自多组参数协同而非额外的纯
几何增益。

### 6.4 data

Raw Baseline Group3-mid Joint

```
图7: data4: 4模型并排对比。
```
现象：raw 4个框，baseline 18，group3 14，joint 24，joint检测数明显更多。分析（聚焦 **joint** ）：
这是joint优势和风险同时显现的样本：它在复杂结构区域把召回拉到了最高，但框数大幅增长
也提示可能有轻微重复框和边界扩张。对工程安全检测任务，召回率提升是有意义的，但一旦
重复与错位增多，会拖累IoU分布，解释了后续mAP50-95仍可能下降。

### 6.5 data

Raw Baseline Group3-mid Joint

```
图8: data5: 4模型并排对比。
```
现象：raw 6个框，baseline 21，group3 20，joint 28。分析（聚焦 **joint** ）：joint在该图中再次
显著抬高检测密度，但由于边界和目标密集，部分新增框更可能来自结构噪声与局部重叠区域。
与baseline比，joint把“少检风险”压低了，但对精确边界的控制不一定更稳定，意味着在严
格IoU指标上收益被部分抵消。

### 6.6 data

Raw Baseline Group3-mid Joint

```
图9: data6: 4模型并排对比。
```

现象：raw 5个框，baseline 17，group3 14，joint 15。分析（聚焦 **joint** ）：这是joint相对不
佳的一张：相较baseline明显没有继续提升检出密度，显示其在该场景下对小目标抑制较明显，
可能受置信度分布影响。说明joint并非每张图都最强，它的收益是条件性的，主要在目标较复
杂或边界模糊时发挥，简单/清晰场景下并不一定优于baseline。

### 6.7 new_test_dataset 上联合调参效果的综合判断

从six图可见，joint在data4、data5这类复杂场景中检出最积极，在data6上优势减弱甚至低
于baseline。这类“高方差”行为与最终验证统计一致：joint的召回率高于baseline，说明它擅
长找更多候选；但精度并未同步提升，特别是在细定位指标上（mAP50-95）有回落，典型原因
是检测框更偏向覆盖式策略，出现轻度过检和边界波动。

在实际施工安全监测任务里，若目标是尽量不漏检，joint可能更适合作为前置信号增强阶段；若
目标是同时兼顾严格误检和边界质量，仍需在阈值后处理/ NMS后处理和置信度标定上继续收
敛。

为什么 **joint** 没有成为综合最优 原因并不在单一参数“更好”，而在于优化目标上的冲突。
joint融合了G2（颜色扰动）、G3（几何扰动）、G4（合成增强）和G5（损失加权）。这会同时
放大模型对小目标的响应，也放大其对低信息区域的响应。结果是：

- 召回率上升：更多框被激活，漏检下降；
- 定位质量下降风险上升：一部分新框偏离真实框，导致mAP50-95下降；
- 精度波动：在某些图像中能提升，某些图像中受场景背景干扰反而抑制有效检测。

## 7 结果汇总

### 7.1 测试集结果（用于最终对比）

```
模型 Precision Recall mAP50 mAP50-
raw（未微调） 0.0038 0.0153 0.0005 0.
Baseline（默认参数微调） 0.5810 0.4990 0.5300 0.
Joint Best（联合最优） 0.5550 0.5370 0.5400 0.
```
```
表5:测试集对比结果。
```
测试集上联合最优在Recall与mAP50有明显改善，Precision与mAP50-95未显著超越base-
line，整体偏向“增加检出、容许一定误检增加”的方向。

更深入地看，joint的优化目标并非单纯追求单一精度指标，而是通过G2+G3+G4+G5的协同
放大特征覆盖面。这个策略在小样本场景中能显著提高“检出不遗漏”的机会，在复杂或遮挡图
像里尤其有效，这也是在data4、data5上joint的检出增长最明显的直接表现。但是，这种放
大也会把一部分边界不确定区域吸引进候选集，导致边界定位抖动和冗余框增加，在mAP50:
这类对框形一致性更敏感的指标上难以继续拉开差距。换言之，joint的优势是“宽覆盖”，弱点
是“边界精致度”，这与工程上“先召回后筛选”的策略目标是匹配的。若后续继续优化，优先
方向是：提高候选框质量（如更严格的置信度分布校准、NMS/soft-NMS再调，或对joint模型
加一轮轻量后处理微调），而不是单纯再加大增广强度。


## 附录：参考文献

## 参考文献

[1] Ultralytics. Construction ppe dataset - ultralytics docs.https://docs.ultralytics.com/
zh/datasets/detect/construction-ppe. Accessed 2026-06-08.

[2] Ultralytics. Ultralytics construction ppe dataset api page. https://docs.ultralytics.
com/zh/datasets. Accessed 2026-06-08.

[3] Ultralytics. Ultralytics oﬀicial construction-ppe dataset yaml. https://
github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/
construction-ppe.yaml. Accessed 2026-06-08.

[4] Ultralytics. Yolov8 model documentation. https://docs.ultralytics.com/zh/models/
yolov8. Accessed 2026-06-08.

[5] Ultralytics. Yolov8 train mode documentation. https://docs.ultralytics.com/zh/
modes/train. Accessed 2026-06-08.


