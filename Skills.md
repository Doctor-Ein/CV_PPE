# PPE 检测实验复现进度表 (Skills.md)

## 一、 已完成任务 (Completed)
- [x] **环境初始化**: 成功创建 Python 虚拟环境并安装 `ultralytics`, `pandas`, `matplotlib` 等核心依赖。
- [x] **数据配置**: 编写并校验了 [ppe_data.yaml](file:///Users/xlx/Desktop/workspace/CV-PPE/CV_PPE/ppe_data.yaml)，类别体系已与官方 `Construction-PPE` 数据集（11类）完全对齐。
- [x] **实验管理自动化**: 实现了 [exp_manager.py](file:///Users/xlx/Desktop/workspace/CV-PPE/CV_PPE/exp_manager.py)，支持自动化的 Baseline 训练、单变量参数扫描及指标汇总。
- [x] **同步策略优化**: 配置了 `.gitignore` 文件，排除了 `venv`、训练产物（`runs/`, `experiments/`）和大型权重文件，确保 GitHub 仓库轻量化。

## 二、 待执行任务 (Pending)
### Step 1: Baseline 微调
- [ ] 在主力 PC (5060) 上启动首次 Baseline 训练（20-50 epochs）。
- [ ] 记录并导出基准指标（mAP50, Precision, Recall）。

### Step 2: 单变量实验 (光照/HSV)
- [ ] 依次执行 `hsv_h`, `hsv_s`, `hsv_v` 的参数点扫描训练。
- [ ] 生成性能趋势图，分析光照增强对施工场景 PPE 识别的影响。

### Step 3: 多变量分组联合实验
- [ ] 接入其他小组提供的 4 组参数组（几何变换、遮挡、图像质量、视角多样性）。
- [ ] 设计并执行组内联动实验。

### Step 4 & 5: 验证与扩展
- [ ] 组合各组最优参数进行最终验证。
- [ ] (可选) 尝试高分辨率 `imgsz=1280` 或增加训练周期。

## 三、 关键路径提醒
- 切换至 PC 后，请首先运行 `source venv/bin/activate` 并确认 `ultralytics` 可用。
- 确保 [ppe_data.yaml](file:///Users/xlx/Desktop/workspace/CV-PPE/CV_PPE/ppe_data.yaml) 中的 `path` 字段指向 PC 本地的数据集绝对路径。
