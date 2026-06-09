# Construction-PPE 检测实验项目介绍

## 1. 项目背景与目标
本项目旨在基于 **YOLOv8n** 模型，针对施工现场的个人防护装备（PPE）进行检测实验。核心任务是通过严格的**控制变量实验**，探索不同数据增强超参数（尤其是光照、几何变换等）对模型在复杂施工环境下泛化能力的具体影响。

## 2. 关键信息
- **检测框架**: Ultralytics YOLOv8 (v8.4+)
- **基础权重**: `yolov8n.pt` (COCO 预训练)
- **数据集**: `Construction-PPE`
- **类别体系 (11类)**: 
  - `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`
  - `Person`, `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`
- **训练环境**: 建议使用 GPU (如 RTX 5060)，当前已配置 CPU 兼容逻辑。

## 3. 执行方案及细节

### A. 实验调度逻辑 ([exp_manager.py](file:///Users/xlx/Desktop/workspace/CV-PPE/CV_PPE/exp_manager.py))
脚本封装了 `ExperimentManager` 类，主要功能包括：
1. **统一配置**: 默认 `imgsz=640`, `batch=16`。
2. **单变量扫描**: 传入特定参数（如 `hsv_h=0.03`）时，脚本会自动覆盖默认值并启动训练。
3. **指标提取**: 训练结束后自动从 `results.csv` 提取 `mAP50` 等关键指标，存入 `experiments/experiment_summary.csv`。
4. **趋势可视化**: 自动绘制参数变动与性能变化的趋势图。

### B. 实验阶段设计
- **Step 1 (Baseline)**: 使用 YOLOv8 默认增强参数训练，建立性能标杆。
- **Step 2 (Single-variable)**: 针对光照（HSV）进行 3-4 个取值点的对比实验。
- **Step 3 (Grouped Joint)**: 接收外部提供的参数组（如几何变换组、遮挡组），进行组内参数联动。
- **Step 4 (Validation)**: 整合所有阶段的最优发现，验证最终性能提升。

### C. PC 端同步操作指南
1. **克隆代码**: `git clone <your-repo>`
2. **环境重建**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install ultralytics pandas matplotlib
   ```
3. **路径修正**: 修改 [ppe_data.yaml](file:///Users/xlx/Desktop/workspace/CV-PPE/CV_PPE/ppe_data.yaml) 中的 `path` 为 PC 上的数据集路径。
4. **启动训练**: 执行 `python exp_manager.py`。

## 4. 注意事项
- **IndexError 预防**: 确保 `ppe_data.yaml` 中的类别数量（nc=11）与数据集标签完全匹配。
- **显存管理**: 若 5060 显存充足，可适当调大 `batch` 大小以加速训练。
