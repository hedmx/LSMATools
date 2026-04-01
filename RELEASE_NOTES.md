# Release Notes - LSMATOOLS V15.0

**发布日期**: 2026-03-09  
**版本**: 15.0.0 (Stable)

---

## 🎉 重大升级 (Major Upgrades)

### 1. V15 双重回退机制

**新增功能**:
- ✅ 全局回退：终板线总数 < 5 时自动重启（更换搜索策略）
- ✅ 点数回退：单条终板线 < 28 点时触发，最多 2 轮回退
- ✅ 动态阈值：回退轮次使用更宽松的阈值 [0.80, 0.65]

**影响**: 
- 终板检测鲁棒性大幅提升
- 适应高噪声/退行性变样本
- 向后兼容 V14 的输出格式

---

### 2. 40 条扫描线体系

**改进内容**:
- ✅ 扫描线数量从 33 条扩展到 40 条
- ✅ 沿皮质线 2 法线方向排列（1mm 间距）
- ✅ 覆盖深度范围 1-40mm

**优势**:
- 更精细的椎体前缘/后缘扫描
- 提升解剖结构覆盖完整性
- 支持更复杂的病例分析

---

### 3. 双模态融合前缘检测

**技术改进**:
- ✅ 上升沿精修（黄线）+ 下降沿检测（青色线）并行
- ✅ 密集窗口融合输出红色前缘线
- ✅ 右侧补充点约束 5mm（排除离群点）

**融合流程**:
```python
1. 主流程：上升沿精修 → refined 点
2. 并行方案：下降沿检测 → confirmed 点
3. 合并统计：dense window 过滤 → best_lo 位置
4. 最终输出：best_lo 区间点 + 右侧补充点 → 红色前缘线
```

---

## 📊 算法改进 (Algorithm Improvements)

### V15 法线扫描线

**延续 V13_2 的设计**:
- 40 条法线方向扫描线（1mm 间距）
- 沿皮质线 2 局部法线排列
- 覆盖 1-40mm 深度范围

**优势**:
- 始终垂直于椎管后壁曲线
- 解剖路径唯一
- 比水平平移更符合解剖结构

---

### 双重回退终板检测

**核心流程**:

```python
# 全局回退检查
if len(all_ep_points) < 5:
    # 更换搜索策略，重新扫描
    
# 单条终板线点数检查
for ep_line in [top_curve, bottom_curve]:
    retry_count = 0
    while len(ep_line) < 28 and retry_count < 2:
        # 放宽阈值因子 [0.80, 0.65]
        # 重新检测该终板线
        retry_count += 1
```

**V15 改进**:
- ✅ 全局回退触发条件：总点数 < 5
- ✅ 点数回退触发条件：单条 < 28 点
- ✅ 回退阈值序列：[1.0 (初始), 0.80, 0.65]

---

### 终板线轮廓拼接优化

**问题修复**:
- ✅ 终板线按 row 排序导致多边形自相交 → 改用原始 off_mm 升序
- ✅ 下终板方向错误导致开口 → 反转原始顺序（front 端→c1 端）
- ✅ 终板线超出交叉点 → 用交叉点列坐标截断

**拼接逻辑**（顺时针闭合环）:
```
上终板：c1 端→front 端（原始顺序，off_mm 升序）
前缘线：row 从小（上）到大（下）
下终板：front 端→c1 端（反转原始顺序）
皮质线 1：row 从大（下）到小（上）
```

---

### 椎体掩模标签扩展

**新增功能**:
- ✅ S1 纳入标签映射表（S1=1）
- ✅ 所有椎体标签连续编号 1-9
- ✅ 避免 fallback 到 255 导致 colormap 极端

**标签映射**:
```python
{
    'S1': 1, 'L5': 2, 'L4': 3, 'L3': 4, 'L2': 5, 
    'L1': 6, 'T12': 7, 'T11': 8, 'T10': 9
}
```

---

## 🔧 技术细节 (Technical Details)

### 参数路由体系

**分辨率等级**:

| 等级 | 像素间距 | tol_mm | min_pts |
|------|---------|--------|---------|
| HR   | ≤0.50mm | 2.0mm  | 7       |
| STD  | ≤0.75mm | 2.5mm  | 7       |
| LR   | >0.75mm | 3.0mm  | 6       |

**场强修正**:
- 3T (≥2.5T): depth_thresh × 1.10
- 并行采集：depth_thresh × 0.92

### 关键参数默认值

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `left_off_mm` | 20.0mm | 谷底搜索 ROI 右边界 |
| `right_off_mm` | 40.0mm | 谷底搜索 ROI 左边界基准 |
| `expand_ratio` | 3.0 | 底行扩展倍数（统一参数） |
| `scan_mm` | 40.0mm | 上升沿精修扫描距离 |
| `rise_ratio` | 0.50 | 上升沿触发比例 |
| `drop_ratio3` | 动态 [0.25-0.60] | 下降沿动态阈值 |
| `window_mm` | 6.0mm | 密集窗口宽度 |
| `smooth_k` | 8.0mm | 前缘线平滑窗口 |

---

## 📁 文件结构 (File Structure)

```
LSMATOOLS_git/
├── main.py                     # 主程序入口（交互式菜单）
├── README.md                   # 项目说明（中英双语）
├── LICENSE                     # MIT 许可证
├── requirements.txt            # Python 依赖
├── CONTRIBUTING.md             # 贡献指南
├── RELEASE_NOTES.md            # 版本发布说明
├── .gitignore                  # Git 忽略规则
├── LSMATOOLS_CP/
│   ├── segmentation/           # 分割模块
│   │   ├── spinal_canal.py     # 椎管分割
│   │   └── anterior_edge.py    # 前缘检测
│   ├── preprocessing/          # 预处理模块
│   ├── postprocessing/         # 后处理模块
│   │   ├── visualization.py    # 可视化
│   │   └── export.py           # CSV/NIfTI导出
│   └── config/                 # 配置文件
├── docs/
│   ├── algorithm_design.md     # 算法设计文档
│   └── api_reference.md        # API 参考文档
└── examples/
    └── basic_usage.py          # 使用示例
```

---

## 🚀 安装与使用 (Installation & Usage)

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/LSMATOOLS_git.git
cd LSMATOOLS_git

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 运行主程序（交互式）
python main.py
```

### Python API

```python
from LSMATOOLS_CP.main import test_single_image

# 单病例处理
test_single_image(
    nifti_path='data/W.nii.gz',
    metadata_path='metadata.json',
    output_dir='results'
)

# 或调用核心模块
from LSMATOOLS_CP.segmentation.anterior_edge import find_endplates_on_water_image
from LSMATOOLS_CP.postprocessing.visualization import visualize_results
```

---

## ✅ 测试验证 (Testing & Validation)

### 建议测试场景

1. **典型腰椎场景** (L5/S1 起始)
2. **胸腰段联合扫描** (包含 T12-T8)
3. **退行性变样本** (终板 Modic 改变)
4. **不同分辨率** (HR/STD/LR三档)

### 验证指标

- 椎管分割 Dice 系数 > 0.85
- 终板检测准确率 > 90%
- 前缘线连续性完整度 > 85%

---

## 🐛 已知问题 (Known Issues)

### 限制条件

1. **图像质量要求**:
   - 推荐分辨率：≤0.75mm (STD 等级)
   - 需要 T2 Dixon W+F 配对序列
   - 矢状位扫描平面

2. **解剖覆盖**:
   - 主要优化用于腰椎 MRI
   - 胸椎识别（T12-T8）为保守扩展
   - 更高节段（T7 及以上）可能误差增大

3. **病理场景**:
   - 严重脊柱侧弯可能影响识别准确性
   - 术后内固定金属伪影会干扰检测
   - 重度退行性变可能导致终板漏检

---

## 📝 变更日志 (Changelog)

### V15.0 (2026-03-09)

**新增功能**:
- ✨ V15 双重回退机制（全局 + 点数回退）
- ✨ 40 条扫描线体系（法线方向，1mm 间距）
- ✨ 双模态融合前缘检测（上升沿 + 下降沿）

**改进优化**:
- ⚡ 终板线轮廓拼接优化（off_mm 升序，下终板反转）
- ⚡ 终板线交叉点截断（保证掩模不溢出）
- ⚡ 椎体标签扩展（S1=1，连续 1-9 编号）
- ⚡ expand_ratio 统一为 3.0（搜索/过滤/可视化一致）

**Bug 修复**:
- 🐛 终板线按 row 排序导致自相交 → 保持原始顺序
- 🐛 S1 标签缺失导致 colormap 极端 → 加入标签表
- 🐛 第二次密集窗口跑偏 → 统一 expand_ratio 参数

**技术债务**:
- 🔧 代码注释规范化
- 🔧 开源文档完善（README/LICENSE/CONTRIBUTING）
- 🔧 项目配置现代化

---

### V14.x (继承版本)

**主要特性**:
- 胸椎命名扩展至 T12-T8
- 椎体几何中心标注优化
- 背部线平滑增强（5mm 一致参数）

---

### V13_2 (继承版本)

**核心改进**:
- 终板语义修正（upper/lower 对应关系）
- 下降沿阈值放宽（low_mean3 × 1.3）
- 双模态密集窗口融合
- 左图可视化重构（移除调试元素）

---

## 🙏 致谢 (Acknowledgements)

感谢所有为腰椎 MRI 分析算法研究做出贡献的团队和个人！

本项目继承了 V13_2 的全部功能，并在其基础上进行了重要升级。

---

## 📧 联系方式 (Contact)

- **问题反馈**: GitHub Issues
- **功能建议**: GitHub Discussions
- **邮件联系**: contact@lsma.tools（待设置）

---

**完整版本**: V15.0.0  
**发布日期**: 2026-03-09  
**许可证**: MIT License
