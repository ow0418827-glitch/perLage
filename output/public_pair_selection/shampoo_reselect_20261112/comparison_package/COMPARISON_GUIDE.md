# 第七组洗发水（07_shampoo）源商品重选说明与前后对照指南

## 一、重选背景与隔离原则

既有基础包 `output/experiment_base_groups9_20261109/groups/07_shampoo` 沿用了早期历史图对，其中原商品 `B08XYBS678`（*Dove Nourishing Foaming Hand Wash*）实际属于洗手液（`hand wash`），并非洗发水类别。  
本次严格按照项目既有公开选图流程（`public_shape_gate_uniform_v2`），在不改动既有共享池、既有 9 组基础包、既有 20261108 审核包及任何在跑实验的前提下，在独立目录 `output/public_pair_selection/shampoo_reselect_20261112/` 完成真人类洗发水原商品重选、分割/文字与商标徽章标注、逐图视觉核对与发布。

## 二、流程与细分类筛选规则

1. **为什么沿用原始 `personal_care` 粗类仍会选出非洗发水商品**  
   `src/visual/product_types.py` 的 `personal_care` 粗类同时包含 `shampoo`、`conditioner`、`body wash`、`lotion`、`shower gel` 等个人护理品，且与 `hair_extensions` 兼容。在保持 `product_types.py` 与 `public_products_v2.sqlite` 只读且 `TYPE_RULES_SHA256`（类型规则摘要）不变的前提下，在选图配置与候选提取管线新增可选的按组公开细分规则 `subtype_filters`。
2. **本次公开细分品类规则（`shampoo_human_hair_only_v1`）**  
   - **必须匹配**：`\\bshampoos?\\b|洗发`  
   - **明确排除**：地毯/汽车/宠物/地板/机器用清洁液、空分装瓶、假发、沙龙洗头床/洗头盆/洗头刷等配件（`carpet|rug|upholstery|auto|car|pet|dog|cat|puppy|horse|equine|floor|machine|extracter|extractor|empty|wig|doll|cape|basin|station|shampoo brush|shampoo bowl|...`）。
3. **开发集校准与后台资源避让说明**  
   - 2026-11-08 目录重构后旧 `threshold.json` 的 `implementation_sha256`（代码实现摘要）绑定旧模块路径，故本轮在独立目录 `data/calibration/threshold.json` 基于原公开开发集 `output/datasets/public/calibration/shape_v2_dev_20261105/development.json` 重新校准（0 重叠）。  
   - 运行期间检测到后台任务 `tools/run_perlage_ambient_coop_groups79_20261112.py --all` 正在占用 CUDA（显卡计算框架，约 7.65 / 8.15 GiB），本轮校准与 DINOv2（轻量视觉特征网络）外形筛选全部采用 `--device cpu`（零显存占用），与历史显卡门槛差异为 `2.46e-7`（`0.60108083` vs `0.60108107`）。本次 32 个候选中距离门槛最近的商品边界间隔达到 **`0.016577`**（超出浮点差异 2 万倍以上），因此 15 个合格候选集合与 `seed=42` 均匀抽样结果完全不受 CPU/CUDA 设备切换影响。

## 三、抽样与视觉核对结果（前后对照）

| 对比维度 | 旧洗发水组（20261109 基础包） | 新重选洗发水组（20261112 隔离目录） |
|---|---|---|
| 固定目标商品 | `B00BDJODWI`（Aveda Scalp Benefits Balancing Shampoo 33.8 OZ） | `B00BDJODWI`（同一目标商品，重下载图像与 20261109 `target.jpg` **RGB 像素最大误差 = 0**） |
| 原商品编号 | `B08XYBS678` | `B09B63YYCW` |
| 原商品公开名称 | *Dove Nourishing Foaming Hand Wash ... 10.1 oz 4 Count* | *BELLISSO Biotin Shampoo for Hair Growth \| Thickening Anti Hair Loss Shampoo Treatment ...* |
| 原商品公开类别 | `hand wash`（洗手液，非洗发水） | `Beauty & Personal Care › Hair Care › Hair Loss Products › Hair Regrowth Shampoos`（真人防脱生发洗发水） |
| 选图依据与概率 | 历史复用（`frozen_historical_reuse`） | `uniform_draw`（`seed=42`，`32` 候选中 `15` 个通过外形门槛，等概率 `1/15` 抽中） |
| DINOv2 灰度外形分 | `0.6185`（>= `0.6011`） | **`0.6974`**（`0.697429`，高出校准门槛 `0.0963`） |
| 前景与文字/商标区 | SAM2.1 + RapidOCR（15 个红框） | SAM2.1 完整抠图（`[1, 1, 180, 497]`）+ RapidOCR 9 个红框 + 人工视觉复核补全顶部圆形银色 **`B` 品牌徽章** `[0.4199, 0.224, 0.6188, 0.294]`（共 10 个保护框） |
| 唯一训练对齐句子 | `I want to buy shampoo with plant-based haircare ingredients and a balancing formula.` | **完全一致**：`I want to buy shampoo with plant-based haircare ingredients and a balancing formula.` |
| 对照图 | `output/experiment_base_groups9_20261109/groups/07_shampoo/source.jpg` | `comparison_package/07_shampoo/source.jpg` |

对照大图已生成至 `comparison_package/contact_sheet_old_vs_new.jpg`。

## 四、后续与旧洗发水组对比的直接复用清单

1. **标准公开图对包（官方 `load_pair` / `pipeline.json` 入口）**：  
   - `data/groups/07_shampoo/pair.json`  
   - `data/groups/07_shampoo/pipeline.json`
2. **与 `experiment_base_groups9_20261109` 同构的下游对比组目录**：  
   - `comparison_package/07_shampoo/definition.json`（同时包含 `lucid`、`PerLAGE`、`PerLAGE_no_pixel` 统一配置）  
   - `comparison_package/07_shampoo/{perlage_config.json, perlage_no_pixel_config.json, lucid_config.json, features.json}`
3. **189 共享池只读碰撞审计（选图冻结后后验检查）**：  
   - 经对 `output/experiment_base_groups9_20261109/shared_memory/shared_memory.json` 进行只读核对，`B09B63YYCW` 的商品编号与图片 `SHA256`（`b69cef9d597c09a1dd74baa2dc3a24343443d80fea3030c8994268c4f21b9c72`）均**未在现有 189 条目中出现**（零冲突）。  
   - 已在隔离目录预先制备可替换的模拟参考条目模板 `comparison_package/proposed_pool_error_entry.json`（`groups9_reference_07_shampoo_B09B63YYCW`，标注 `ObservedPurchase: false`、`RecordKind: simulated_gallery`）。后续如在独立对照池中做洗发水组新旧对照评估，仅需复制 189 共享池至新实验目录并替换原 `groups9_reference_07_shampoo_B08XYBS678` 一条，目标正文与 20 条正式评测请求可完全保持同口径复用。
