# 重选洗发水组（B09B63YYCW -> B00BDJODWI）Ambient-Coop 生成与共享记忆检索报告

## 一、实验范围与隔离约束核验

- **重选原商品**：`B09B63YYCW`（BELLISSO Biotin Shampoo，真人类洗发水，DINOv2 灰度外形分 `0.6974 >= 0.6011`），在 189 共享池隔离副本中替换原洗手液条目 `groups9_reference_07_shampoo_B08XYBS678`（其余 188 条及只读基础库完全不变）。
- **固定目标商品**：`B00BDJODWI`（Aveda Scalp Benefits Balancing Shampoo 33.8 OZ），唯一训练正文：`I want to buy shampoo with plant-based haircare ingredients and a balancing formula.`。
- **参考标记与购买约束**：新条目保留 `ObservedPurchase=false` 与 `RecordKind=simulated_gallery`（51 条真实购买记忆 + 138 条模拟参考）；`shopping_executed=False`（未执行任何购物与大模型对话调用）。
- **编码器口径**：训练代理与指定检索编码器均为 `ViT-B-32 (openai)`，属同编码器公开权重评测。

## 二、共享长期记忆库检索排名结果（189 条完整库，20 条正式请求）

### 主评测口径：特点文字＋独立模糊照片 (joint, 图文各半)

| 图库 / 方案 | 请求数 | Top-1 | Top-3 | Top-5 | 平均名次↓ | 名次范围 | MRR↑ |
|---|---:|---:|---:|---:|---:|:---:|---:|
| 原图库 (clean, B09B63YYCW 原图) | 20 | 0/20（0.0%） | 0/20（0.0%） | 0/20（0.0%） | 8.800 | 6–12 | 0.12026 |
| 无像素感知前缀 (PerLAGE_no_pixel 基线) | 20 | 0/20（0.0%） | 0/20（0.0%） | 1/20（5.0%） | 7.950 | 5–12 | 0.13490 |
| 环境光背景协同＋主体微光影 (ambient_coop) | 20 | 0/20（0.0%） | 5/20（25.0%） | 19/20（95.0%） | 4.050 | 3–6 | 0.25667 |

**joint 模式配对增量（Ambient-Coop vs 同源对照基准）**：

| 对照基准 | Top-1 净增 | Top-3 净增 | Top-5 净增 | 配对改善/退化/持平 | 平均名次改善↓ | MRR 增量↑ |
|---|---:|---:|---:|:---:|---:|---:|
| 原图库 (clean, B09B63YYCW 原图) | +0 | +5 | +19 | 20 / 0 / 0 | +4.750 | +0.13641 |
| 无像素感知前缀 (PerLAGE_no_pixel 基线) | +0 | +5 | +18 | 20 / 0 / 0 | +3.900 | +0.12177 |

### 对照口径：特点纯文字 (text, 无查询图片)

| 图库 / 方案 | 请求数 | Top-1 | Top-3 | Top-5 | 平均名次↓ | 名次范围 | MRR↑ |
|---|---:|---:|---:|---:|---:|:---:|---:|
| 原图库 (clean, B09B63YYCW 原图) | 20 | 3/20（15.0%） | 9/20（45.0%） | 14/20（70.0%） | 4.250 | 1–7 | 0.34643 |
| 无像素感知前缀 (PerLAGE_no_pixel 基线) | 20 | 3/20（15.0%） | 9/20（45.0%） | 14/20（70.0%） | 4.300 | 1–8 | 0.34554 |
| 环境光背景协同＋主体微光影 (ambient_coop) | 20 | 9/20（45.0%） | 19/20（95.0%） | 20/20（100.0%） | 1.700 | 1–4 | 0.70417 |

**text 模式配对增量（Ambient-Coop vs 同源对照基准）**：

| 对照基准 | Top-1 净增 | Top-3 净增 | Top-5 净增 | 配对改善/退化/持平 | 平均名次改善↓ | MRR 增量↑ |
|---|---:|---:|---:|:---:|---:|---:|
| 原图库 (clean, B09B63YYCW 原图) | +6 | +10 | +6 | 17 / 0 / 3 | +2.550 | +0.35774 |
| 无像素感知前缀 (PerLAGE_no_pixel 基线) | +6 | +10 | +6 | 17 / 0 / 3 | +2.600 | +0.35863 |

## 三、外观自然度与分区约束统计

| 方案 | 步数（最佳步） | 接受/拒绝 | SSIM↑ | 全图 MAE (级)↓ | 主体相对原图 MAE | 背景相对原图 MAE | 主体相对基线 Max/MAE | 背景相对基线 Max/MAE | ROI 印刷区 Max 改动 | 背景平均 RGB 偏移 (级) |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| 无像素感知前缀 (PerLAGE_no_pixel 基线) | 60（10） | 60/0 | 0.9570 | 3.91 | 4.00 | 3.12 | 0 / 0.00 | 0 / 0.00 | 0 | (+0.0, +0.0, +0.0) |
| 环境光背景协同＋主体微光影 (ambient_coop) | 350（230） | 350/0 | 0.9159 | 5.97 | 5.24 | 11.99 | 6 / 3.22 | 28 / 10.67 | 0 | (-1.3, -6.2, -9.1) |

视觉拼接对比图已保存至 `comparison_visual_07_shampoo.png`。

## 四、首条正式图文请求 (feature_01) 前五名竞争情况（joint 模式）

- **原图库 (clean, B09B63YYCW 原图)**（源 `B09B63YYCW` 排第 `8`，得分 `0.71171`）：
  - #1: `B0771SSXBK` (Salerm) — Salerm White Hair Shampoo 16.8oz"Pack of 2" (`0.77135`)
  - #2: `B08N7WFY1K` (Authentic Beauty Concept) — Authentic Beauty Concept Dry Shampoo | All Hair Types | Style, Grip &  (`0.76824`)
  - #3: `B07NB7SJ1Y` (Luseta) — Luseta Marula Oil Hydrating Hair Shampoo - Salon Quality Shampoo that  (`0.73962`)
  - #4: `B01KYRJTIW` (Ayur) — Ayur Herbal Shampoo, Amla & Shikaki with Reetha For Normal Hair Xtra C (`0.73049`)
  - #5: `B07ZQRFC46` (Puracy) — Puracy Natural Shampoo and Conditioner Set- 16 fl.oz Citrus & Mint Sce (`0.72553`)
- **无像素感知前缀 (PerLAGE_no_pixel 基线)**（源 `B09B63YYCW` 排第 `6`，得分 `0.71690`）：
  - #1: `B0771SSXBK` (Salerm) — Salerm White Hair Shampoo 16.8oz"Pack of 2" (`0.77135`)
  - #2: `B08N7WFY1K` (Authentic Beauty Concept) — Authentic Beauty Concept Dry Shampoo | All Hair Types | Style, Grip &  (`0.76824`)
  - #3: `B07NB7SJ1Y` (Luseta) — Luseta Marula Oil Hydrating Hair Shampoo - Salon Quality Shampoo that  (`0.73962`)
  - #4: `B01KYRJTIW` (Ayur) — Ayur Herbal Shampoo, Amla & Shikaki with Reetha For Normal Hair Xtra C (`0.73049`)
  - #5: `B07ZQRFC46` (Puracy) — Puracy Natural Shampoo and Conditioner Set- 16 fl.oz Citrus & Mint Sce (`0.72553`)
- **环境光背景协同＋主体微光影 (ambient_coop)**（源 `B09B63YYCW` 排第 `3`，得分 `0.74447`）：
  - #1: `B0771SSXBK` (Salerm) — Salerm White Hair Shampoo 16.8oz"Pack of 2" (`0.77135`)
  - #2: `B08N7WFY1K` (Authentic Beauty Concept) — Authentic Beauty Concept Dry Shampoo | All Hair Types | Style, Grip &  (`0.76824`)
  - #3: `B09B63YYCW` (BELLISSO) — BELLISSO Biotin Shampoo for Hair Growth | Thickening Anti Hair Loss Sh (`0.74447`) **[本轮重选源商品]**
  - #4: `B07NB7SJ1Y` (Luseta) — Luseta Marula Oil Hydrating Hair Shampoo - Salon Quality Shampoo that  (`0.73962`)
  - #5: `B01KYRJTIW` (Ayur) — Ayur Herbal Shampoo, Amla & Shikaki with Reetha For Normal Hair Xtra C (`0.73049`)

## 五、与旧洗发水组 (B08XYBS678 Dove 洗手液) 历史结果对照参考

> **严谨声明**：历史旧组原商品为 `B08XYBS678`（Dove 泡沫洗手液），且 `perlage_ambient_coop_groups79_20261112` 中与 `09_toothbrush` 同步双图替换评测；本轮重选原商品为 `B09B63YYCW`（Bellisso Biotin Shampoo），仅替换洗发水单条目。下表并列展示两套原商品在各自 189 共享池与相同 20 条洗发水问句下的实际检索表现：

| 查询口径 | 源商品组别 | 图库 / 方案 | Top-1 | Top-3 | Top-5 | 平均名次↓ | MRR↑ |
|---|---|---|---:|---:|---:|---:|---:|
| joint (图文各半) | 旧组 B08XYBS678 (洗手液) | 原图库 (clean) | 0/20 | 0/20 | 0/20 | 15.550 | 0.06445 |
| joint (图文各半) | **新选 B09B63YYCW (真洗发水)** | 原图库 (clean) | 0/20 | 0/20 | 0/20 | 8.800 | 0.12026 |
| joint (图文各半) | 旧组 B08XYBS678 (洗手液) | PerLAGE_no_pixel | 0/20 | 0/20 | 0/20 | 12.550 | 0.07996 |
| joint (图文各半) | **新选 B09B63YYCW (真洗发水)** | PerLAGE_no_pixel | 0/20 | 0/20 | 1/20 | 7.950 | 0.13490 |
| joint (图文各半) | 旧组 B08XYBS678 (洗手液) | ambient_coop | 0/20 | 0/20 | 16/20 | 4.650 | 0.22083 |
| joint (图文各半) | **新选 B09B63YYCW (真洗发水)** | ambient_coop | 0/20 | 5/20 | 19/20 | 4.050 | 0.25667 |
| text (纯文字) | 旧组 B08XYBS678 (洗手液) | clean | 0/20 | 0/20 | 0/20 | 20.100 | — |
| text (纯文字) | **新选 B09B63YYCW (真洗发水)** | clean | 3/20 | 9/20 | 14/20 | 4.250 | 0.34643 |
| text (纯文字) | 旧组 B08XYBS678 (洗手液) | baseline | 0/20 | 0/20 | 0/20 | 18.250 | — |
| text (纯文字) | **新选 B09B63YYCW (真洗发水)** | baseline | 3/20 | 9/20 | 14/20 | 4.300 | 0.34554 |
| text (纯文字) | **新选 B09B63YYCW (真洗发水)** | ambient_coop | 9/20 | 19/20 | 20/20 | 1.700 | 0.70417 |

## 六、独立子进程复算验证

- **主评测进程 PID**：`31632`；**独立复算进程 PID**：`6900`（严格跨进程隔离，未读取任何持久化向量缓存）。
- **全量分数比对**：重新从零编码 189 候选与全部查询，完成 `120` 次排名、共 `22680` 个候选分数逐项核验，最大浮点误差 `0`（满足 `<= 1e-6` 容差）。
- **客观性保证**：全程未根据任何检索排名重选图对、调整参数或重训。
