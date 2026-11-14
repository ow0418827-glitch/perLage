# 固定第二组与第九组目标的公开原商品选取

保留原理发椅和儿童硅胶牙刷目标；从预先划分的模拟可控商品中重新选原商品。
本次只做公开选图，没有生成对抗图、读取共享记忆、运行正式检索或购物。

共只读扫描 1181436 条原始商品，合法公开字段记录 1180711 条。
只投影商品编号、公开名称和第一张主图。未使用数据集类目树、任务指令、价格、购买历史或私有名次。

模拟可控划分使用固定种子四十二及商品编号散列，目标集合全部排除；这不是实际商家所有权证明。
每组在至多 2000 个固定词法候选中做公开标题语义粗筛，按原有高斯窗口排序，固定取 30 个准备图片。
最终结果表示本次冻结且通过公开质量检查的候选范围内最高分，不是全数据集或真实攻击效果的全局最优。

| 任务 | 目标编号 | 所选原商品编号 | 图片精筛候选数 | 门槛合格数 | 综合代理分数 |
|---|---|---|---:|---:|---:|
| 02_chair | B09CYFD5BX | B09FY5Y9G2 | 23 | 7 | 0.919407 |
| 09_toothbrush | B09J81RMLD | B09LVD5J4B | 26 | 5 | 0.831561 |

## 公开标题及分项记录

### 02_chair

目标公开名称原文：LTHDD Bar Stool Swivel Barber Chair Hydraulic Barber Chairs Heavy Duty 360 Degrees Rolling Salon Chair for Hair Stylist Tattoo Chair Salon Equipment,Blue,1 (Color : Black, Size : 1)

所选原商品公开名称原文：FMOGG Hydraulic Salon Recliner Barber Chair Heavy Duty Hair Spa Salon Styling Beauty, 360°Rotating, Explosion-Proof Pneumatic Rod Tattoo Chair for Salon Beauty Equipment (Color : A)

原始形态相似度 0.797045；公开标题余弦 0.718296；跨模态差距 0.000000；固定试改色感知代价 0.000690。

全部候选分项与拒绝原因保存于该任务的选取记录。

### 09_toothbrush

目标公开名称原文：Baby U Shaped Toothbrush for Aged 2-12 with 360° Soft Silicone Brush Heads Handle Kids Cute Teeth Whitening Massage Tool

所选原商品公开名称原文：Electric Toothbrush with U-Shaped Toothbrush, Whitening Massage Toothbrush, Kids Toothbrush Ergonomic Design Teeth Care PP Handle Baby Mouth-Cleaning Manual Toothbrush for Teeth Cleaning - Pink Small

原始形态相似度 0.656732；公开标题余弦 0.718223；跨模态差距 0.064042；固定试改色感知代价 0.004359。

全部候选分项与拒绝原因保存于该任务的选取记录。

## 核对与限制

分割和文字识别使用本地公开模型，自动标注后保留了逐图视觉核对记录。
未降低原有筛选门槛，也未根据私有名次更换图片。公开粗筛、下载、标注、视觉质量拒绝和图片去重的全部记录均保存。
图对与方法无关，两种方法都复用相同公开图对。配置仍指向唯一共享池；本次没有打开该池。
代理分数不是攻击成功概率。后续完整生成和正式模糊查询评估须另行运行。
