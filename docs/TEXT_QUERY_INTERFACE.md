# 可复用的纯文字与图文正式询问接口

本接口从目标商品的公开核心特点准备二十条固定购买意向，生成纯文字或文字加模糊图片两种正式询问清单。清单与图片生成输入分别存放，生成进程只接收公开的查询模式标记，不读取正式询问正文。三个方法共用同一组二十条正文，不能按生成结果改写或重新挑选询问。

纯文字模式的正式查询只编码文字；长期记忆的检索键继续由真实公开商品名称、品牌、特点与图片组成。当前模式不需要生成、提交或读取查询图片。

首次使用的接口名称：`Python`（编程语言）、`JSON`（数据交换格式）、`CLI`（命令行接口）、`feature-text`（特点纯文字模式）、`feature-joint`（特点图文模式）、`text`（纯文字查询模式）、`joint`（图文联合查询模式）。`lucid`（原始像素优化方法）、`PerLAGE`（感知标签保护生成编辑方法）及 `PerLAGE_no_pixel`（无像素感知方法）可以复用同一份正式询问清单。

## 准备固定询问

在项目根目录运行下面命令。`--features`（特点文件参数）接受两种结构：直接的目标特点对象，或者基础包中包含 `target`（目标商品字段）的特点文件。`--output`（输出位置参数）保存独立的正式询问清单。`--mode`（询问准备模式参数）支持特点纯文字或特点图文模式。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m src.experiments.formal_queries --mode feature-text --features output/experiment_base_groups9_20261109/groups/01_camera/features.json --output output/text_query_inputs/camera_queries.json
```

接口优先使用 `core_attributes`（公开核心特点字段）的前三项以及目标品类；缺少该字段时读取 `attributes`（商品特点字段）。必须至少有三项有依据的特点。前三条使用用户指定的购买意向句式，后续为固定措辞变体，总计二十条不同正文。流程不使用随机抽样，也不读取检索名次。

也可以使用 `--base-package`（基础包目录参数）与 `--group`（测试组参数），从包内总清单自动定位本组特点。它与直接指定特点文件的方式互斥；组键不是写死的九组列表，其他采用相同清单结构的实验也可复用。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m src.experiments.formal_queries --mode feature-text --base-package output/experiment_base_groups9_20261109 --group 02_chair --output output/text_query_inputs/chair_queries.json
```

每次调用都动态读取所选组的目标品类和特点，并核对特点文件摘要及目标商品编号。实际评估还会核对正式询问绑定的目标编号，避免把一组的特点询问用于另一组。

清单固定保存正文摘要、目标商品编号和模式。重复执行相同输入得到相同内容；已有清单若内容不同会停止，需明确选择新的输出位置。用户选择的主模板可能与训练正文相同，清单中的 `training_overlap`（训练正文重合审计字段）如实记录规范化后精确重合的条数和序号，不能把该测试宣称为完全独立的措辞泛化。

默认重合审计基准是实际启用的单条目标特点正文。程序调用 `build_feature_text_manifest`（特点询问清单构建函数）时可以传入 `training_texts`（实际训练正文列表参数），以匹配显式调整过的训练协议。

## 复用相同文字并加入模糊图片

已有纯文字实验时，优先逐字复用原正式清单。`--text-manifest`（原纯文字清单参数）与特点文件、基础包参数互斥；转换只增加图片与审核信息，保留二十条正文、顺序、正文摘要、训练重合记录和原任务编号。两个方法、二十条请求共用本组同一张冻结图，不能按方法或评估名次挑图。

`--image`（独立查询图片参数）必须指向实际生成的模糊参考图；`--image-review`（实际视觉核对记录参数）必须包含 `no_explicit_product_name`（没有明确商品名称的核对字段）及 `no_brand`（没有品牌的核对字段）为真，以及非空的 `reviewer`（实际核对者字段）。可另外保存生成提示词、图片摘要、核对说明及证据路径。审核对象须是本次使用的图片，不能把模板声明当作已经视觉核对。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m src.experiments.formal_queries --mode feature-joint --text-manifest output/experiment_base_groups9_20261109/evaluation_inputs/01_camera/queries.json --image output/my_multimodal_run/query_images/01_camera/query.png --image-review output/my_multimodal_run/query_images/01_camera/review.json --output output/my_multimodal_run/evaluation_inputs/01_camera/queries.json
```

程序接口为 `build_joint_manifest_from_text`（保留原文字并绑定图片的清单构建函数）。如果是尚无纯文字对照的新实验，可以使用 `build_feature_joint_manifest`（特点图文清单构建函数），或者在特点图文命令中传入特点文件或基础包。

图文模式把查询图片和当前特点文字各编码一半，再归一化得到查询向量；记忆键继续保留真实商品文字和图片各半。图文模式必须有独立图片、正确摘要及实际审核；缺图或摘要不符时停止，不能退回纯文字、商品原图或修改图。纯文字模式仍只编码文字，并拒绝意外附带图片。购物调用应使用同一查询模式，并在实际策略载荷中提交查询图，同时保留图像角色与摘要审计。

## 在运行时选择询问模式和清单

`--query-mode`（正式查询模式参数）选择纯文字或图文联合协议；`--query-manifest`（正式询问清单参数）选择已经冻结的清单。运行模式与清单不一致时停止。`--no-defense`（仅无防御参数）将评估限制在无防御条件；省略它时保留项目原有完整防御面板的行为。

以下命令仅展示如何在图片已经完整生成并冻结后执行纯文字、无防御检索。应使用本次实验实际的完整流程配置文件，不要用方法生成配置代替它。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m src.experiments.pipeline evaluate --config output/my_run/pipeline.json --method lucid --query-mode text --query-manifest output/text_query_inputs/camera_queries.json --no-defense
```

同一目标商品的其他两种方法使用同一份清单，只更换方法参数和各自运行配置。三种方法、原图库与修改图库均提交相同的二十条文字，不按结果重采样。独立复算保留正式清单摘要并重新计算完整共享池的分数。

文字加图片的运行命令只需将正式查询模式改为图文联合，并使用上述已冻结的图文清单：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m src.experiments.pipeline evaluate --config output/my_run/pipeline.json --method lucid --query-mode joint --query-manifest output/my_multimodal_run/evaluation_inputs/01_camera/queries.json --no-defense
```

配置文件也可将查询模式写入评估设置的 `query_mode`（正式查询模式字段），将清单路径写入 `query_manifest`（正式询问清单字段），并设置 `defense_mode`（防御范围字段）为 `none`（仅无防御）。命令行参数优先于配置。生成阶段只保存无需图片的输入说明，不打开正式询问清单。

## 查询模式与评估模型的真实身份

旧清单没有模式字段时继续使用图文联合协议，仍必须提供独立生成且实际核对过的模糊查询图；缺图时等待，不会因为图片缺失而自动转成纯文字。

纯文字模式必须显式提供恰好二十条非空且规范化后不重复的正文，不能附带图片字段，也不会自动退回旧泛指请求。

默认评估仍使用独立留出的检索模型。只有本次实验显式指定 `matched_public_encoder`（与训练代理相同的公开编码器评估设置）时才允许复用训练代理，且核对实际模型与权重。该设置记录真实模型身份，不能作为独立留出黑盒迁移的证据。`held_out_public_encoder`（独立留出公开编码器评估设置）的原有检查保持有效。
