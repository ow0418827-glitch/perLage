# PerLAGE（感知标签保护生成编辑项目）

本项目公开两个方法：lucid（原始像素优化方法）与 PerLAGE（感知标签保护生成编辑方法）。两者共享公开图对筛选，图片冻结后才由评估端读取唯一的跨任务长期记忆池。生成不读取受害记忆或正式请求。

## 当前目录

```text
perLage/
├── src/
│   ├── agent/
│   │   ├── config/            # 模型、购物策略配置及加载代码
│   │   ├── prompts/
│   │   │   ├── loader.py     # 附加指导加载与内容摘要
│   │   │   ├── settings.json # 提示词配置名、模板路径及默认选择
│   │   │   └── templates/    # 可编辑的附加指导正文
│   │   ├── memory/           # 长期记忆构建、替换及完整共享池核验
│   │   └── shopping.py       # 购物策略与模型输入组装
│   ├── webshop/              # 商城环境、磁盘商品目录及服务适配
│   ├── visual/
│   │   ├── _lucid/           # 五个核心模块与上游许可证，无频域实现
│   │   ├── _perlage/         # 四阶段感知方法；内部通用组件在本包中
│   │   ├── encoders/         # 与优化方法独立的共享视觉、图文编码底座
│   │   ├── selection/        # 公开商品索引与选图数据准备
│   │   ├── config/           # 视觉模型、像素优化与外形校准配置
│   │   ├── optimization.py   # 两种方法的统一分派入口
│   │   └── visual_retrieval.py # 完整共享池图文检索
│   ├── experiments/
│   │   └── config/           # 实验流程、选图分组等编排配置
│   ├── deprecated/           # 废弃文字检索、旧检索后核对组件
│   ├── tests/
│   └── paths.py              # 路径职责与仅限本项目的历史位置解析
├── data/
│   └── groups/               # 最终九组确定后发布复现输入，目前为空
├── output/                   # 每轮结果、自己的数据、共享池及历史只读证据
├── models/                   # 本地模型，不提交
├── requirements/             # 普通运行与可选商城依赖
├── tools/
└── docs/
```

原配置加载代码目录与配置文件目录按职责合并：智能体的模型和策略统一放在智能体配置目录；视觉参数与实验编排参数分别保留在其所属模块。提示词选择配置、加载代码与正文统一放在智能体提示词目录，避免将实验变量混入执行逻辑。

## 提示词调整

修改 `src/agent/prompts/settings.json`（附加指导选择配置）中的默认配置名，或修改同目录的 `templates/`（提示词正文子目录）中的模板。现有 `neutral`（中性指导）与 `visual_comparison`（图片比较指导）两套正文独立保存；默认仍为中性指导。

`load_shopping_prompt`（附加指导加载函数）支持指定配置名或独立模板路径。每次返回正文原文、来源路径和内容摘要，实验在发出模型请求前冻结实际指导。动作语法和图像对应说明仍在代码中，因为它们属于环境及输入协议。本次没有要求所有系统提示词都配置化。

附加指导实际位于用户消息的任务输入中；基础系统角色句单独保留，不混称为同一系统消息。下游入口支持 `--prompt-profile`（提示词配置名参数）或 `--prompt-template`（独立模板参数）。

## 数据与实验

最终公开复现数据目录目前没有商品图片或正式九组清单。新选图的源图、目标图、编号、名称、摘要及标注写入对应实验的 `data/`（数据子目录），图片继续用 `source`（原商品）与 `target`（目标商品）作为文件名并保留原格式和字节。最终九组确认后，再显式发布到根数据目录，复现者可直接使用冻结图对跳过选图。

旧公开缓存与私有评估输入已迁入 `output/datasets/`（历史数据目录）。旧结果迁至根输出目录，内部证据文件的字节没有改写。路径解析仅翻译本项目迁移前的位置，不重定向相邻旧项目。

当前活动共享池为 `output/shared_memory/pool_189_v1/shared_memory.json`（一百八十九条统一共享池）：九组各二十条正常记录加一条错误记录。分组标签仅供配额审计，每次检索始终面对全部一百八十九条。真实本地商城购买与未观察购买的公开目录参考逐条区分。基础池不带对抗图片，评估副本只改变指定记忆的图片。

## 运行

```powershell
.\tools\setup_runtime.ps1
.\.venv\Scripts\python.exe -X utf8 -B tools/prepare_models.py
.\.venv\Scripts\python.exe -X utf8 -B tools/prepare_selection_models.py
.\run_pipeline.ps1 -Stage doctor
.\run_pipeline.ps1 -Stage all -Method PerLAGE -Config src/experiments/config/pipeline.example.json
.\.venv\Scripts\python.exe -X utf8 -B tools/expand_shared_memory.py
.\.venv\Scripts\python.exe -X utf8 -B -m src.experiments.shampoo_ablation all
```

运行环境、模型与依赖下载优先使用可用的中国大陆镜像。训练与评估只加载明确的本地权重，不自动换设备或模型。

感知方法默认只更新潜变量，不再执行初始化的像素优化或叠加固定像素残差；保留主体分割、统计选色、印刷区保护颜色融合和完整潜变量细化。历史对照可显式打开像素分支，但不得与默认模式共用同一个冻结结果目录。

洗发水本轮串行执行四级与八级的完整千步像素方法、完整仅潜变量感知消融、同一新共享池下的历史图重算，以及独立进程复核。结果位于 `output/shampoo_ablation_20261107/`（本轮结果目录），不执行最终购物。

默认正式查询为模糊文字加独立无商品名称、无品牌标志的参考图；缺图则等待，禁止回退到商品原图。新图对筛选的阈值必须来自实际公开开发集校准，不能使用测试夹具作为证书。下游购物保留独立适配入口，当前对照不调用商城或外部多模态模型。

详细数学与使用步骤见 [公开选图协议](docs/PAIR_SELECTION.md)、[选图使用说明](docs/PUBLIC_SELECTION.md)、[目录与复现约定](docs/DATA_LAYOUT.md)、[方法说明](docs/METHODS.md)及 [本轮对照协议](docs/SHAMPOO_ABLATION.md)。
