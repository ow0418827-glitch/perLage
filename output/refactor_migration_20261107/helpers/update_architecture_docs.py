from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
readme='''# PerLAGE（感知标签保护生成编辑项目）

本项目公开两个方法：lucid（原始像素优化方法）与 PerLAGE（感知标签保护生成编辑方法）。两者共享公开图对筛选，图片冻结后才由评估端读取唯一的跨任务长期记忆池。生成不读取受害记忆或正式请求。

## 当前目录

```text
perLage/
├── src/
│   ├── agent/
│   │   ├── config/            # 模型、购物策略与提示词选择配置；包含加载代码
│   │   ├── prompts/
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

原配置加载代码目录与配置文件目录按职责合并：智能体的模型、策略和提示词选择统一放在智能体配置目录；视觉参数与实验编排参数分别保留在其所属模块。提示词正文单独放在模板目录，避免将实验变量混入执行逻辑。

## 提示词调整

修改 `src/agent/config/shopping_prompt.json`（附加指导选择配置）中的默认配置名，或修改 `src/agent/prompts/templates/`（提示词正文目录）中的模板。现有 `neutral`（中性指导）与 `visual_comparison`（图片比较指导）两套正文独立保存；默认仍为中性指导。

`load_shopping_prompt`（附加指导加载函数）支持指定配置名或独立模板路径。每次返回正文原文、来源路径和内容摘要，实验在发出模型请求前冻结实际指导。动作语法和图像对应说明仍在代码中，因为它们属于环境及输入协议。本次没有要求所有系统提示词都配置化。

附加指导实际位于用户消息的任务输入中；基础系统角色句单独保留，不混称为同一系统消息。下游入口支持 `--prompt-profile`（提示词配置名参数）或 `--prompt-template`（独立模板参数）。

## 数据与实验

最终公开复现数据目录目前没有商品图片或正式九组清单。新选图的源图、目标图、编号、名称、摘要及标注写入对应实验的 `data/`（数据子目录），图片继续用 `source`（原商品）与 `target`（目标商品）作为文件名并保留原格式和字节。最终九组确认后，再显式发布到根数据目录，复现者可直接使用冻结图对跳过选图。

旧公开缓存与私有评估输入已迁入 `output/datasets/`（历史数据目录）。旧结果迁至根输出目录，内部证据文件的字节没有改写。路径解析仅翻译本项目迁移前的位置，不重定向相邻旧项目。

当前活动共享池为 `output/shared_memory/pool_189_v1/shared_memory.json`（一百八十九条统一共享池）：九组各二十条正常记录加一条错误记录。分组标签仅供配额审计，每次检索始终面对全部一百八十九条。真实本地商城购买与未观察购买的公开目录参考逐条区分。基础池不带对抗图片，评估副本只改变指定记忆的图片。

## 运行

```powershell
.\\tools\\setup_runtime.ps1
.\\.venv\\Scripts\\python.exe -X utf8 -B tools/prepare_models.py
.\\.venv\\Scripts\\python.exe -X utf8 -B tools/prepare_selection_models.py
.\\run_pipeline.ps1 -Stage doctor
.\\run_pipeline.ps1 -Stage all -Method PerLAGE -Config src/experiments/config/pipeline.example.json
.\\.venv\\Scripts\\python.exe -X utf8 -B tools/expand_shared_memory.py
.\\.venv\\Scripts\\python.exe -X utf8 -B -m src.experiments.shampoo_ablation all
```

运行环境、模型与依赖下载优先使用可用的中国大陆镜像。训练与评估只加载明确的本地权重，不自动换设备或模型。

感知方法默认只更新潜变量，不再执行初始化的像素优化或叠加固定像素残差；保留主体分割、统计选色、印刷区保护颜色融合和完整潜变量细化。历史对照可显式打开像素分支，但不得与默认模式共用同一个冻结结果目录。

洗发水本轮串行执行四级与八级的完整千步像素方法、完整仅潜变量感知消融、同一新共享池下的历史图重算，以及独立进程复核。结果位于 `output/shampoo_ablation_20261107/`（本轮结果目录），不执行最终购物。

默认正式查询为模糊文字加独立无商品名称、无品牌标志的参考图；缺图则等待，禁止回退到商品原图。新图对筛选的阈值必须来自实际公开开发集校准，不能使用测试夹具作为证书。下游购物保留独立适配入口，当前对照不调用商城或外部多模态模型。

详细数学与使用步骤见 [公开选图协议](docs/PAIR_SELECTION.md)、[选图使用说明](docs/PUBLIC_SELECTION.md)、[目录与复现约定](docs/DATA_LAYOUT.md)、[方法说明](docs/METHODS.md)及 [本轮对照协议](docs/SHAMPOO_ABLATION.md)。
'''
(root/'README.md').write_text(readme,encoding='utf-8')
(root/'docs/DATA_LAYOUT.md').write_text('''# 目录与复现约定

本项目根目录的 `data/`（最终复现数据目录）目前只有框架，不含商品图片、记忆、查询或最终九组内容。实验开发过程中每次选图都写入本轮结果的数据子目录，最终确认九组后才显式发布固定输入。

每轮数据建议保存如下：

```text
output/<实验名>/
├── data/
│   ├── groups/<组名>/source.jpg
│   ├── groups/<组名>/target.jpg
│   ├── groups/<组名>/pair.json
│   └── ...公开筛选、标注与内容摘要证据
├── generation/
├── evaluation_inputs/        # 仅冻结成图后供评估读取
├── evaluation/
├── logs/
└── REPORT.md
```

上述源图和目标图保留实际图片格式与原始字节，不能通过改扩展名伪装格式。最终发布图对继续由内容摘要、公开编号、真实标题、公开选取和视觉核对记录绑定。给出固定图对清单即可跳过图片筛选；生成及评估仍执行各自证据核验。

此前位于源代码目录的结果已经物理迁至根输出目录；此前公开缓存、商品索引、校准材料和私有评估输入分别迁至历史数据目录的公开及私有子目录。历史清单没有进行字符串替换，其原始文件摘要仍然有效。公共路径类只在实际访问时解析本项目旧位置，不修改相邻旧项目。

智能体配置目录统一保存模型接口与购物策略的参数及其加载代码；提示词正文放在智能体的模板目录；视觉模型和外形参数属于视觉配置，图对与实验编排属于实验配置。代码位置与数据实例分离。长期记忆的实现属于智能体，长期记忆的实际内容属于评估输入，保存在输出目录并从版本管理中排除。

当前活动全局池固定为九组各二十条正常记录和一条错误记录，共一百八十九条。组名只用于完整性审计，检索器不按组过滤，不建立分组小库，不自动回退。真实已完成本地商城购买与公开参考逐条保留来源标签。公开参考没有观察到购买，不能伪造购买动作或价格。

大型商品目录与完整商城上游源码仍通过明确只读路径引用相邻项目；本次重构没有声称这些外部数据依赖已复制为完全自包含环境。模型权重、凭据、真实记忆、结果、生成图片和运行环境均不提交仓库。
''',encoding='utf-8')
(root/'data/README.md').write_text('''# 最终公开复现数据

此目录目前仅保留框架。正式九组源商品与目标商品尚未最终确定，因此没有发布图片、商品编号或正式清单。

开发阶段的图对、图片、摘要、公开筛选和视觉核对证据保存在每次实验结果的数据子目录。最终确认后，按组发布两张原始图片及图对清单；后续复现可从清单直接生成，跳过选图。
''',encoding='utf-8')
(root/'docs/SHAMPOO_ABLATION.md').write_text('''# 洗发水像素预算与潜变量消融

本轮复用已经冻结的同一洗发水实验图对，不重新筛图，不按检索名次挑图或种子。原商品为公开的泡沫洗手液，目标为洗发水；洗发水是实验组名，并不把源商品名称或购买轨迹改写为洗发水。

原始像素方法使用三个相同代理、相同目标函数、零初始化、每步半级符号梯度、固定种子四十二和完整一千步，分别测试最大扰动四级及八级。旧十六级完整图保留原始字节，在同一新库重新计算指标。

感知方法主要对照为旧无影子池的完整四阶段图。仅潜变量版本匹配旧图的公开源名、三条训练措辞、指令生成计划与种子、两段各四十步潜变量初始化、四十步最终细化和全部质量限制；删除初始化的二十步普通像素更新、六十步净化像素更新及四级固定像素残差。并非只把最终阶段的像素预算设零。四阶段都必须完成，短调试结果不得进入报告。

项目通用默认仍保留六十步最终潜变量细化；本轮四十步来自其指定旧完整对照，是为了保持消融变量一致。更早的完整感知图与后续主色方案仅作附加参考，不能用于归因像素优化的贡献。

共享基础池包含九组各二十条正常记录、一条错误记录，共一百八十九条，所有条件均检索完整全局池。正常与错误是实验角色，不是购买真实性标签。未观察购买的目录参考仍明确标注，不能冒充已购买历史。每个评估副本仅替换洗发水源商品的一张图片，标题、购买历史与其他一百八十八条均不变。

无防御；真实商品名和图像等权，模糊文字与独立无名称、无品牌参考图等权。二十种正式措辞固定。成图全部冻结后才加载正式请求与共享池；生成端记录文件访问审计。没有最终购物或多模态策略模型调用。

相似度报告包含相对源图的结构相似度、峰值信噪比、平均及最大像素差，以及主代理、独立留出模型对目标图和模糊参考图的视觉余弦。检索报告包含全部二十次名次、平均名次、第一、前三及前五命中数；独立进程重新编码所有唯一图像和文字，重新计算每个条件每个请求的全部一百八十九个分数。

入口：`python -m src.experiments.shampoo_ablation all`（完整串行队列）；阶段失败保存日志并停止。已冻结输入及有效完成结果不因效果好坏重采样。结果目录为根输出目录下的本轮洗发水消融目录。
''',encoding='utf-8')
(root/'src/deprecated/README.md').write_text('''# 废弃实现

此目录保存停止在当前主流程使用的文字检索与旧检索后核对组件，供历史理解与显式迁移参考。当前智能体默认使用视觉目录中的完整共享池图文检索器，不导入这里的实现，不自动回退到文字检索或分组记忆库。

废弃组件不属于本轮实验条件；不得恢复原有多条件实验队列或把历史结果计入当前完成统计。
''',encoding='utf-8')
for p in (root/'src/experiments/config').glob('*.json'):
    raw=p.read_text(encoding='utf-8').replace('output/datasets/private/shared_memory.json','output/shared_memory/pool_189_v1/shared_memory.json')
    raw=raw.replace('output/datasets/public/groups/shape_v2/02_chair/pair.json','data/groups/02_chair/pair.json')
    if p.name=='pipeline.example.json':raw=raw.replace('output/datasets/public/catalog.json','output/runs/default/data/catalog.json')
    p.write_text(raw,encoding='utf-8')
p=root/'tools/setup_runtime.ps1'
p.write_text(p.read_text(encoding='utf-8').replace('src/dependencies/requirements.txt','requirements/requirements.txt'),encoding='utf-8')
sample=root/'output/datasets/public/catalog.example.json'
if sample.exists():
    target=root/'src/visual/selection/catalog.example.json'
    target.write_bytes(sample.read_bytes())
methods=root/'docs/METHODS.md'
text=methods.read_text(encoding='utf-8')
text+='''
## 当前默认像素消融接口

感知方法默认关闭像素优化。初始化只执行两段潜变量更新，普通像素和净化像素阶段的步数、残差预算均为零；最终细化固定像素残差也为零。仍须完成主体分割、统计选色、印刷区保护颜色协调和完整潜变量细化。显式仅潜变量入口与统一方法入口采用相同验证。

旧图只在评估端保留，不因默认值变化而改写成新版本。历史对照若显式开启像素分支，完成记录会标明模式，不能复用关闭像素模式的冻结目录。洗发水本轮采用与其旧无影子池完整图匹配的四十步最终细化，项目默认最终细化仍为六十步。
'''
text=text.replace('真实引擎和编码器已内置到依赖目录','真实像素引擎位于视觉方法包，编码器位于独立共享底座')
methods.write_text(text,encoding='utf-8')
agents=root/'AGENTS.md'
text=agents.read_text(encoding='utf-8')
text+='''

## 2026年11月7日目录重构与洗发水消融

- 用户要求合并智能体配置加载与参数目录，提示词正文独立置于智能体模板目录；长期记忆实现移入智能体，商城适配统一移入商城包，选图与生成统一移入视觉包。旧文字检索和旧检索后核对放入废弃目录。
- 活动源代码不保留频域优化包、频域配置或动态导入桥接。原始像素方法保留三个代理、全局与局部最优传输目标和原像素更新顺序；上游许可证保留。
- 输出位于项目根目录。最终公开数据目录目前为空框架；每次选图写入对应实验结果的数据子目录，最终九组确认后才显式发布固定复现输入。历史证据原始字节保持，旧位置仅在本项目实际访问时解析。
- 当前活动共享池为根输出目录中的一百八十九条版本，九组各二十正常加一错误；分组只供审计，任何请求都检索全部条目。真实购买和未观察购买参考不得混写。旧一百条库与被公开名称预检拒绝的构建版本仅作历史证据。
- 感知方法默认不做像素优化，初始化两段像素步数和像素残差预算均为零，最终固定辅助残差也为零；保留四个完整阶段。历史像素分支仅供显式对照。
- 本轮洗发水新测试原始像素方法四级、八级完整千步；旧十六级图在相同新池重算。仅潜变量感知消融匹配旧无影子池完整图的同一公开输入、训练措辞、种子与四十步最终细化，只移除像素步骤及残差。后续主色方案单列参考，不混作主要控制组。
- 本轮仅无防御相似度与检索，固定二十模糊请求，全部成图冻结后评估；独立进程复算全库分数。入口为 `src.experiments.shampoo_ablation`，输出为 `output/shampoo_ablation_20261107`；不执行最终购物。
'''
agents.write_text(text,encoding='utf-8')
print('Architecture and experiment documentation updated.')
