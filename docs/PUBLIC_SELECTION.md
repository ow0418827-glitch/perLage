# 轻量公开商品选图入口

当前入口为 `tools/select_public_pairs.py`（公开选图工具），配置使用第二版。默认保留两组公开目标，粗类型候选预算32张，使用公开开发集门槛和固定种子均匀抽样。完整数学定义见[选图协议](PAIR_SELECTION.md)。

## 准备本地模型与真实开发标签

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/prepare_selection_models.py
```

选图阶段只需要一个冻结的小型视觉模型，准备工具优先使用中国大陆镜像。正式生成的分割、文字识别与生成模型仍在相应阶段准备。

把 `src/configs/shape_development.example.json`（开发标签模板）复制到公开数据目录，填入实际公开图片、来源、核对者、逐对说明及零或一标签。模板中的空标签不能用于校准。正负样本须等量；开发商品及其图片不能作为正式目标或候选。若需隔离同系列变体，可额外登记开发保留商品。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py calibrate --development src/data/public/calibration/development.json
```

证书写到配置指定的公开校准路径，包含实际逐对分数、阈值、拟合误差及模型身份。没有真实校准证书时停止，不将旧阈值或合成测试标签当作正式开发结果。更换模型、预处理或标签须另建证书与运行目录。

## 从原始公开数据取候选并抽图

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py prepare --products 'D:\Material\Literature Reading\code\MINJA-main\rap\webshop\data\items_shuffle.json'
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py select
```

第一次建立本项目自己的公开字段索引；不复制原项目源码或实验结果，也不使用原始文件中的私有类目、任务、历史、价格和奖励。后续不同目标复用索引。`--verify-source`（重验原始文件内容参数）会额外重新计算原文件摘要；正常复用检查源文件元数据与类型规则身份。

未识别或多义类型必须提供配置中的公开类型核对清单。每条核对包括粗类型、核对者、依据及当前公开名称摘要。程序不把未知类型默认为兼容。

候选按粗类型和既定模拟可控划分截取，不计算标题相似度。下载失败保留证据，不自动补足到预算或改变种子。外形门槛之后，从合格集合抽样。空集合保存逐候选拒绝原因，返回状态码三，停止本组。

## 只标注选中的图对并发布

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py annotate
```

此阶段才调用主体与文字标注模型，仅处理实际抽中的源图与目标图；不会给全部三十二张候选做分割和文字区域标注。

查看运行目录中的公开核对图，将实际核对结果写入公开资产目录的 `selected_visual_reviews.json`（选中图对视觉核对文件）。格式沿用逐图审核：顶层为实际核对者与当前标注清单摘要，记录中包含图片摘要、提议摘要、接受状态、必要的主体框及印刷区修正。不合格则停止，不能按正式名次重选。程序不生成自动通过的视觉声明。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py publish
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py report
```

也可以运行一次完整选图命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/select_public_pairs.py all --products 'D:\Material\Literature Reading\code\MINJA-main\rap\webshop\data\items_shuffle.json'
```

未填写真实视觉核对记录时，完整选图命令在标注后返回状态码二，等待核对，再使用发布阶段继续。这是公开标注输入缺失，不是额外审批流程。

## 正式文件与两种方法

默认运行目录为 `src/output/public_pair_selection/shape_v2/`（选图输出）；公开准备目录为 `src/data/public/selection_runs/simple_v2/`（候选与标注）；正式分组目录如下。

```text
src/data/public/groups/shape_v2/02_chair/
  source.jpg
  target.jpg
  pair.json
  pipeline.json
  metadata/
```

图片保留原始格式与字节，实际扩展名可能为其他受支持格式。协议版本目录保留旧两组已冻结的文件，不覆盖此前 source、target 或生成图。

生成时使用该目录中的完整流程配置，两种方法共同读取相同 source 与 target；生成准备须已有实际核对的主体和印刷区。新配置的正式模糊查询路径也按运行版本隔离，避免自动复用已经观察过的查询。所有组仍指向一个共享长期记忆池。

本工具不修改共享记忆、不生成方法最终图、不进行正式检索或购物。旧多指标选择只保留历史文件读取，第一版配置不能启动新的旧评分实验。
