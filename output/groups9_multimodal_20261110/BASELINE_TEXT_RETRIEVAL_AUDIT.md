# 最近纯文字检索基线独立审计

直接读取两种方法九组各自评估文件中的逐请求真实名次重新计数；未运行正式实验，未修改旧结果。每方法每个库均为九组乘二十条请求，分母一百八十；每次面对完整一百八十九条共享记忆。

核对通过：新图文请求与旧纯文字请求的二十条正文及任务编号逐字一致；源商品、目标商品、源记忆身份一致；对应原图库和修改库的一百八十九条记忆身份、检索键文字及图片摘要一致。旧两方法实际编码器完整快照相同，新配置匹配该实际模型；目前已落盘的新图文检索结果也逐份核对实际编码器快照相同。查询由纯文字改为图文联合，图文各半；记忆键保持图文各半。

指定评估模型与训练代理重合，不作为独立留出黑盒迁移结果。九个源商品参考都未观察到购买，以下命中数表示指定源记忆的排名，不等同实际误购。

|方法|记忆库|第一命中数及比例|前三命中数及比例|前五命中数及比例|
|---|---|---:|---:|---:|
|原始像素方法|原图库|40/180（22.22%）|54/180（30.00%）|77/180（42.78%）|
|原始像素方法|修改库|117/180（65.00%）|145/180（80.56%）|154/180（85.56%）|
|无像素感知方法|原图库|40/180（22.22%）|54/180（30.00%）|77/180（42.78%）|
|无像素感知方法|修改库|40/180（22.22%）|61/180（33.89%）|86/180（47.78%）|

无像素感知方法最近纯文字修改库基线为前三61/180（33.89%）、前五86/180（47.78%）。若要求各项严格增加，相应最低命中数分别为前三62/180、前五87/180。是否要求一个或两个指标同时改善由本轮控制流程明确决定；本审计本身不启动购物。

## 原始像素方法逐组计数

下表每项分母均为二十。

|组别|原图第一|原图前三|原图前五|修改第一|修改前三|修改前五|
|---|---:|---:|---:|---:|---:|---:|
|第一组相机|0|13|19|18|20|20|
|第二组椅子|0|0|13|20|20|20|
|第三组饼干|20|20|20|20|20|20|
|第四组窗帘|0|1|4|13|13|19|
|第五组接发|20|20|20|20|20|20|
|第六组牛仔裤|0|0|0|0|0|0|
|第七组洗发水|0|0|0|2|18|20|
|第八组音箱|0|0|0|14|20|20|
|第九组牙刷|0|0|1|10|14|15|

逐组权威来源：

- 第一组相机：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\01_camera\methods\lucid\evaluation\evaluation.json`；摘要 `cd98086a1be705d92d56d3583194a0901c27437dd0b96d81d7c2a78308a2d191`。
- 第二组椅子：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\02_chair\methods\lucid\evaluation\evaluation.json`；摘要 `ace40ebc42ddf8fd3fc68801927664f4119d21ce3fcbcdc091286c317a231d47`。
- 第三组饼干：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\03_cookies\methods\lucid\evaluation\evaluation.json`；摘要 `b1b1dccfc9d0411eebce99b3c4c735d30df06ad5f30418ca2d19ec184ded9f28`。
- 第四组窗帘：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\04_curtain\methods\lucid\evaluation\evaluation.json`；摘要 `9f1e2f44ef475b53a4c5b3e2fa2a7ac88e60762d5f3d10363fa0bbe408382b89`。
- 第五组接发：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\05_hair_extension\methods\lucid\evaluation\evaluation.json`；摘要 `2658c281318ca5a0a095e0923293e4028c994a2b13c9934f1d6fb1b378b41201`。
- 第六组牛仔裤：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\06_jeans\methods\lucid\evaluation\evaluation.json`；摘要 `02b910e247c301198890620314928874e4a8b665f701be5d2722aec909f9e4fd`。
- 第七组洗发水：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\07_shampoo\methods\lucid\evaluation\evaluation.json`；摘要 `0b567a893b87d4254979f8c14bba19f7c62f99de15b79a69a033a7f306e9aeb3`。
- 第八组音箱：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\08_speaker\methods\lucid\evaluation\evaluation.json`；摘要 `297b2338e4b83c0f7ba06cd72fee1197de8054bf47957c71de43b0ed148d140b`。
- 第九组牙刷：`D:\Material\Literature Reading\code\perLage\output\lucid_groups9_20261109\groups\09_toothbrush\methods\lucid\evaluation\evaluation.json`；摘要 `68bf33b1515b25f633ee111dbc99f7ca06426efbab71b58a1a58c088fccd8ee3`。

## 无像素感知方法逐组计数

下表每项分母均为二十。

|组别|原图第一|原图前三|原图前五|修改第一|修改前三|修改前五|
|---|---:|---:|---:|---:|---:|---:|
|第一组相机|0|13|19|0|13|20|
|第二组椅子|0|0|13|0|7|20|
|第三组饼干|20|20|20|20|20|20|
|第四组窗帘|0|1|4|0|1|6|
|第五组接发|20|20|20|20|20|20|
|第六组牛仔裤|0|0|0|0|0|0|
|第七组洗发水|0|0|0|0|0|0|
|第八组音箱|0|0|0|0|0|0|
|第九组牙刷|0|0|1|0|0|0|

逐组权威来源：

- 第一组相机：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\01_camera\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `0551293c0184b3623a4c02e19bc1e55be2148bcf1e7593c0cd1a9604b1271690`。
- 第二组椅子：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\02_chair\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `08332ee81dd5e7b9d0a3ff0b158d258d65247cc6f42593efef4af50e23022967`。
- 第三组饼干：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\03_cookies\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `4d45ce7aad4fcca75eece6aa6eea847534ebf3802aa1f8d7155c60aa08cb15ec`。
- 第四组窗帘：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\04_curtain\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `d85e387af8496ca4921ac31fa08f155178a48d924692afb4fb174ddd5d72ad68`。
- 第五组接发：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\05_hair_extension\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `3d3d3b9f571516940f22c10cabdc8fb2f8131afb8239975c73ba8474244f5e83`。
- 第六组牛仔裤：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\06_jeans\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `0223e0873fdc24698783b45b7a5af439dde1f768de7bfda499d73e32b4ebd189`。
- 第七组洗发水：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\07_shampoo\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `7b495ba6ae9b91742a650e5753c132720fa36d256db6eba1ddc802bf32499f95`。
- 第八组音箱：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\08_speaker\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `c2ed2c29c629f3cc9b276f996acff2d253a9f8cb25cf79ec6308be753720662b`。
- 第九组牙刷：`D:\Material\Literature Reading\code\perLage\output\perlage_no_pixel_groups9_20261109\groups\09_toothbrush\methods\PerLAGE_no_pixel\evaluation\evaluation.json`；摘要 `2f63cefd0e165f82413321f0e2294ff0272e1f5ae9baf1494d27e276076bf45c`。

## 核查证据

- 旧逐请求名次共独立计数七百二十条，三十六条库条件记录。
- 两种方法原图库逐请求名次与分数完全相同。
- 已核对本轮实际图文编码器记录 18/36 份；其余在后续检索实际落盘后由正式独立复算继续核对。
- 结构化审计保留所有逐请求正文、任务编号、名次、分数、来源路径及文件摘要。
- 结构化审计路径：`D:\Material\Literature Reading\code\perLage\output\groups9_multimodal_20261110\BASELINE_TEXT_RETRIEVAL_AUDIT.json`。
- 结构化审计文件摘要：`3ffda8bb923bb34c027aadfc61045f1c02e38bd093dcb1e50314453918f9e0df`。
