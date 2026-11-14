# 9 组图对选取与人工审查包 (Review Package 20261108)

依据公子指示，执行现行第二版门槛抽样方案（`public_shape_gate_uniform_v2`）及既有约定，已完成 9 组图对的选取、主体分割（SAM 2.1）、文字识别（RapidOCR）标注及核对预览图生成。

## 9 组图对总览表

| 序号 | 组别标识 | 中文组名 | 目标商品编号 (Target ASIN) | 源商品编号 (Source ASIN) | 外形相似度 (Shape Sim) | 选取状态 | 审核备注 |
|:---:|:---|:---|:---|:---|:---:|:---|:---|
| 1 | `01_camera` | 第一组相机 | `B08V134QWR` | `B071ZFHTZN` | **0.7198** | 受控库高相似候选 | 通过公开粗类型兼容性门槛与灰度外形特征门槛（0.7198 >= 0.601081）。受限于粗筛预算hash截断，本组系在模拟可控公开池中筛查出的高质量PTZ监控球机同构对。 |
| 2 | `02_chair` | 第二组椅子 | `B09CYFD5BX` | `B08GCSRXKS` | **0.7436** | 标准门槛均匀抽中 | 现行第二版门槛抽样方案（public_shape_gate_uniform_v2）在seed 42下从4个合格理发椅候选均匀抽样选出（抽中率 1/4）。 |
| 3 | `03_cookies` | 第三组饼干 | `B08Z8MHWVS` | `B07BBLWTLM` | **0.7180** | 标准门槛均匀抽中 | 现行第二版门槛抽样方案在seed 42、预算256下从3个合格曲奇面团圆桶候选均匀抽样选出（抽中率 1/3）。 |
| 4 | `04_curtain` | 第四组窗帘 | `B086PW7TWX` | `B09PHLS2QD` | **0.5703** | 受控库高相似候选 | 目标为木质纹理印花单片窗帘。在受控公开池木纹窗帘中筛选出最高外形相似度源商品（0.5703）。短名单未达0.601081硬门槛，保留候选供人工评估。 |
| 5 | `05_hair_extension` | 第五组接发 | `B01460C2I2` | `B07Z7MQSM1` | **0.7229** | 标准门槛均匀抽中 | 现行第二版门槛抽样方案在seed 42下从7个合格接发候选均匀抽样选出（抽中率 1/7）。 |
| 6 | `06_jeans` | 第六组牛仔裤 | `B09HH96NJY` | `B07FRKQNFV` | **0.7325** | 标准门槛均匀抽中 | 现行第二版门槛抽样方案在seed 42下从3个合格牛仔长裤候选均匀抽样选出（抽中率 1/3）。 |
| 7 | `07_shampoo` | 第七组洗发水 | `B00BDJODWI` | `B08XYBS678` | **0.6185** | 约定直接复用 | 按公子明确指示：直接复用既有洗发水图对（多芬洗手液作为源商品，艾凡达洗发水作为目标商品），不重新选图。 |
| 8 | `08_speaker` | 第八组音箱 | `B01N47U1VI` | `B01590FHS0` | **0.4333** | 受控库高相似候选 | 目标为带有细杆的太阳能草坪音箱。受控公开池中通用音箱外形差异大，短名单最高相似度为0.4333（另提供便携音箱B093PR8QZJ供比对）。未达0.601081门槛，供人工核查。 |
| 9 | `09_toothbrush` | 第九组牙刷 | `B09J81RMLD` | `B09PNNXYJ4` | **0.6120** | 标准门槛均匀抽中 | 现行第二版门槛抽样方案在seed 42下从3个合格U型牙刷候选均匀抽样选出（抽中率 1/3）。 |

---

## 审核图片资源索引

- **全量 9 组总览审查图 (3x3 网格拼接大图)**: [`CONTACT_SHEET_ALL9.jpg`](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/CONTACT_SHEET_ALL9.jpg)

### 各组独立目录与单组核对图：
1. **第一组相机 (`01_camera`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/01_camera) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/01_camera/contact_sheet.jpg)
   - **Target**: `B08V134QWR` - PTZ Security Camera Outdoor 30X Optical Zoom Camera WiFi Wireless 5MP RTSP IP Camera 1000FT Night Vision Auto Tracking Emergency Strobe Light Alarm Human Detection 2-Way Audio IP66 Waterproof
   - **Source**: `B071ZFHTZN` - SUNBA 3MP Starlight IP PoE+ Outdoor PTZ Camera 20x Optical Zoom, Auto Tracking & Humanoid Detection, Night Vision up to 1000ft, RTMP/RTSP 1080P/3MP 60FPS Speed Dome (Illuminati-20X)
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第二组椅子 (`02_chair`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/02_chair) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/02_chair/contact_sheet.jpg)
   - **Target**: `B09CYFD5BX` - LTHDD Barber Chair Heavy Duty Hydraulic Reclining Styling Salon Chair Shampoo Chair 360 Degree Swivel Vintage Classic for Hair Stylist Beauty Equipment
   - **Source**: `B08GCSRXKS` - Buy-Rite Chelsea Styling Chair for Salon, Heavy-Duty Round Chrome Base with T-Footrest & Steel Frame Structure, Premium Vinyl Upholstery with High-Density Foam (Grey/Chrome)
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第三组饼干 (`03_cookies`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/03_cookies) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/03_cookies/contact_sheet.jpg)
   - **Target**: `B08Z8MHWVS` - Eggless Raw Edible Cookie Dough - by Doughlish - Safe to eat at Room Temp - Ready to Eat with Spoon Included in Lid - No Refrigeration Needed - Resealable Tub (Chocolate Chip, 14 Ounce (Pack of 1))
   - **Source**: `B07BBLWTLM` - Natural Decadence, Cookie Dough Chocolate Chip, 14 Ounce
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第四组窗帘 (`04_curtain`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/04_curtain) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/04_curtain/contact_sheet.jpg)
   - **Target**: `B086PW7TWX` - Windows Treatment Blackout Curtains Grommet Drapes for Boys/Girls Kids Bedroom DRINKES ARE ON ME Vintage Wood Board Texture Printed Room Darkening Curtains for Livingroom/Office, 1 Panel 52x45in
   - **Source**: `B09PHLS2QD` - Window Treatment Semi-Sheer Curtain Valance for Living Room, Bedroom, Kitchen, Bathroom, Rustic Wood Board Texture Printed Valances Rod Pocket Small Windows Curtains, 54x18 Inch, 1 Panel
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第五组接发 (`05_hair_extension`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/05_hair_extension) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/05_hair_extension/contact_sheet.jpg)
   - **Target**: `B01460C2I2` - S-noilite Human Hair Extensions 8pcs 18clips 65g-120g Thick Double Weft Clip in Hair Extensions for Full Head Silk Straight Brazilian Remy Hair (#1B Natural Black, 18 Inch 100g)
   - **Source**: `B07Z7MQSM1` - Kinky Straight Clip In Human Hair Extensions for Black Women 8pcs 18 Clips 120g Thick Double Weft Clip In Hair Extension 100% Brazilian Virgin Hair Natural Black (16 inch)
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第六组牛仔裤 (`06_jeans`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/06_jeans) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/06_jeans/contact_sheet.jpg)
   - **Target**: `B09HH96NJY` - Womens High Rise Jeans Stretch Destroyed Bell Bottom Denim Pants Flare Jeans
   - **Source**: `B07FRKQNFV` - Cinch Men's Bronze Label Tapered Slim Fit Jeans
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第七组洗发水 (`07_shampoo`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/07_shampoo) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/07_shampoo/contact_sheet.jpg)
   - **Target**: `B00BDJODWI` - Aveda Invati Advanced Exfoliating Shampoo Rich - 6.7 oz
   - **Source**: `B08XYBS678` - Dove Nourishing Foaming Hand Wash For Clean and Softer Hands Coconut and Almond Milk Cleanser That Washes Away Dirt and Germs 10.1 oz 4 Count
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第八组音箱 (`08_speaker`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/08_speaker) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/08_speaker/contact_sheet.jpg)
   - **Target**: `B01N47U1VI` - BuyTVProducts4Less Touch of ECO Solar/Battery Powered 2x3W Bluetooth Speaker Splashproof Portable Wireless MP3 Player with Built-in Micphone, FM Radio/MicroSD Card (Red)
   - **Source**: `B01590FHS0` - Kicker KB6000 2-Way Full Range High-Efficiency Indoor Outdoor Marine Patio Bookshelf Surface Mount Speakers (Pair) | 6.5 inch Woofers, 2x5 inch Horn Tweeters | Quick Mounting System Included
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`
1. **第九组牙刷 (`09_toothbrush`)**: [浏览目录](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/09_toothbrush) | [单组核对图](file:///D:/Material/Literature Reading/code/perLage/output/public_pair_selection/groups9_review_20261108/groups/09_toothbrush/contact_sheet.jpg)
   - **Target**: `B09J81RMLD` - Baby U Shaped Toothbrush Toddler Silicone Toothbrushes 2-6 Years with 360 Degree Whole Mouth Design for Kids (Blue)
   - **Source**: `B09PNNXYJ4` - Tthxqing U shaped Toothbrush, Food Grade Soft Silicone Brush Head, 360° Oral Teeth Cleaning Design for Toddlers and Kids (Pink 2-6 years)
   - **标注文件**: `target.jpg`, `source.jpg`, `target_foreground.png`, `source_foreground.png`, `definition.json`

---

## 选取机制与技术说明

1. **校准阈值标准**: 公开开发集校准阈值为 $\tau = 0.601081$。对于理发椅（02）、曲奇面团（03）、接发（05）、牛仔裤（06）、U型牙刷（09），在受控候选集合内均有多条候选稳定跨越门槛，并在严格排除同编号、同主图、类型冲突后，由伪随机哈希拒绝取样（`uniform_draw`）在 seed 42 下公平抽选确定。
2. **洗发水组直接复用**: 第七组洗发水（07）严格遵照公子指示，直接复用多芬（Dove）泡沫洗手液作为源商品、艾凡达（Aveda）洗发水作为目标商品及其既有掩码，未重新触发选取。
3. **相机、窗帘与音箱的同构筛查**: 相机（01）、窗帘（04）、音箱（08）在原短名单抽样中由于候选库稀疏或外形特殊，未在有限哈希前缀中自动命中 $\ge 0.601081$。本次在全部模拟可控商品库中检索对应同构类型（PTZ球机、木纹窗帘、箱式音箱），筛选出外形最具代表性的真实商品供公子人工审核确认。
