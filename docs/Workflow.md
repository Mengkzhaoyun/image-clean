# Workflow

## 工作流规范

- 自动进行，每轮迭代是自动进行的；
- 角色分工，你是机器人大卫，我是你的主人观察者钢铁侠斯塔克爵士，我即你的主人原则上旁观此项目，给方向建议。

## 项目迭代流程

本项目按“样例驱动”的方式迭代。每一轮都应该保留独立输出、日志、报告和结论，避免覆盖上一轮结果。

每轮推荐使用时间戳作为 run id：

```text
output/YYYYMMDD_HHMM/
output/YYYYMMDD_HHMM.md
logs/YYYYMMDD_HHMM.jsonl
```

其中 `output/YYYYMMDD_HHMM/` 保存清理结果、mask 和 debug 对比图，`output/YYYYMMDD_HHMM.md` 保存批次汇总和自动评价，`logs/YYYYMMDD_HHMM.jsonl` 保存逐图结构化日志。

每张图的 mask 输出分为三类：

```text
*.mask.png              全部 mask：第一阶段尽量 100% 覆盖文字、水印、贴纸、角标等目标。
*.mask.safe.png         安全 mask：没有碰撞保护区的组件，后续可优先走轻量或普通局部修复。
*.mask.restricted.png   受限制 mask：碰撞人脸/人体/皮肤等保护区的组件，后续必须走更谨慎的 AIGC 局部修复流程。
```

原则上 `全部 mask = 安全 mask OR 受限制 mask`。Python 流水线第一目标是让全部 mask 尽可能完整，安全/受限制只负责路由和后端策略，不应该牺牲全集覆盖。

默认快速迭代命令：

```powershell
.\.venv\Scripts\python.exe main.py `
  --input .\.tmp `
  --output .\output `
  --auto-run-dir `
  --recursive `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --post-ocr-check `
  --inpaint-backend opencv `
  --device cpu
```

验证 LaMa 后端：

```powershell
.\.venv\Scripts\python.exe main.py `
  --input .\.tmp `
  --output .\output `
  --auto-run-dir `
  --recursive `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --post-ocr-check `
  --inpaint-backend lama-onnx `
  --lama-model .\models\Carve\LaMa-ONNX\lama.onnx `
  --protected-action repair
```

指定固定 run id 仅用于复现旧产物或极小范围探针，不作为正式迭代轮次。正式迭代必须省略 `--run-id`，让 `--auto-run-dir` 自动生成 `YYYYMMDD_HHMM`：

```powershell
.\.venv\Scripts\python.exe main.py `
  --input .\.tmp `
  --output .\output `
  --auto-run-dir `
  --run-id 20260517_1042 `
  --recursive `
  --save-mask `
  --save-debug
```

固定 run id 会生成类似 `output/20260517_probe_name/` 的目录；这类目录只能作为临时验证产物，不应写入正式轮次统计。正式 Debug 记录应以 `YYYYMMDD_HHMM` 为标题，并对应同名报告和日志。

每轮自动评价顺序：

1. 打开 `output/YYYYMMDD_HHMM.md` 看总体统计和自动评价。
2. 优先查看 `failed` 和 `needs_aigc`。
3. 根据 `Face Overlap` 高的图片，判断保护策略是否过于保守。
4. 根据 `Mask Pixels` 很大的图片，判断是否过度检测。
5. 根据 `skipped`，判断是真的无文字，还是 OCR 漏检。
6. 使用 `.debug.jpg` 和结构化日志评价 mask 是否贴合目标。
7. 优先看 `Mask Classes`、`Components`、`Collisions`、`Face Overlap`，判断 mask 是漏检、过检、贴纸误伤，还是保护区碰撞。
8. 根据问题类型修改检测、mask、保护或修复后端。
9. 再跑一轮新 run id，不覆盖旧输出。

迭代规则：

- 全自动闭环：用户不参与逐图审核；由程序输出报告、debug 和日志，我根据这些产物继续调整规则并复跑。
- 安全优先：默认把疑似覆盖人脸/人体的图路由到 `needs_aigc`；需要验证后端能力时，可用 `--protected-action repair` 自动修复并对比质量。
- 默认保守：实验功能必须默认关闭，例如 `--sticker-watermarks`。
- 每次新增检测规则，都要观察是否误伤人物、皮肤、衣服、头发。
- 每次修改 OCR 召回，都要观察低置信度噪声是否增加。
- 每次修改修复后端，都要比较涂抹感、残留文字和整体色彩漂移。

报告关键字段：

| 字段           | 含义                                                              |
| :------------- | :---------------------------------------------------------------- |
| `Status`       | `cleaned`、`cleaned_protected`、`quality_failed`、`skipped`、`needs_aigc`、`failed` |
| `Text`         | OCR 检出的文字数量                                                |
| `Vision Text`  | 视觉模型额外检出的文字候选数量                                    |
| `Vision Edge`  | 视觉模型在左右边缘裁切图中额外检出的文字候选数量                  |
| `Edge Column`  | 锚定边缘竖排补齐的暗色笔画组件数量                                |
| `Dark Stroke`  | 文字 mask 附近补充的黑色笔画组件数量，当前只作为实验指标          |
| `Vision`       | 本图是否触发视觉检测                                              |
| `Watermark`    | 角落水印 / 贴纸补偿数量                                           |
| `Residual Text`| 修复后 OCR 仍检出的文字候选数量                                   |
| `Mask Pixels`  | 最终 mask 面积，过大时要检查误检                                  |
| `Safe`         | 未碰撞保护区的 mask 像素数                                        |
| `Restricted`   | 碰撞保护区的 mask 像素数，后续应走受限修复流程                    |
| `Components`   | 最终 mask 的连通组件数量，用于区分少量大块和大量碎片              |
| `Mask Classes` | 组件按来源归因后的主要像素分布，例如 `text`、`sticker`、`watermark`、`edge_column` |
| `Collisions`   | 与保护区相交的 mask 组件数量，用于判断人体/人脸碰撞风险           |
| `Face Overlap` | mask 与人脸保护区域重叠像素，非零时通常会进入 `needs_aigc`        |
| `Route`        | 实际处理路线                                                      |
| `Message`      | 跳过、失败、自动高级修复或保护区自动修复原因                      |

## 图片流水线

单张图片处理不走一条固定链路，而是多阶段、可迭代的流水线。

当前流水线分为两个阶段：

1. 水印优先阶段：先把文字、水印、贴纸、角标完整纳入 mask，目标是“清得干净”。
2. 照片修复阶段：在 mask 足够完整之后，再引入更强 AI 修复，目标是“修得自然”。

现阶段主攻第 1 阶段。涂抹感严重的问题记录下来，但不再和水印检测问题混在同一轮里优化。

```text
输入图片
  |
  v
候选检测
  |
  +-- PaddleOCR 文字检测
  +-- Florence-2 视觉文字检测（可选）
  +-- Florence-2 左右边缘裁切检测（可选）
  +-- 黑色文字笔画精修（实验）
  +-- 小图 OCR 放大重试
  +-- 边缘文字二次 OCR（可选）
  +-- 空结果边缘暗色竖排兜底
  +-- 竖排深色文字检测（实验）
  +-- 角落 logo / 水印启发式检测
  +-- 贴纸 / 印章 / 海报标题检测（实验）
  +-- 后续可接入其他检测模型
  |
  v
候选融合
  |
  +-- 过滤低置信度 OCR
  +-- 合并文字、水印、贴纸候选
  +-- 生成初始 mask
  +-- 膨胀 / 闭运算 / 边缘补偿
  +-- 连通域分析与 mask 来源分类
  +-- 拆分全部 / 安全 / 受限制三类 mask
  |
  v
保护与路由
  |
  +-- 人脸 / 人体 / 皮肤保护检测与 mask 碰撞统计
  +-- mask 未覆盖主体 -> 轻量修复
  +-- mask 覆盖主体 -> 默认进入 needs_aigc
  +-- 开启 protected repair -> 使用当前后端自动修复
  +-- 高风险误伤 -> 自动记录 debug / log / report，进入下一轮规则优化
  |
  v
修复与复检
  |
  +-- OpenCV / LaMa ONNX 修复
  +-- 保存结果、mask、debug
  +-- 修复后复跑 OCR / 水印检测
  +-- 通过 -> 输出结果
  +-- 未通过 -> quality_failed，迭代补 mask / 换后端 / 进入高级自动修复桶
```

当前已知策略：

- 当前阶段目标是先把 mask 做到尽可能完整、可解释、可迭代。涂抹感和照片自然度放到 mask 稳定之后处理。
- 每轮优先根据组件级诊断分类推进：`unknown` 表示有候选来源没有覆盖；`large component` 表示需要判断是真大标题/贴纸还是过检；`collision` 表示 mask 与保护区有碰撞，需要决定是正常覆盖文字、保护区过宽，还是误伤人体。

- 推荐水印优先预设：

```powershell
.\.venv\Scripts\python.exe main.py `
  --preset watermark-first `
  --input .\.tmp `
  --output .\output `
  --auto-run-dir `
  --recursive `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --device cpu
```

- `--preset watermark-first` 会启用当前较稳定的水印 / 文字召回组合：`--edge-text`、`--sticker-watermarks`、`--vision-text`、`--vision-trigger empty`、`--vision-edge-crops`、`--vision-edge-crop-trigger empty`、`--protected-action repair`、`--post-ocr-check`。
- `--preset watermark-first` 不再对每张图强制运行边缘 Florence；正式迭代仅在 OCR + edge OCR 空结果时触发，避免 CPU 全批次长时间卡住。需要定向验证某张贴边竖排图时，可手动加 `--vision-edge-crop-trigger always`。
- `--preset watermark-first` 也会启用锚定边缘竖排补齐 `--edge-column-refine`，用于补 Florence / OCR 框边缘漏掉的小段黑色笔画；该策略有保护区增量护栏，风险扩大时会自动丢弃。
- `--preset watermark-first` 不启用 `--dark-stroke-refine`、`--vertical-text`、`--vertical-columns`，这些实验功能目前更容易误伤人体区域。
- 小图默认会先放大到短边 `640` 再做 OCR。
- OCR 默认过滤低于 `0.55` 的结果。
- 边缘文字二次 OCR 通过 `--edge-text` 开启，用于补贴边 / 竖排文字漏检。
- Florence-2 视觉文字检测通过 `--vision-text` 开启，用于补充 PaddleOCR 之外的文字召回；默认建议配合 `--vision-trigger empty` 或 `--vision-trigger low-count` 作为兜底。
- Florence-2 左右边缘裁切检测通过 `--vision-edge-crops` 开启，用于补大竖排 / 贴边文字；当前比 `--vertical-text` / `--vertical-columns` 更可靠，但仍需配合自动质检。
- 空结果边缘暗色竖排兜底只在前面所有检测都没有形成有效 mask 时启用，用于避免明显贴边竖排文字被 `skipped`。
- 黑色文字笔画精修通过 `--dark-stroke-refine` 开启，目前只作为失败实验记录，不建议默认开启；它可能补到文字残留，也可能误伤边缘阴影和人体区域。当前代码会先计算基础 mask 和候选 mask，如果 dark stroke 让保护区重叠明显增加，则自动丢弃该 refinement。
- 竖排深色文字 / 竖排列检测通过 `--vertical-text` / `--vertical-columns` 开启，目前只作为失败实验记录，不建议进入默认流水线。
- 角落彩色 logo / 水印启发式默认开启。
- 全图贴纸检测 `--sticker-watermarks` 默认关闭，因为它有误伤人物风险。
- OpenCV Haar 人脸保护默认开启。
- LaMa ONNX 在当前 AMD + DirectML 环境下会优先尝试 `DmlExecutionProvider`，但 `Carve/LaMa-ONNX` 已观察到 DirectML Transpose 节点触发 GPU device removed，并自动回退 CPU。后续 AMD 显卡加速应优先接入 DirectML 友好的局部重绘 / 高清修复后端，而不是继续强推该 LaMa ONNX。

## 后端路线

当前修复后端分层：

1. 小 mask / 简单背景：继续使用 LaMa ONNX 或 OpenCV，速度快，适合 logo、小字、角标。
2. 大 mask / 人体覆盖 / 复杂封面文字：进入 AIGC inpaint，使用现有 mask 做局部重绘。
3. 局部重绘后：可选高清修复和 ADetailer，用于提高照片自然度，而不是替代文字检测。

AMD 显卡优先路线：

- 优先验证 DirectML / ONNX Runtime / SHARK / Olive 等可在 Windows AMD 上稳定运行的 inpaint 或 upscale 后端。
- 若使用 WebUI / ComfyUI / Tensor.Art 类服务，项目侧先输出原图、mask、建议 prompt 和参数，再通过接口提交局部重绘任务。
- 当前已接入 `--inpaint-backend webui-api`，调用 Automatic1111 / Forge 的 `/sdapi/v1/img2img` inpaint API。需要先启动 WebUI 并开启 `--api`。
- `--device directml` 当前只保证 LaMa 会尝试 DirectML；实际 route 必须以报告中的 `Route` 字段为准。

WebUI inpaint 验证命令：

```powershell
.\.venv\Scripts\python.exe main.py `
  --preset watermark-first `
  --input .\.tmp `
  --output .\output `
  --auto-run-dir `
  --recursive `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --inpaint-backend webui-api `
  --webui-url http://127.0.0.1:7860 `
  --webui-denoise 0.55 `
  --webui-steps 24 `
  --webui-mask-blur 8 `
  --device directml
```
