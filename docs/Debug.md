# Debug

## 编制规范

- 隐私保护，不要透露测试文件名，这属于隐私

## 20260517_1042

### 本轮成果

- 生成独立输出目录 `output/20260517_1042/`、批次报告 `output/20260517_1042.md`、结构化日志 `logs/20260517_1042.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`needs_aigc=7`、`skipped=4`、`failed=0`。
- Markdown 报告已包含自动评价、逐图状态、mask 面积、人脸重叠像素。
- 人脸保护路由生效，高风险图片进入 `needs_aigc`，没有继续自动修复人物主体。

### 下一轮方向

- 检查 `needs_aigc` 是否过多，判断 Haar 人脸保护是否过于保守。
- 检查 `skipped` 图片是否存在 OCR 漏检，特别是边缘竖排文字。
- 抽看 cleaned 图片的 debug，对比 OpenCV 修复是否有明显涂抹或残留。

## 20260517_1102

### 本轮成果

- 按 `docs/Workflow.md` 的默认快速迭代命令完成一轮。
- 生成独立输出目录 `output/20260517_1102/`、批次报告 `output/20260517_1102.md`、结构化日志 `logs/20260517_1102.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`needs_aigc=7`、`skipped=4`、`failed=0`。
- 本轮安全性符合预期：大量靠近人物主体的 mask 被路由到 `needs_aigc`，没有直接自动修坏人物。
- 抽看 debug 后确认：
  - 样例 A：mask 靠近人物主体，进入 `needs_aigc` 合理。
  - 样例 B：仍有右侧竖排文字残留，说明贴边文字检测不够。
  - 样例 C：有明显海报文字但被跳过，属于 OCR 漏检。

### 下一轮方向

- 为 `skipped` 和残留场景增加二次检测策略，优先做边缘竖排文字检测。
- 为大标题和海报文字增加非 OCR 的版面检测。
- 引入更可靠的人体 / 人脸 / 皮肤分割模型，逐步替代 Haar 保护。
- 为报告增加自动质检字段，记录“本轮通过 / 自动复检失败 / 需要规则调整”。
- 将稳定的命令参数固化为预设配置文件。

## 20260517_1116

### 本轮成果

- 开启 `--edge-text` 跑通一轮边缘文字二次 OCR。
- 生成独立输出目录 `output/20260517_1116/`、批次报告 `output/20260517_1116.md`、结构化日志 `logs/20260517_1116.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`needs_aigc=9`、`skipped=2`、`failed=0`。
- 边缘 OCR 新增 33 个文字候选，`skipped` 从上一轮 4 张降到 2 张。
- 样例 A：贴边竖排文字被补检出来，mask 明显覆盖到目标文字。
- 样例 B：海报边缘大字从漏检变为命中，并因靠近人物主体进入 `needs_aigc`，符合安全优先策略。
- 样例 C：仍然没有检出文字，说明纯边缘 OCR 还不能覆盖所有竖排 / 低对比文字。

### 下一轮方向

- 把边缘 OCR 保持为可选参数，继续观察误检率，不直接改成默认开启。
- 增加“竖排细长连通域 / 高对比笔画”的非 OCR 候选检测，用于补样例 C 这类漏检。
- 报告需要支持匿名模式，避免测试文件名出现在可分享报告中。
- 继续评估 `needs_aigc` 增多是否合理，重点看新增 edge mask 是否真的贴合文字。

## 20260517_1138

### 本轮成果

- 按工作流规范推进自动化路线，新增匿名报告参数 `--anonymous-report`。
- 使用 `LaMa ONNX + --protected-action repair + --edge-text` 跑通一轮保护区自动修复。
- 生成独立输出目录 `output/20260517_1138/`、匿名批次报告 `output/20260517_1138.md`、结构化日志 `logs/20260517_1138.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`cleaned_protected=9`、`skipped=2`、`failed=0`。
- 样例 A：大面积文字可被 LaMa 自动处理，但仍能看到残留 / 虚影，不能直接认定为通过。
- 样例 B：贴边文字修复效果好于 OpenCV，但固定 `512x512` 输入会限制细节质量。
- 样例 C：仍然漏检，说明边缘 OCR 无法覆盖所有低对比 / 竖排文字。

### 下一轮方向

- 增加修复后自动复检字段，不依赖人工逐图判断。
- 先用修复后 OCR 检测残留文字，验证自动质检链路。
- 如果修复后 OCR 不敏感，再加入 mask 区域纹理 / 边缘残留分数。

## 20260517_1146

### 本轮成果

- 新增 `--post-ocr-check`，修复后会复跑 OCR，并在报告里输出 `Residual Text`。
- 使用 `LaMa ONNX + protected repair + edge text + post OCR` 跑通一轮自动质检。
- 生成独立输出目录 `output/20260517_1146/`、匿名批次报告 `output/20260517_1146.md`、结构化日志 `logs/20260517_1146.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`cleaned_protected=9`、`skipped=2`、`quality_failed=0`、`failed=0`。
- 本轮确认：修复后 OCR 字段链路有效，但对残留虚影不敏感，所有已修复图片的 `Residual Text` 都是 0。

### 下一轮方向

- 增加 mask 区域局部质量评分，检测残留边缘、异常高频纹理和明显色块。
- 把自动质检结果拆成 `ocr_residual`、`texture_residual`、`large_mask_risk` 等字段。
- 对 `cleaned_protected` 默认保持谨慎，不因 post OCR 为 0 就判定质量通过。
- 继续补竖排 / 低对比文字检测，减少 `skipped`。

## 20260517_1235

### 本轮成果

- 引入 Florence-2-base 作为可选视觉文字检测层，参数为 `--vision-text`。
- 下载模型到 `models/microsoft/Florence-2-base`，并补齐 `torch`、`transformers`、`timm`、`einops` 等依赖。
- 修复 Florence-2 在当前环境下的兼容问题：
  - `transformers` 固定为 `>=4.45,<5`。
  - `numpy` 固定为 `<2.4`，`PyYAML` 固定为 `6.0.2`，避免破坏 PaddleOCR / PaddleX。
  - Florence 生成关闭 cache，并使用 eager attention。
  - 视觉检测前按最长边 `768` 缩放，避免 CPU 全量推理过慢。
- 使用 `PaddleOCR + edge OCR + Florence-2 + OpenCV` 跑通一轮。
- 生成独立输出目录 `output/20260517_1235/`、匿名批次报告 `output/20260517_1235.md`、结构化日志 `logs/20260517_1235.jsonl`。
- 处理 14 张样例图：`cleaned=4`、`needs_aigc=10`、`skipped=0`、`failed=0`。
- Florence-2 新增 28 个视觉文字候选，成功把上一轮 `skipped` 降到 0。
- 代价是 mask 面积和保护区重叠增加，说明全量启用视觉检测会提高召回，但也会放大误检 / 高风险路由。

### 下一轮方向

- Florence-2 不应默认全量启用，改为视觉兜底策略：
  - OCR + edge OCR 无结果时启用。
  - 或候选数量低、但图片疑似海报 / 贴边文字时启用。
- 为视觉候选增加过滤规则，避免大框、人物区域、背景纹理导致 mask 过大。
- 报告继续保留 `Vision Text`，用于比较 OCR-only 与 vision fallback 的差异。
- 后续再评估 SAM / SAM2，用于把 Florence 候选框精修为更贴合的 mask。

## 20260517_1305

### 本轮成果

- 将 Florence-2 从全量检测改为可配置触发策略。
- 新增参数：
  - `--vision-trigger always`
  - `--vision-trigger empty`
  - `--vision-trigger low-count`
  - `--vision-low-count`
- 报告新增 `Vision` 字段，记录本图是否实际触发视觉检测。
- 使用 `--vision-trigger empty` 跑通一轮视觉兜底。
- 生成独立输出目录 `output/20260517_1305/`、匿名批次报告 `output/20260517_1305.md`、结构化日志 `logs/20260517_1305.jsonl`。
- 处理 14 张样例图：`cleaned=4`、`needs_aigc=10`、`skipped=0`、`failed=0`。
- Florence-2 只在 2 张 OCR + edge OCR 空结果图片上运行，新增 2 个视觉文字候选。
- 与全量 Florence 相比，兜底策略保留了 `skipped=0` 的召回收益，同时避免 28 个视觉候选带来的 mask 扩张风险。

### 下一轮方向

- 将 `--vision-trigger empty` 作为推荐视觉策略。
- 为 `low-count` 策略跑一轮，对比是否能进一步补低召回图片，同时不显著增加 mask 面积。
- 增加视觉候选过滤规则：过大框、靠近人物主体、长宽比异常的候选先降权或丢弃。
- 开始设计 mask 区域局部质量评分，用于识别 LaMa / OpenCV 修复后的涂抹和虚影。

## 20260517_1322

### 本轮成果

- 针对观察者指出的 debug 问题进行复查。
- 确认上一轮样例 A 在 `output/20260517_1305/` 中状态为 `needs_aigc`，没有进入修复阶段，因此 debug 第三栏显示原图，不代表修复失败。
- 修正 debug 面板：`skipped` / routed 图片会在第三栏标注 `SKIPPED` 或 `NOT REPAIRED`，避免把未修复图误看成修复结果。
- 使用 `LaMa ONNX + --protected-action repair + vision fallback + edge text` 跑通一轮真正自动修复。
- 生成独立输出目录 `output/20260517_1322/`、匿名批次报告 `output/20260517_1322.md`、结构化日志 `logs/20260517_1322.jsonl`。
- 处理 14 张样例图：`cleaned=4`、`cleaned_protected=10`、`skipped=0`、`failed=0`。
- 样例 A 这轮确实输出了修复图，但仍有明显竖排文字残留，说明问题是 mask 召回不足，不是 LaMa 没运行。

### 下一轮方向

- 增加竖排文字补 mask 策略，优先解决大竖排文字残留。
- 注意竖排补 mask 不能误伤头发、人物轮廓和皮肤阴影。

## 20260517_1331

### 本轮成果

- 新增实验参数 `--vertical-text`，尝试通过深色竖排组件补充 OCR 漏掉的大字。
- 生成独立输出目录 `output/20260517_1331/`、匿名批次报告 `output/20260517_1331.md`、结构化日志 `logs/20260517_1331.jsonl`。
- 处理 14 张样例图：`cleaned=1`、`cleaned_protected=13`、`failed=0`。
- 本轮竖排检测新增 188 个组件候选，能补到部分残留文字，但明显误伤头发 / 人物边缘，导致保护区重叠暴涨。

### 下一轮方向

- 竖排检测不能全局独立运行，必须锚定已有 OCR / edge / vision 文本附近，只做补洞。
- 收紧组件候选，降低误伤人物主体。

## 20260517_1337

### 本轮成果

- 将竖排检测改成“锚定补洞”：只有靠近已有文本 mask 的深色组件才会加入。
- 生成独立输出目录 `output/20260517_1337/`、匿名批次报告 `output/20260517_1337.md`、结构化日志 `logs/20260517_1337.jsonl`。
- 处理 14 张样例图：`cleaned=2`、`cleaned_protected=12`、`failed=0`。
- 竖排候选从 188 降到 112，但仍然偏激进；样例 A 的目标文字补得有限，同时仍误伤人物区域。

### 下一轮方向

- `--vertical-text` 保持实验功能，不能默认开启。
- 下一步应改为更智能的版面级文字检测，而不是简单深色组件启发式。
- 对样例 A 这类大竖排文字，优先尝试 Florence / 版面模型给出大框，再用分割模型或结构规则精修 mask。

## 20260517_1404

### 本轮成果

- 新增实验参数 `--vertical-columns`，尝试用列级投影检测竖排文字。
- 生成独立输出目录 `output/20260517_1404/`、匿名批次报告 `output/20260517_1404.md`、结构化日志 `logs/20260517_1404.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`cleaned_protected=11`、`failed=0`。
- 列级检测新增 9 个候选，比组件检测更少，但样例 A 仍未覆盖左侧大竖排文字。

### 下一轮方向

- 继续排查列级投影为何漏掉左侧低对比 / 抗锯齿文字。
- 如果仍需要大矩形才能覆盖文字，则停止该启发式路线，转向模型级版面检测。

## 20260517_1416

### 本轮成果

- 调整列级投影：加入自适应阈值、形态学闭合和邻近列合并。
- 生成独立输出目录 `output/20260517_1416/`、匿名批次报告 `output/20260517_1416.md`、结构化日志 `logs/20260517_1416.jsonl`。
- 处理 14 张样例图：`cleaned=4`、`cleaned_protected=10`、`failed=0`。
- 单独探针确认：当前投影法会抓到中间文字列，但仍漏掉左侧大竖排，并且容易生成大矩形误伤。

### 下一轮方向

- 停止继续调 `--vertical-text` / `--vertical-columns` 启发式，不作为默认路线。
- 对样例 A 这类大竖排文字，下一步改用模型级检测：
  - Florence 全量或局部裁切重试。
  - 或引入专门文本检测模型，输出更可靠的文字框。
- 修复质量不能只看是否生成输出，必须检查目标文字是否完整进入 mask。

## 20260517_1529

### 本轮成果

- 使用 `Florence-2 全量检测 + LaMa ONNX + protected repair` 跑通一轮。
- 生成独立输出目录 `output/20260517_1529/`、匿名批次报告 `output/20260517_1529.md`、结构化日志 `logs/20260517_1529.jsonl`。
- 处理 14 张样例图：`cleaned=3`、`cleaned_protected=10`、`skipped=1`、`failed=0`。
- 本轮不算通过：全量 Florence 仍然成本高，而且仍有 1 张样例被跳过。
- 样例 A 检出候选增加，但视觉检查显示目标竖排文字仍不能稳定完整清除。

### 下一轮方向

- 不继续把 Florence 全量检测作为默认路线。
- 改为局部裁切视觉检测，优先检测左右边缘大竖排文字。
- 报告中需要区分整图视觉候选和边缘裁切视觉候选。

## 20260517_1539

### 本轮成果

- 新增 `--vision-edge-crops`，对图片左右边缘裁切后运行 Florence-2，再映射回原图。
- 新增报告字段 `Vision Edge`，记录边缘裁切视觉检测新增候选数。
- 针对样例 A 做定向验证：整图视觉未触发，边缘裁切视觉新增 4 个候选。
- mask 总面积从上一轮同样例的 `74072` 增加到 `80073`，新增区域主要集中在左侧文字区域。
- 视觉检查确认：左侧大竖排文字的主体已进入 mask，检测路线比启发式竖排检测可靠。
- 但结果仍未完全通过：顶部附近仍有小段残留，说明后续问题转向 mask 边缘精修和修复质量。

### 下一轮方向

- 将 `--vision-edge-crops` 作为贴边竖排文字的推荐实验路线。
- 不用简单加大 `--dilate` 硬扩 mask，避免把残留变成黑色碎片或块状修复。
- 下一步引入更贴合文字轮廓的 mask 精修，或接入专门文本检测 / 分割模型。

## 20260517_1540

### 本轮成果

- 针对样例 A 尝试 `--dilate 14` 和 `--vision-shrink-ratio 0`。
- 自动质检结果为 `quality_failed`，修复后 OCR 检出 1 个残留候选。
- 视觉检查确认：残留从淡色虚影变成黑色碎片，说明单纯扩大 mask 不是正确方向。

### 下一轮方向

- 回退大膨胀策略。
- 优先做 mask 精修和修复策略分流，而不是继续盲目扩大 mask。

## 20260517_1542

### 本轮成果

- 针对样例 A 使用 OpenCV Telea 做后端对照。
- 自动质检结果为 `quality_failed`，修复后 OCR 仍检出 1 个残留候选。
- 视觉检查确认：OpenCV 在该场景出现明显块状重建，比 LaMa ONNX 更差。

### 下一轮方向

- 样例 A 继续使用 LaMa ONNX 作为较优基线。
- 检测侧保留边缘裁切视觉路线；修复侧需要更强的模型或更精细的 mask。

## 20260517_1601

### 本轮成果

- 针对观察者指出的两处残留进行定向验证。
- 开启 `--sticker-watermarks` 后，样例 A 的底部粉色贴纸被更完整纳入 mask。
- 本轮 `watermark=5`，`sticker_mask_pixels=16916`，说明彩色贴纸检测比只依赖文字框更适合处理圆形贴纸水印。
- 视觉检查确认：底部贴纸残留明显改善。
- 上方竖排文字附近仍有小段残留，说明该问题不是贴纸检测问题，而是黑色文字笔画 / 修复残影问题。

### 下一轮方向

- 将 `--sticker-watermarks` 作为贴纸水印场景的推荐实验参数。
- 继续解决上方黑色文字残留，避免只扩大 mask 导致块状修复。

## 20260517_1609

### 本轮成果

- 新增 `--dark-stroke-refine`，尝试在已有文本 mask 附近补黑色文字笔画。
- 样例 A 的上方残留区域被进一步纳入 mask。
- 但该策略过激，错误吸收了边缘皮肤 / 手部附近的深色纹理，`face_overlap_pixels` 上升。

### 下一轮方向

- `--dark-stroke-refine` 不能作为默认功能。
- 收紧为低饱和黑色笔画，只作为边缘文字残留的实验补 mask。

## 20260517_1611

### 本轮成果

- 收紧 `--dark-stroke-refine` 的颜色条件，降低彩色区域误检。
- 样例 A 的底部贴纸仍保持较好处理。
- 上方黑字残留仍没有达到可接受质量，且 dark stroke 仍会增加边缘误伤风险。

### 下一轮方向

- 暂停继续依赖简单黑色笔画启发式。
- 样例 A 的上方残留需要更可靠的文本分割 / mask 精修模型，而不是继续手写阈值。
- 下一步优先评估专用文字检测 / 分割模型，或把 LaMa mask 输入改为更贴合文字轮廓的局部策略。

## 20260517_1624

### 本轮成果

- 针对观察者指出的“黑点残留 + 手部涂抹”进行复查。
- 确认 `20260517_1611` 的手部涂抹来自 `--dark-stroke-refine` 误伤，而不是 LaMa 本身。
- 新增自动护栏：先计算基础 mask，再计算加入 dark stroke 后的候选 mask；如果候选 mask 让保护区重叠明显增加，则自动丢弃 dark stroke refinement。
- 本轮 `dark_stroke_discarded=true`，`Dark Stroke=0`，mask 面积和保护区重叠回到 `20260517_1601` 水平。
- 结果：腿部和底部贴纸保留较好处理，手部涂抹风险被撤掉。
- 仍有上方黑点残留，说明该问题不能靠当前 dark stroke 启发式解决。

### 下一轮方向

- 当前推荐基线为：`edge OCR + vision edge crops + sticker watermarks + LaMa ONNX`，不开 `dark-stroke-refine`，或仅在护栏允许时使用。
- 黑点残留进入下一阶段：评估专用文本分割 / mask 精修模型，或改造局部修复策略。

## 20260517_watermark_preset

### 本轮成果

- 根据观察者建议，调整项目流水线目标：先完整清理水印 / 文字，再解决照片涂抹感。
- 新增 `--preset watermark-first`，固化当前水印优先组合：
  - `--edge-text`
  - `--sticker-watermarks`
  - `--vision-text`
  - `--vision-trigger empty`
  - `--vision-edge-crops`
  - `--protected-action repair`
  - `--post-ocr-check`
  - 本地存在 LaMa ONNX 时自动使用 `lama-onnx`
- 该预设明确不启用 `--dark-stroke-refine`、`--vertical-text`、`--vertical-columns`。
- 定向验证结果：`Watermark=5`、`Vision Edge=4`、`Dark Stroke=0`，复现 `20260517_1601 / 20260517_1624` 的水印优先基线。

### 下一轮方向

- 第一阶段继续围绕“完全清理水印 / 文字”迭代 mask 覆盖。
- 涂抹感、手部结构、照片自然度进入第二阶段，由 AI 修复模型解决，不再和水印召回混在同一轮。

## 20260517_post_context_fill（定向探针，非正式时间戳轮次）

### 本轮成果

- 说明：该目录由固定 `--run-id` 生成，用于对同一张样例做定向复现，不符合正式 `YYYYMMDD_HHMM` 轮次命名。后续正式迭代必须回归 `--auto-run-dir` 自动时间戳。
- 针对 `output/20260517_watermark_preset/` 中观察到的上方竖排黑字残留继续迭代。
- 新增 `--edge-column-refine`，作为锚定边缘竖排补齐策略：
  - 只在已有文字 mask 附近搜索低饱和暗色小组件。
  - 只处理左右边缘带。
  - 先计算保护区重叠增量，风险扩大时自动丢弃。
- 将 `--edge-column-refine` 加入 `--preset watermark-first`，默认参数提升到 `edge_ratio=0.5`、`anchor_radius=32`。
- 新增修复后暗残留检测，命中后尝试 `context-fill + OpenCV Telea` 局部二次清理。
- 定向验证 `output/20260517_post_context_fill/`：
  - `Edge Column=6`，说明上方竖排漏检笔画被部分补入 mask。
  - `post_dark_residual_count=1`，二次残留清理链路已触发。
  - `Residual Text=0`，但视觉检查仍可见上方孤立黑色残影。

### 下一轮方向

- 本轮仍未完全通过；不能把 `Residual Text=0` 当成视觉通过。
- 上方孤立残影需要更可靠的局部文字分割 / 修复方式，当前组件级启发式只能部分改善。
- 下一步优先尝试专用文本检测 / 分割模型，或对边缘竖排文字区域做局部裁切的更高分辨率 mask 精修。

## 20260517_2343

### 本轮成果

- 回归 `docs/Workflow.md` 标准工作流：正式批次不再传固定 `--run-id`，由 `--auto-run-dir` 生成时间戳目录。
- 将 `watermark-first` 中的边缘 Florence 从全量 / low-count 收紧为 `--vision-edge-crop-trigger empty`，避免 CPU 全批次长时间卡住。
- 正式输出：`output/20260517_2343/`、`output/20260517_2343.md`、`logs/20260517_2343.jsonl`。
- 处理 14 张样例：`cleaned=2`、`cleaned_protected=11`、`skipped=1`、`failed=0`。
- 速度恢复到可完成的正式批次，但仍有 1 张明显右侧竖排文字被跳过。

### 下一轮方向

- 继续围绕 skipped 图补召回，不能因为 Florence 原始退化框存在就判定通过。
- 对 OCR / Florence 都返回空或退化框的图，增加更保守的边缘暗色竖排兜底。

## 20260517_2358

### 本轮成果

- 新增空结果边缘暗色竖排兜底：仅在 OCR、edge OCR、vision 都没有形成有效 mask 时触发。
- 正式输出：`output/20260517_2358/`、`output/20260517_2358.md`、`logs/20260517_2358.jsonl`。
- 处理 14 张样例：`cleaned=2`、`cleaned_protected=12`、`skipped=0`、`failed=0`。
- `sample-013` 从 skipped 变为 cleaned_protected，`empty_edge_count=1`、`empty_edge_mask_pixels=519`，说明兜底确实补到了右侧竖排文字的一部分。

### 下一轮方向

- 本轮不算视觉通过：`sample-013` 的右侧竖排文字仍有残留，当前兜底列 mask 太保守。
- 下一步应扩大空结果边缘竖排兜底的列合并范围或改用局部高分辨率 OCR / 文本分割，优先覆盖整列而不是孤立组件。
- `extrafanart-7` 仍有上方黑色残影，继续归入“需要更可靠局部文字分割 / mask 精修”的问题，不再用全局 dark stroke 硬扩。

## 20260518_0156

### 本轮成果

- 新增 5 张高难度样例，单独复制到 `.tmp_new_hard/` 并按标准时间戳工作流验证。
- 使用 `--preset watermark-first --device directml` 跑通小批次。
- 正式输出：`output/20260518_0156/`、`output/20260518_0156.md`、`logs/20260518_0156.jsonl`。
- 处理 5 张样例：`cleaned_protected=4`、`quality_failed=1`、`failed=0`、`skipped=0`。
- DirectML provider 可用，但当前 `Carve/LaMa-ONNX` 在 DirectML 上运行 Transpose 节点时触发 GPU device removed，已自动回退 `CPUExecutionProvider`。
- 新样例确认当前路线的上限：
  - 检测召回很强，mask 面积很大，能覆盖大量标题、贴纸和角标。
  - 但 LaMa 512 对大面积封面文字、文字压人体、复杂背景和底部锁图标修复不足。
  - 多张图仍有明显残字、虚影、块状重建或主体涂抹，不能认定第一阶段通过。

### 下一轮方向

- AMD 显卡路线不能继续押当前 LaMa ONNX；需要接入 DirectML 友好的局部重绘 / 高清修复后端。
- 下一阶段应把现有 mask 输出给 SD/Flux Inpaint 或 TensorRT/ONNX/DirectML 兼容的 AIGC 后端：
  - 小 mask / 简单背景继续 LaMa。
  - 大 mask / 覆盖人体 / 复杂封面文字进入 AIGC inpaint。
  - 去字后再考虑高清修复和 ADetailer。
- 第一阶段目标仍是“水印文字真正清干净”，当前新高难度样例未通过。

## 20260518_webui_api

### 本轮成果

- 按“直接上 inpaint”的方向新增 `--inpaint-backend webui-api`。
- 新后端调用 Automatic1111 / Forge WebUI 的 `/sdapi/v1/img2img` inpaint API：
  - 项目继续负责 OCR / 视觉检测 / mask 生成。
  - WebUI 负责基于 mask 做 AIGC 局部重绘。
  - AMD 显卡由 WebUI / Forge 的启动后端负责使用。
- 新增参数：`--webui-url`、`--webui-prompt`、`--webui-negative-prompt`、`--webui-steps`、`--webui-denoise`、`--webui-cfg-scale`、`--webui-sampler`、`--webui-mask-blur`、`--webui-timeout`。
- 本机探测 `http://127.0.0.1:7860/sdapi/v1/options` 当前连接失败，说明 WebUI API 尚未启动，因此本轮只完成代码接入，尚未跑真实 AIGC inpaint 输出。

### 下一轮方向

- 启动 WebUI / Forge，并确保开启 `--api`。
- 用新高难度 5 张样例跑 `--inpaint-backend webui-api` 标准时间戳小批次。
- 对比 LaMa 轮次 `20260518_0156`，重点看大面积文字压人、底部锁图标、彩色标题和白色低对比文字是否明显改善。

## 20260518_0247

### 本轮成果

- 按 `docs/Workflow.md` 回归标准时间戳工作流，重建 `.tmp_new_hard/` 小批次并处理 5 张高难度样例。
- 正式输出：`output/20260518_0247/`、`output/20260518_0247.md`、`logs/20260518_0247.jsonl`。
- 处理 5 张样例：`cleaned_protected=4`、`quality_failed=1`、`failed=0`、`skipped=0`。
- DirectML 再次在 `Carve/LaMa-ONNX` 的 Transpose 节点触发 GPU device removed，代码自动回退 `CPUExecutionProvider`，因此本轮不属于真正 AMD AIGC inpaint 输出。

### 视觉 Review

- `三好佑香-794`：右侧/中部仍有明显竖排字残留，底部小字区和解锁按钮残留严重；自动后 OCR 仍检出 2 个残留候选，第一步未通过。
- `三好佑香-824`：大字和底部条幅被大面积抹除，但右侧文字区域变成明显白色/灰色块，边缘局部仍像残字；不应视作干净通过。
- `三好佑香-833`：主标题大体清掉，但左下白色大字仍有可见笔画/阴影残留，人物与背景被明显涂抹；文字第一步仍未稳定通过。
- `三好佑香-853`：左侧大标题清掉一部分，但右侧竖排小字和底部字仍有残留，左侧生成块感很强。
- `三好佑香-858`：上方竖排小字残留明显，底部大字和解锁按钮仍基本可见；该图说明当前 mask/修复组合对低对比叠字仍不够。

### 结论

- 当前检测召回已经能抓到大量文字、水印、贴纸：本轮总计 `Text=74`、`Edge Text=26`、`Watermark=109`、`Edge Column=99`。
- 失败瓶颈不再只是 OCR 漏检，而是“mask 后的修复后端不够强”：LaMa 512 对封面级大字、半透明字、文字压人体、底部锁图标无法自然重绘。
- 第一阶段“所有水印文字移除”仍未通过。下一步必须让 WebUI/Forge 或 ComfyUI API 真实可用，走 AIGC inpaint；继续扩大 LaMa mask 只会增加块状涂抹和主体损伤。

## 20260518_2256

### 本轮成果

- 新增 `--inpaint-backend comfyui-api`，项目侧自动上传 RGBA 输入图到 ComfyUI，再提交 API-format workflow，轮询 history 并下载输出图。
- `.env` 支持 `COMFYUI_URL`、`COMFYUI_WORKFLOW`、`COMFYUI_TIMEOUT`、`COMFYUI_POLL_INTERVAL`，命令行参数优先级高于 `.env`。
- 新增 `workflows/sd1.5_inpaint_api.json`，对应现有 `workflows/sd1.5_inpaint.json` 的 API 调用版本。
- 云端 ComfyUI 探测正常：`https://comfyui.wodcloud.com/shucheng/system_stats` 返回 RTX 4090 CUDA 设备。
- 正式探针输入目录回归 `.tmp/comfy_probe/`，避免在仓库根目录散落临时目录。
- 修正 ComfyUI alpha mask 方向：`LoadImage` 的 alpha mask 工作流需要“待重绘区域透明”，因此项目侧现在将水印 mask 区域写为 alpha=0，其他区域 alpha=255。
- 探针输出：`output/20260518_2256/`、`output/20260518_2256.md`、`logs/20260518_2256.jsonl`。

### 视觉 Review

- ComfyUI API 链路已跑通，`Route` 为 `comfyui-api:https://comfyui.wodcloud.com/shucheng:...`，说明上传、提交、轮询、下载四步均可用。
- 修正 alpha 后，输出不再全图重绘，已经进入局部 inpaint 路线。
- 当前 `512-inpainting-ema.safetensors` + SD1.5 workflow 质量仍不合格：画面明显蓝偏，底部大字和右侧竖排字仍有残留，自动后 OCR 仍检出 1 个残留候选。

### 下一轮方向

- 接入层已完成，下一步重点不再是 API，而是 workflow 质量：
  - 换更适合真人照片的 inpaint checkpoint。
  - 降低 denoise 或改 sampler/scheduler，避免整图色彩漂移。
  - 评估是否改成“原图和 mask 分开上传”的 workflow，便于控制 mask 预处理。
  - 对大 mask 图尝试局部裁切 inpaint，而不是整图 512 inpaint。

## 20260518_2322

### 本轮成果

- 根据观察者反馈，停止追外部 Moody/ZImage 大工作流，回到项目本体问题：
  - 文字水印没有全部标注出来。
  - ComfyUI 结果人物/画面发蓝。
- 修正 ComfyUI 后端合成策略：下载的 ComfyUI 输出不再整图替换，而是只按最终 mask 合回原图，避免未 mask 区域被模型色偏污染。
- 新增亮色文字补 mask：在已有文字 mask 附近，补边缘和下半区的白色/浅色文字笔画，用于底部大白字、半透明字和低对比白字。
- 探针输出：`output/20260518_2322/`、`output/20260518_2322.md`、`logs/20260518_2322.jsonl`。

### 结果

- 自动状态从上一轮 `quality_failed` 变为 `cleaned_protected`，post OCR 残留从 `1` 降到 `0`。
- 亮色补 mask 生效：`bright_text_count=32`、`bright_text_mask_pixels=6854`。
- 全图蓝偏被控制住，未 mask 的脸部、肤色、背景基本保留原图颜色。

### 仍未通过

- 视觉上仍不能算最终通过：底部大字区域出现明显块状重建，右侧竖排小字仍有残留。
- 当前 ComfyUI workflow 仍是整图 inpaint 后局部合成，局部纹理和边缘一致性不足。

### 下一轮方向

- 继续扩大和精修白色/半透明文字 mask，尤其是右侧竖排小字和底部大字尾部。
- 引入局部裁切 inpaint：只把 mask 附近 crop 发给 ComfyUI，再 stitch 回原图，减少大块色彩漂移和结构生成。
- 保持 ComfyUI 结果按 mask 合成，不允许整图替换。
