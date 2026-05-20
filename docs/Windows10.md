# Windows 10 + AMD 7900 XT 后端判断

## 当前困境

本项目当前的核心问题不是“能不能检测到文字”，而是高难度图片在生成 mask 后，需要一个足够强的局部重绘后端。现有 `LaMa ONNX 512` 对大面积封面文字、半透明标题、文字压人体、底部解锁按钮等场景修复不足，会出现残字、虚影、块状重建和主体涂抹。

在本机 Windows 环境中，`onnxruntime-directml` 能看到 `DmlExecutionProvider`，说明 DirectML 路线本身可用。但当前 `models/Carve/LaMa-ONNX/lama.onnx` 在 DirectML 上运行时会在 Transpose 节点触发 `GPU device removed`，项目会自动回退到 `CPUExecutionProvider`。因此当前报告中看到的 `--device directml` 并不等于真的使用 AMD GPU 完成修复，必须以 report 里的 `Route` 字段为准。

## Windows 10 的限制

Windows 10 不是完全不能做 AMD GPU inpaint，但每条路线的确定性不同：

| 路线 | Windows 10 状态 | RX 7900 XT 判断 | 备注 |
| :-- | :-- | :-- | :-- |
| DirectML / ONNX Runtime | 可用 | 取决于模型算子兼容性 | 当前 LaMa ONNX 已失败并回退 CPU |
| 原生 Windows PyTorch ROCm | 不作为主线 | 官方 Windows 矩阵更偏 Windows 11 和限定型号 | RX 7900 XT 在当前原生 Windows 矩阵里不清晰 |
| WSL2 + ROCm | Windows 10 21H2+ 可作为宿主 | AMD WSL 矩阵明确列 RX 7900 XT | 当前最值得投入的 AMD GPU 路线 |
| WebUI / Forge / ComfyUI API | 可用，取决于后端启动方式 | 可走 DirectML、ROCm/WSL 或其他后端 | 项目侧通过 API 提交 image + mask |
| CLI worker | 可用 | 可封装 WSL2 ROCm、DirectML、IOPaint 等 | 最适合项目长期抽象 |

参考：

- AMD Windows ROCm / PyTorch compatibility: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
- AMD WSL ROCm compatibility: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html
- Microsoft WSL GPU compute: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute

## 是否必须升级 Windows 11

升级 Windows 11 是一条有价值的路，但不是唯一前提。

升级的好处：

- 更贴近 AMD 当前原生 Windows PyTorch ROCm 的官方方向。
- 后续使用 WebUI / ComfyUI / AMD AI 工具链时，系统环境更少遇到版本边界。
- WSL2、GPU 驱动、图形和 AI 工具链整体更现代。

但对 `RX 7900 XT` 来说，最明确的官方支持点仍然是 `WSL2 + ROCm`，而不是原生 Windows PyTorch ROCm。因此即使升级到 Windows 11，项目后端主线也仍建议优先设计成可调用 WSL2/外部进程的形式。

## 下一步后端策略

项目不应该绑定死某一种 UI 或服务。推荐新增一个通用高级修复层，把当前 OCR / mask / report 流水线和实际 inpaint 执行解耦。

优先级建议：

1. 保留现有 `opencv` / `lama-onnx` 作为小 mask、简单背景的快速后端。
2. 新增通用 `cli` 后端，用命令行 worker 处理 image + mask + output。
3. `cli` 后端第一目标是 WSL2 ROCm worker，让 RX 7900 XT 真正参与 AIGC inpaint。
4. API 后端也可以继续推进，例如 WebUI / Forge / ComfyUI。API 不是坏路线，它能更快验证模型质量，只是运行形态和自动化部署不同。

推荐抽象形态：

```powershell
.\.venv\Scripts\python.exe main.py `
  --preset watermark-first `
  --input .\.tmp_new_hard `
  --output .\output `
  --auto-run-dir `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --inpaint-backend cli `
  --cli-inpaint-command "wsl python /home/mengk/image-clean-workers/inpaint_worker.py --image {image} --mask {mask} --output {output}"
```

API 验证形态：

```powershell
.\.venv\Scripts\python.exe main.py `
  --preset watermark-first `
  --input .\.tmp_new_hard `
  --output .\output `
  --auto-run-dir `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --inpaint-backend webui-api `
  --webui-url http://127.0.0.1:7860
```

## 当前结论

- Windows 10 的主要困境是：原生 AMD AI 生态不如 Windows 11 清晰，而当前 DirectML + LaMa ONNX 已经实际失败。
- 继续坚持 AMD GPU 是合理的，但不应继续死磕当前 LaMa ONNX。
- 最稳的工程路线是：Windows 主项目继续负责检测、mask、报告；高级修复交给 `WSL2 ROCm CLI worker` 或 `WebUI/ComfyUI API`。
- 下一步可以先做 API 路线验证修复质量，同时并行设计 `cli` 后端接口。只要接口统一，后面从 API 切到 WSL2 CLI 不会推倒重来。
