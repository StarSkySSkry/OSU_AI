# AutoPlayOSU! 修改紀錄 (CHANGELOG)

> 此檔案記錄所有對專案的修改，方便追蹤和除錯。

---

## 2025-02-17

### [v0.4.0] dxcam 截圖引擎（取代 mss）

**修改檔案**: `ai/eval.py`
**新增依賴**: `pip install dxcam`

**問題**: `mss.grab()` 被 Windows DWM vsync 限制在 ~33 FPS，這是推理管線的主要瓶頸。

**修改內容**:
1. **`mss` → `dxcam`** — 使用 DirectX Desktop Duplication API，可達 60-240+ FPS
2. **`camera.start(target_fps=0)`** — 持續截圖模式，無 FPS 上限
3. **`get_latest_frame()`** — 每次都取最新幀，跳過舊幀（減少延遲）
4. **`output_color="GRAY"`** — dxcam 直接輸出灰度，省去 `cv2.cvtColor` 轉換
5. **`try/finally`** — 確保 `camera.stop()` 被調用，避免資源洩漏

---

### [v0.3.1] 進一步延遲優化 + FPS 計數器

**修改檔案**: `ai/eval.py`

**問題**: 大幅改善後仍有「慢半拍」的延遲感。

**修改內容**:
1. **Pinned Memory + `non_blocking=True`** — 預分配 pinned memory tensor，CPU→GPU 傳輸改為非同步，GPU 可以在傳輸進行時開始其他工作。
2. **專用 CUDA Stream** — 推理在獨立的 `torch.cuda.Stream()` 上執行，避免與其他 GPU 操作互相阻塞。
3. **FPS 計數器** — 每 2 秒印出 `[Eval] FPS: xx.x`，方便診斷實際吞吐量。

**備註**: 模型訓練時的 `lookahead=3`（`dataset.py` 第 157 行）僅補償 3 幀的延遲。若推理管線總延遲超過 3 幀時間，模型的預測仍會落後。這需要重新訓練時增加 lookahead 值才能根本解決。

---

### [v0.3.0] 推理延遲優化

**修改檔案**: `ai/eval.py`

**問題**: 推理迴圈有明顯的延遲和卡頓。

**修改內容**:
1. **移除 `FixedRuntime` 強制 10ms 等待** — 每幀原本被強制等待至少 10ms，這對即時遊戲來說是純浪費。改為推理迴圈全速運行，未啟用時用 `time.sleep(0.001)` 低功耗等待。
2. **`clone()` → `torch.roll()`** — 緩衝區滾動改用 `torch.roll()`，直接在 GPU kernel 層級操作，比 `slice + clone` 更快。
3. **`COLOR_RGB2GRAY` → `COLOR_BGRA2GRAY`** — `mss` 截圖輸出的是 BGRA 格式，之前誤用 RGB2GRAY 會多一次不必要的轉換。
4. **`output.cpu()` 移出 `no_grad` 區塊** — 加上 `.detach()` 確保不追蹤梯度圖。

---

### [v0.2.1] 移除 PID 控制器

**修改檔案**: `ai/eval.py`

**問題**: PID 控制器導致滑鼠在兩個點之間來回震盪，整體移動緩慢。

**修改內容**:
1. **移除 `PidMixin` 類和所有 PID 相關代碼**
2. **改用 `ctypes.windll.user32.SetCursorPos()` 直接設定絕對座標** — 比 `mouse.move()` 更快更精準
3. `AimThread` 和 `CombinedThread` 不再繼承 `PidMixin`，簡化為直接調用 `_move_mouse_absolute()`

---

### [v0.2.0] GPU 推理 + FP16 + 預分配緩衝區

**修改檔案**: `ai/eval.py`, `ai/constants.py`

**問題**: 推理使用 CPU，速度受限。

**修改內容**:
1. **GPU 推理** — 模型載入到 `PYTORCH_DEVICE`（有 CUDA 則用 GPU）
2. **FP16 半精度** — CUDA 上自動啟用 `model.half()`，推理速度約 2x
3. **預分配 tensor 緩衝區** — 用固定大小的 `torch.Tensor` 取代 `deque + np.stack`，避免每幀重新分配記憶體
4. **CUDA warmup** — 首次推理前用 dummy tensor 預熱 CUDA kernel
5. **`PidMixin` 消除重複代碼** — `AimThread`/`CombinedThread` 的 PID 邏輯提取為共用 Mixin（後在 v0.2.1 移除）
6. **`constants.py`**: 新增 `USE_FP16 = torch.cuda.is_available()`

---

### [v0.1.1] 修復模型載入錯誤

**修改檔案**: `ai/eval.py`

**Bug 1**: `model.pth` → `model.pt`
- `models.py` 中 `save()` 用 `torch.jit.script` 保存為 `model.pt`
- 但 `eval.py` 中 `run()` 嘗試載入 `model.pth`
- 修正：第 74 行 `'model.pth'` → `'model.pt'`

**Bug 2**: `torch.load` → `torch.jit.load`
- PyTorch 2.6 將 `torch.load` 的 `weights_only` 預設改為 `True`
- TorchScript 模型不支援 `weights_only=True`
- 修正：第 82 行 `torch.load(...)` → `torch.jit.load(...)`

---

### [v0.1.0] 初始備份

**操作**: 將專案推送到 GitHub (`https://github.com/SkySSkry/OSU_AI`)

**包含修改**:
- `ai/eval.py` (已修改)
- `commit_message.txt` (新增)
- `conversation_history.md` (新增)
