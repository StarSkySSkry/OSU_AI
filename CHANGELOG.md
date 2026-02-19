# AutoPlayOSU! 修改紀錄 (CHANGELOG)

> 此檔案記錄所有對專案的修改，方便追蹤和除錯。

---

## 2025-02-19 ~ 02-20

### [v0.8.0] 雙模型推理 + 分模型 Lookahead

**修改檔案**: `ai/eval.py`, `ai/play.py`, `ai/dataset.py`

**問題**: Combined 模型用 MSE Loss 同時學座標和按鍵，座標損失主導梯度 → **模型幾乎不學按鍵**（2% 準確率、1050 miss）。Dual Mode（兩線程各自截圖）搶 mss 資源 → FPS 29→19。

**修改內容**:

1. **分模型 Lookahead** (`dataset.py`) — 不同模型類型使用不同 lookahead：
   - Aim: 6 幀（~200ms）— 滑鼠移動連續可預測
   - Actions: 2 幀（~67ms）— 按鍵是瞬間事件，需精準
   - Combined: 3 幀（折衷）
   - 快取檔名加入 `la{N}` 避免覆蓋
2. **`DualEvalThread`** (`eval.py`) — 單線程共用截圖，序列跑兩個模型：
   - 一次 `mss.grab(30ms)` → Aim 推理(4ms) → Actions 推理(4ms)
   - FPS: 19 → 21（vs 單模型 29）
3. **推理選單** (`play.py`) — 新增 `[3] Dual Mode`，選 Aim + Actions 模型同時運行

---

### [v0.7.1] 回退管線化截圖

**修改檔案**: `ai/eval.py`

**問題**: v0.7.0 引入的 `ScreenCaptureThread` 使用 busy-wait 跳過重複幀（`if frame is last_frame: continue`），無 sleep 的空轉吃光 CPU → FPS 從 29 暴跌至 10。

**修改內容**: 移除 `ScreenCaptureThread`，回歸串行截圖+推理。

**教訓**: 忙碌等待會搶走其他線程的 CPU 時間。

---

### [v0.7.0] 管線化截圖（實驗性，已回退）

**修改檔案**: `ai/eval.py`

**問題**: 嘗試用獨立線程截圖，推理線程讀最新幀。理論上可重疊 30ms 截圖和 4ms 推理。

**結果**: 因 busy-wait 問題 FPS 反而暴跌，已在 v0.7.1 回退。

---

### [v0.6.1] 修復訓練記憶體不足（OOM）

**修改檔案**: `ai/dataset.py`

**問題**: 兩處 OOM：
1. `np.concatenate` 合併所有 dataset 需要 5-10 GB 連續記憶體
2. `process_raw_dataset` 收集 float32 幀列表 + 轉 float16 = 雙倍峰值

**修改內容**:
1. **懶加載 Dataset** (`OsuLazyDataset`) — 只存索引，`__getitem__` 時才從 chunk 讀取。完全不合併大陣列
2. **即時轉 float16** — 幀處理時立即轉 float16（96KB/幀 vs 192KB/幀）
3. **增量陣列建立** — `np.empty` 預分配 + 逐幀填入 + 逐幀釋放舊記憶體
4. **索引過採樣** — `RandomOverSampler` 只複製索引，不複製圖像資料
5. 記憶體：**~10 GB → ~2 GB**

---

## 2025-02-18

### [v0.6.0] 訓練管線優化 + 環境可攜性

**修改檔案**: `ai/dataset.py`, `ai/train.py`, `ai/utils.py`, `requirements.txt`

**修改內容**:

1. **`lookahead` 3 → 6** (`dataset.py`) — 模型改為預測未來 6 幀（~200ms @ 30fps），更好地補償推理管線延遲
2. **補全 `get_datasets()` 函數** (`utils.py`) — `train.py` import 的函數不存在，導致無法訓練。新增函數列出 `data/raw` 下的所有子資料夾
3. **修正 `get_validated_input` 呼叫** (`train.py:210`) — project name 輸入缺少必要的驗證參數
4. **訓練完自動保存** (`train.py`) — 移除手動確認提示，`finally` 區塊自動保存最佳模型
5. **float16 存儲** (`dataset.py`) — 圖像資料改用 `np.float16` 儲存（記憶體減半），`__getitem__` 時自動轉回 float32 訓練
6. **分塊 Dataset** (`dataset.py`) — 不再將所有 dataset `np.concatenate` 成一個巨大陣列（原本 10.2 GB OOM！），改為分塊保存，使用累計索引查找
7. **更新 `requirements.txt`** — 新增缺少的 `imbalanced-learn`, `mouse`, `tensorboard`, `scikit-learn`，版本改為彈性範圍

---

### [v0.5.2] 移除 FixedRuntime 人為延遲

**修改檔案**: `ai/eval.py`

**問題**: `FixedRuntime(FRAME_DELAY=0.01)` 每幀強制至少 10ms sleep，在 mss 已花 ~33ms 的情況下多加不必要延遲。

**修改內容**:
- 移除 `FixedRuntime` 包裹，迴圈全速運行
- 未啟用時用 `time.sleep(0.001)` 低功耗等待
- 清理未使用的 import（`FRAME_DELAY`, `FixedRuntime`, `PID`）

---

### [v0.5.1] 修復 /255 歸一化遺漏

**修改檔案**: `ai/eval.py`

**問題**: v0.5.0 回退時移除了 `/255.0` 歸一化。模型訓練時使用 `gray_frame / 255.0`（`dataset.py:82`），期望 0-1 範圍輸入。沒有 `/255` 的 0-255 輸入導致模型輸出垃圾值 → **滑鼠在四個角瘋狂跳動**。

**修改內容**: `tensor = torch.from_numpy(stacked).unsqueeze(0).float().div_(255.0)`

**教訓**: 推理的前處理必須與訓練完全一致。

---

## 2025-02-17

### [v0.5.0] 回歸原始推理邏輯（CPU）

**修改檔案**: `ai/eval.py`

**問題**: 一連串 GPU 優化導致成績持續下降（44.73% → 35.34%）。

**根因分析**:
1. **GPU 傳輸開銷** — 80×60 像素的小輸入，CPU→GPU→CPU 的資料搬運比 CPU 直接推理更慢（33 FPS → 27 FPS）
2. **/255 歸一化** — 訓練資料有做 `/255.0`（dataset.py:82），但原始 eval.py 沒有做。模型可能已經「適應」了 0-255 的輸入範圍。加了 `/255` 反而破壞模型預期輸入
3. **FP16 精度損失** — 模型用 FP32 訓練，FP16 推理可能影響小數值的準確度

**修改內容**: 幾乎完全回歸原始 eval.py 邏輯：
- CPU 推理（無 GPU）、FP32、無 /255 歸一化
- 恢復 `FixedRuntime` + `FRAME_DELAY`
- 保留的改動：`torch.jit.load`、`COLOR_BGRA2GRAY`、`SetCursorPos` 直接滑鼠、FPS 計數器

**教訓**: 對於小模型+小輸入，GPU 推理的傳輸開銷可能超過計算加速。出入不要改太多東西。

---

### [v0.4.1] 回退過度優化，恢復穩定版本

**修改檔案**: `ai/eval.py`

**問題**: v0.3.1~v0.4.0 的「優化」反而導致成績從 44.73% 掉到 40.25%（+31 miss）。

**根因分析**:
- `torch.roll()` 每幀分配新 tensor，比 `clone()` 更慢
- `pinned_memory` 對 80×60 的小張量開銷 > 收益
- 獨立 CUDA stream 的 `synchronize()` 反而增加同步等待

**修改內容**: 回退到簡潔版，**保留有效優化，移除有害的**：
- ✅ 保留：GPU 推理、FP16、CUDA warmup、`SetCursorPos`、`COLOR_BGRA2GRAY`、FPS 計數器
- ❌ 移除：`torch.roll`、pinned memory、CUDA stream、dxcam
- 🔄 緩衝區回歸 `deque` + `np.stack`（經驗證最穩定）

---

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
