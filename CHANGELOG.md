# AutoPlayOSU! 修改紀錄 (CHANGELOG)

> 此檔案記錄所有對專案的修改，方便追蹤和除錯。

---

## 2026-02-24

### [v0.8.6] 突破「線下神準、實戰瞎子」的四大認知 BUG

**修改檔案**: `danser-go/settings/default.json`, `ai/utils.py`, `ai/train.py`, `ai/eval.py`

**問題**: 
模型在 TensorBoard 上的 Validation Accuracy 高達 99.25%，但一旦進入遊戲實戰，準確率驟降至 40% ~ 60%。這揭露了訓練與實戰環境中存在毀滅性的 **Domain Shift（領域偏移）**。經排查有四大沉痾漏洞：

1. **Danser 內建繪製游標 (AI 變成游標追蹤器)**：影片轉檔時錄製到了遊戲內的白色游標，導致模型放棄學習「尋找縮小的外圍圈圈」，改為尋找圖片上的一顆白點。在實戰時 AI 只要看到自己目前的游標位置錯誤，就會陷入追蹤自己錯誤位置的死胡同。
2. **座標幾何數學溢出 (目標位置 < 0 且 > 1)**：在將 `512x384` 的 osu 座標轉換到擷取框時，先前使用的數學公式把 X 軸扣除了 240 像素來置中。這導致畫面左右兩側的目標座標出現 `[-0.15]` 或 `[1.15]` 的奇葩數字。由於神經網路最後一層是 `Sigmoid` (極限是 0.0 ~ 1.0)，模型在物理上「絕對算不出螢幕邊緣的答案」，這構成了早期訓練最高只有 64% 的隱形障蔽。
3. **`cv2.resize` 圖片壓縮法不同步 (INTER_LINEAR 雜訊化)**：轉檔器 (`dataset.py`) 使用了平滑平均的 `cv2.INTER_AREA` 把 1080p 畫面縮成 80x60，光暈非常完美；但實戰推理 (`eval.py`) 漏寫了參數而使用了預設的 `INTER_LINEAR`。實戰中 80x60 的圖片充滿了鋸齒和隨機消失的像素，AI 根本不認識實體的遊戲畫面。
4. **DXCam 時間膨脹 (Time Compression)**：我們移除了 DXCam 每秒擷取 60 幀 (16ms) 的節流閥，讓其以 120+ FPS 狂奔。但 AI 的「記憶池」只有 10 張圖片 (`CURRENT_STACK_NUM = 10`)。全速奔馳會導致這 10 張圖片只涵蓋了 `~80ms` 的時間，遠遠不足以讓 AI 看出 Approach Circle 有沒有在縮小。

**修改內容**:
1. 修改 `danser-go/settings/default.json` 將 `DrawCursors` 設為 `false`，強迫 AI 從畫面無游標的情況下重新學習物件移動特徵。
2. 重寫 `ai/utils.py` 裡的 `playfield_coords_to_screen`，完美按照 4:3 比例原生錨定 `1440x1080` 的擷取框框，確保所有座標輸出 100% 落在 `0.0~1.0` 範圍，並同步調整 `train.py` 中的 `get_acc` 乘數。
3. 修改 `ai/eval.py` 補上 `.resize(..., interpolation=cv2.INTER_AREA)` 來確保實戰與訓練畫面的紋理完全吻合。
4. 在 `ai/eval.py` 中重寫 DXCam 迴圈。DXCam 核心保持不延遲且不排隊的極速提取方式，但在餵入給 GPU 前用 `time.perf_counter()` 強制卡死 `target_frame_time = 1.0 / 60.0`。因此 10 幀的記憶池可以完美橫跨 166ms 的標準時區。

**成效**: 移除了這些讓 AI 變成瞎子的物理與時間障礙後，99% 的核心大腦可以完全發揮其真正實力，實戰準確率將直接飆升，且不再有滑鼠神秘偏移卡死的問題。

---

## 2026-02-23

### [v0.8.5] PyTorch 訓練引擎終極加速

**修改檔案**: `ai/train.py`

**問題**:
使用者回報 AI 訓練時，顯示卡 (RTX 5070) 的使用率只有 51%，而且每秒處理的資料批次太少，訓練過程拖沓。經排查為 PyTorch 預設的 `DataLoader` 採用單線程運行，且未啟用新世代 NVIDIA 顯示卡的自動混合精度 (AMP) 加速。

**修改內容**:
1. **解除 DataLoader I/O 瓶頸**: 將 `num_workers` 改為 8（或自動偵測），並啟用 `prefetch_factor=2` 與 `persistent_workers=True`。這讓 CPU 能夠提前在背景把接下來要用的圖片讀好塞進記憶體，顯示卡永遠不需要等待資料。
2. **啟用全域硬體加速 (AMP & Tensor Cores)**: 在模型 Forward / Backward 計算過程包上一層 `torch.amp.autocast('cuda')`，並搭配 `GradScaler`，這能讓顯示卡使用 FP16 (半精度) 型態計算，不僅 VRAM 消耗減半，計算速度還能大幅提升，徹底榨乾 Tensor Core 的效能。
3. **擴大 Batch Size**: 配合 AMP 的記憶體瘦身，將預設的訓練 `batch_size` 提升 4 倍（從 64 改為 256），一口氣餵更多資料給 GPU 運算。

**成效**: 訓練速度 (`it/s`) 預計將有倍數等級的突破，並能充分發揮高階顯示卡的算力極限。

---

## 2026-02-23

### [v0.8.4] 資料集轉換與訓練載入器 多執行緒加速

**修改檔案**: `ai/convert.py`, `ai/dataset.py`

**問題**: 
使用者回報無論是在轉換資料集，或是開始訓練前「處理原始資料集」（`Processing raw dataset [datasets1]...`）時，每秒只能處理約 50 張圖片 (53 it/s)，導致短影片都要耗費數分鐘才能開始訓練。這是因為 `cv2.imwrite` 和 `cv2.imread` 的單線程 I/O 阻塞了運算。

**修改內容**:
1. **轉換器背景寫入 (`ThreadPoolExecutor`)**：在 `convert.py` 將每一個處理好的圖片交給背景工作執行緒並行寫入硬碟，並調降 PNG 壓縮級別至 1 減少 CPU 開銷。
2. **訓練器並行讀取與縮放 (`ThreadPoolExecutor`)**：在 `dataset.py` 的 `process_raw_dataset` 階段，我們現在也會用線程池（自動開滿玩家所有 CPU 核心）同時讀取與縮放幾萬張圖片，然後再統一排序堆疊起來！

**成效**: 預期原先 53 FPS 的讀寫速度將獲得飛躍性的提升，大幅減少 AI 訓練前置準備的垃圾時間。

---

## 2026-02-22

### [v0.8.3] 修復 AI 時間膨脹與幀數脫節問題

**修改檔案**: `ai/eval.py`

**問題**: 
在解除 DXCam 幀率上限 (`target_fps=0`) 後，AI 的準確率與效能發生了懸崖式下跌（從 78% 掉到 75%，Max Combo 變低，Miss 變多）。
經過除錯發現原因有二：
1. **DXCam 影片緩衝區導致延遲**：原本使用 `video_mode=True`，DXCam 會將抓到的畫面塞進 Ring Buffer 裡排隊。如果背景擷取速度是 200 FPS，但 AI 推理一幀需要 8ms (約 125 FPS)，Ring Buffer 就會塞滿尚未讀取的舊畫面，導致 AI 一直在看「過去的畫面」，發生嚴重延遲。
2. **模型時間感測失調 (Time Dilation)**：模型訓練時是基於 30~60 FPS 的畫面間隔去計算「動態速度」的。如果我們把每秒擷取的幀數拉到 200 FPS，兩張畫面之間的時間間隔只有 5ms，模型會覺得目標移動得非常緩慢，進而預測錯誤的軌跡。

**修改內容**:
1. **關閉影片模式 (`video_mode=False`)**：這告訴 DXCam 不要將畫面排隊，只要在 AI 呼叫 `get_latest_frame()` 時，**直接把最新出爐的那張圖給你**，丟棄中間所有來不及讀取的舊幀，確保絕對零延遲。
2. **AI 推理 60 FPS 鎖定 (`target_frame_time = 1.0 / 60.0`)**：確保 AI 讀取畫面的頻率嚴格鎖定在 60 Hz。這樣一來，背景的 DXCam 依然可以不受限地執行以確保遊戲 FPS 不被鎖死，而 AI 模型也能以他最習慣的「時間流速」來看遊戲，恢復先前的神準表現！

---

## 2025-02-22

### [v0.8.2] 引入 DXCam 顯著降低截圖延遲

**修改檔案**: `ai/eval.py`, `requirements.txt` (自行補充安裝了 `dxcam`)

**問題**: 原本使用的 `mss` (BitBlt) 截圖在快速更新的遊戲畫面中延遲較大，導致 AI 看見的畫面永遠慢了一小段時間，滑鼠跟不上圓圈。雖然在 v0.4.0 嘗試過 DXCam，但當時的實作方式 (無節流的 `get_latest_frame()` 忙碌等待) 導致遊戲卡頓且 CPU 100% 滿載，因此被倒退。

**修改內容**:
重新引入並正確配置 `dxcam` (DirectX Desktop Duplication API) 來取代 `mss`：
1. **精準擷取特定螢幕與區域**：攥寫了 `get_dxcam_monitor_and_region()`，透過 `win32api.EnumDisplayMonitors` 精準對應 osu! 視窗所在的螢幕 (`output_idx`)，並計算出此螢幕上的相對區域 (`region`) 傳給 DXCam，避免截取整個桌面，大幅減少記憶體搬運。
2. **完全解鎖螢幕擷取限制**：修正了 DXCam 預設的 60 FPS 限制。將截圖核心改為 `camera.start(target_fps=0, video_mode=True)`，讓影像提取在背景全速執行，配合 `time.sleep(0.001)` 防止主線程鎖死。這樣一來，AI 模型就能以最微小的延遲拿到最新的畫面（從 60 FPS 解放至 200+ FPS）。
3. **智慧待機釋放資源**：為了確保 AI 未啟用時玩家能完全發揮系統效能，現在程式在進入待機 (`eval = False`) 時會主動呼叫 `camera.stop()` 釋放桌面擷取權，進入遊戲前再重新啟動，徹底解放非遊玩狀態的 FPS。

---

## 2025-02-22

### [v0.8.1] 雙螢幕座標修復 + 推理延遲優化

**修改檔案**: `ai/eval.py`

**問題**: 
1. 當在雙螢幕環境（且在副螢幕遊玩）時，由於 OSU 遊戲標題會在遊玩時改變（例如變成 `"osu!  - 歌手 - 歌曲名稱"`），原本使用 `win32gui.FindWindow(None, "osu!")` 會找不到視窗，導致 AI 預設抓取主螢幕桌面。這會讓模型看到靜態桌面畫面並輸出極端值，導致滑鼠直接卡死在螢幕最右邊。
2. 即使模型使用 CPU 運算，原有的 NumPy 轉 Tensor 的過程 (`torch.from_numpy(stacked).unsqueeze(0).float().div_(255.0)`) 仍有不必要的記憶體複製與轉換開銷，增加推理延遲。

**修改內容**:

1. **強健的視窗定位 (`find_osu_window`)**: 改用 `win32gui.EnumWindows` 來匹配標題**開頭**為 `"osu!"` 的可見視窗，確保不管在選歌還是遊玩中都能正確定位遊戲視窗，並加上 `[Warning]` 提示讓終端機能顯示預設抓取主螢幕的警告。
2. **Tensor 運算優化**: 
   - 將 `torch.no_grad()` 替換為更輕量快速的 `torch.inference_mode()`。
   - 使用 `torch.as_tensor(..., dtype=torch.float32)` 替代 `torch.from_numpy(...).float()` 來盡可能避免記憶體複製。
   - 將除法操作改為原地運算 (`tensor.div_(255.0)`) 來節省記憶體分配開銷，進一步降低 CPU 推理管線延遲。

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
