# ai/dataset.py

import os
import re
from os import path
import cv2
import numpy as np
import torch
import traceback
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ai.constants import CURRENT_STACK_NUM, FINAL_PLAY_AREA_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from collections import deque
from torch.utils.data import Dataset, Subset
from ai.enums import EModelType

# 需要安裝: pip install imbalanced-learn
# imblearn 用於處理數據不平衡問題，比手動複製更高效、更健壯
from imblearn.over_sampling import RandomOverSampler

KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
}

# 每種模型類型使用不同的 lookahead（影格數）
# Aim: 滑鼠移動連續平滑，可以預測較遠
# Actions: 按鍵是瞬間事件，需要精準時機
# Combined: 折衷
LOOKAHEAD_BY_MODEL = {
    EModelType.Aim: 6,       # ~200ms @ 30fps
    EModelType.Actions: 2,   # ~67ms @ 30fps
    EModelType.Combined: 3,  # ~100ms @ 30fps
}


class OsuLazyDataset(Dataset):
    """
    懶加載 Dataset：只存索引，__getitem__ 時從 chunk 中按需讀取。
    完全不需要把圖像合併到一個大陣列中，零額外記憶體。
    """
    def __init__(self, builder, indices, label_type: EModelType):
        self._builder = builder
        self._indices = list(indices)
        self.label_type = label_type

    def __getitem__(self, idx):
        global_idx = self._indices[idx]
        c, l = self._builder._get_chunk_and_index(global_idx)
        
        image = self._builder._image_chunks[c][l].astype(np.float32)
        
        if self.label_type == EModelType.Aim:
            label = self._builder._coord_chunks[c][l]
        elif self.label_type == EModelType.Actions:
            label = self._builder._key_chunks[c][l]
        elif self.label_type == EModelType.Combined:
            coords = self._builder._coord_chunks[c][l]
            key = self._builder._key_chunks[c][l]
            k1 = np.float32(key == 2)
            k2 = np.float32(key == 1)
            label = np.concatenate([coords, [k1, k2]])
        
        return image, label

    def __len__(self):
        return len(self._indices)


class OsuFrameProcessor:
    """
    將所有靜態方法組織到一個類中，便於管理。
    """
    FILE_REG_EXPR = r"-([0-9]+),[0-1],[0-1],[-0-9.]+,[-0-9.]+.png"

    @staticmethod
    def extract_info_from_state(state: str, dims: tuple[int, int]):
        """從檔名中解析出按鍵和座標信息"""
        width, height = dims
        _, k1, k2, x_str, y_str = state.split(',')
        
        # 鍵盤狀態
        key_state = KEY_STATES.get(f"{k1}{k2}".strip(), 0)
        
        # 座標歸一化
        x = max(0, float(x_str.strip()))
        y = max(0, float(y_str.strip()))
        x_norm = x / width if x > 0 else 0
        y_norm = y / height if y > 0 else 0
        
        mouse_state = np.array([x_norm, y_norm], dtype=np.float32)
        
        return key_state, mouse_state

    @staticmethod
    def process_and_stack_frame(frame: np.ndarray, state: str, original_dims: tuple[int, int], frame_queue: deque):
        """
        處理單個影格：resize -> 灰度 -> 歸一化 -> 解析信息 -> 堆疊
        """
        # 1. Resize
        resized_frame = cv2.resize(frame, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_AREA)
        
        # 2. 灰度化和歸一化
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = (gray_frame / 255.0).astype(np.float32)

        # 3. 解析檔名信息
        key_state, mouse_state = OsuFrameProcessor.extract_info_from_state(state, original_dims)
        
        # 4. 堆疊影格
        # 維持一個固定長度的隊列
        if len(frame_queue) < CURRENT_STACK_NUM - 1:
            frame_queue.append(normalized_frame)
            return None, None, None # 還不夠堆疊
        
        # 將舊影格和當前影格合併
        all_frames = list(frame_queue) + [normalized_frame]
        stacked_frames = np.stack(all_frames, axis=0) # shape: (C, H, W)
        
        # 更新隊列，移除最舊的影格，加入最新的
        frame_queue.append(normalized_frame)
        
        return stacked_frames, key_state, mouse_state

    @staticmethod
    def process_raw_dataset(dataset_name: str, lookahead: int = 3, force_rebuild=False):
        """
        處理單個原始數據集文件夾。
        如果存在快取，則加載。如果不存在，則處理並保存快取。
        快取檔名包含 lookahead 值，不同設定互不覆蓋。
        """
        processed_data_path = path.join(PROCESSED_DATA_DIR, f"{CURRENT_STACK_NUM}-{FINAL_PLAY_AREA_SIZE[0]}-la{lookahead}-{dataset_name}.npz")
        raw_data_path = path.join(RAW_DATA_DIR, dataset_name)

        if not force_rebuild and path.exists(processed_data_path):
            print(f"Loading cached processed dataset [{dataset_name}]...")
            try:
                data = np.load(processed_data_path)
                return data['images'], data['keys'], data['coords']
            except Exception as e:
                print(f"Failed to load cached file {processed_data_path}. Rebuilding... Error: {e}")

        print(f"Processing raw dataset [{dataset_name}]...")
        
        files_to_load = os.listdir(raw_data_path)
        if not files_to_load:
            print(f"Warning: Dataset directory {raw_data_path} is empty.")
            return np.array([]), np.array([]), np.array([])
            
        files_to_load.sort(key=lambda x: int(re.search(OsuFrameProcessor.FILE_REG_EXPR, x).groups()[0]))
        
        all_stacked, all_keys, all_coords = [], [], []
        
        frame_queue = deque(maxlen=CURRENT_STACK_NUM - 1)
        
        first_frame = cv2.imread(path.join(raw_data_path, files_to_load[0]))
        original_dims = first_frame.shape[:2][::-1]

        loading_bar = tqdm(desc=f"Processing [{dataset_name}]", total=len(files_to_load))
        
        for filename in files_to_load:
            try:
                frame = cv2.imread(path.join(raw_data_path, filename), cv2.IMREAD_COLOR)
                state_str = filename[:-4].split(os.sep)[-1] # 確保只取檔名部分
                
                stacked, key, coords = OsuFrameProcessor.process_and_stack_frame(frame, state_str, original_dims, frame_queue)
                
                if stacked is not None:
                    all_stacked.append(stacked)
                    all_keys.append(key)
                    all_coords.append(coords)
            except Exception:
                traceback.print_exc()
            finally:
                loading_bar.update()

        loading_bar.close()
        
        # --- Lookahead Implementation ---
        # 讓模型學習預測未來，以補償延遲
        
        if len(all_stacked) > lookahead:
            # 圖像使用較早的影格，標籤使用較晚的影格
            images_final = all_stacked[:-lookahead]
            keys_final = all_keys[lookahead:]
            coords_final = all_coords[lookahead:]
        else:
            # 如果數據太少，無法應用 lookahead，則返回空
            print(f"Warning: Not enough frames in {dataset_name} to apply lookahead of {lookahead}. Skipping.")
            images_final, keys_final, coords_final = [], [], []

        images_np = np.array(images_final, dtype=np.float16)
        keys_np = np.array(keys_final, dtype=np.int64)
        coords_np = np.array(coords_final, dtype=np.float32)
        
        print(f"Saving processed dataset [{dataset_name}]...")
        # 使用壓縮存檔，更節省空間
        np.savez_compressed(processed_data_path, images=images_np, keys=keys_np, coords=coords_np)
        
        return images_np, keys_np, coords_np


class OsuDatasetBuilder:
    """
    負責構建和組織訓練/驗證數據集。
    使用分塊存儲 + 懶加載，完全避免大陣列合併（零額外記憶體）。
    """
    def __init__(self, datasets: list[str], label_type: EModelType, force_rebuild=False):
        self.label_type = label_type
        self._image_chunks = []
        self._key_chunks = []
        self._coord_chunks = []
        self._cumulative_lengths = [0]
        
        lookahead = LOOKAHEAD_BY_MODEL.get(label_type, 3)
        print(f"Using lookahead={lookahead} for {label_type.name} model")
        
        for ds_name in datasets:
            images, keys, coords = OsuFrameProcessor.process_raw_dataset(ds_name, lookahead=lookahead, force_rebuild=force_rebuild)
            if len(images) > 0:
                self._image_chunks.append(images)
                self._key_chunks.append(keys)
                self._coord_chunks.append(coords)
                self._cumulative_lengths.append(self._cumulative_lengths[-1] + len(images))
        
        self._total_len = self._cumulative_lengths[-1]
        print(f"Total samples loaded: {self._total_len} (across {len(self._image_chunks)} chunks)")

    def _get_chunk_and_index(self, global_idx):
        """將全局索引轉換為 (chunk_idx, local_idx)"""
        for chunk_idx in range(len(self._image_chunks)):
            start = self._cumulative_lengths[chunk_idx]
            end = self._cumulative_lengths[chunk_idx + 1]
            if start <= global_idx < end:
                return chunk_idx, global_idx - start
        raise IndexError(f"Index {global_idx} out of range [0, {self._total_len})")

    def __len__(self):
        return self._total_len

    def get_train_val_datasets(self, val_split=0.1, random_seed=42):
        """
        核心方法：分割數據集，並只對訓練集進行平衡。
        全程只操作索引，不複製圖像資料（零額外記憶體）。
        """
        dataset_size = len(self)
        if dataset_size == 0:
            raise ValueError("Cannot create datasets. No data was loaded.")
            
        indices = list(range(dataset_size))
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        generator = torch.Generator().manual_seed(random_seed)
        train_subset, val_subset = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
        train_indices = list(train_subset)
        val_indices = list(val_subset)

        # ------------------- 驗證集 -------------------
        validation_dataset = OsuLazyDataset(self, val_indices, self.label_type)
        print(f"Validation set created with {len(validation_dataset)} samples.")

        # ------------------- 訓練集 -------------------
        if self.label_type == EModelType.Aim:
            # Aim: 回歸任務，不需要平衡
            training_dataset = OsuLazyDataset(self, train_indices, self.label_type)
        else:
            # Actions / Combined: 需要平衡按鍵分佈
            print("Balancing training set (index-only, no image copy)...")
            
            # 只收集按鍵標籤用於平衡（很小，幾十 KB）
            train_keys = []
            for idx in train_indices:
                c, l = self._get_chunk_and_index(idx)
                train_keys.append(self._key_chunks[c][l])
            train_keys = np.array(train_keys)
            
            # 對索引進行過採樣（不複製任何圖像資料！）
            ros = RandomOverSampler(random_state=random_seed)
            train_idx_array = np.array(train_indices).reshape(-1, 1)
            resampled_idx_array, _ = ros.fit_resample(train_idx_array, train_keys)
            resampled_global_indices = resampled_idx_array.flatten().tolist()
            
            training_dataset = OsuLazyDataset(self, resampled_global_indices, self.label_type)
            
            # 打印平衡結果
            resampled_keys = []
            for gidx in resampled_global_indices:
                c, l = self._get_chunk_and_index(gidx)
                resampled_keys.append(self._key_chunks[c][l])
            unique, counts = np.unique(resampled_keys, return_counts=True)
            balance_report = dict(zip(map(str, unique), counts))
            print("Final Training Dataset Balance:", balance_report)
        
        print(f"Training set created with {len(training_dataset)} samples (after balancing).")
        return training_dataset, validation_dataset