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

class OsuDataset(Dataset):
    """
    一個基礎的 PyTorch Dataset 類，用於包裝最終的圖像和標籤數據。
    它接收已經處理好的 numpy 陣列。
    """
    def __init__(self, images: np.ndarray, labels, label_type: EModelType):
        self.images = images
        self.labels = labels
        self.label_type = label_type

    def __getitem__(self, idx):
        # 從 numpy 陣列轉換為 PyTorch 需要的 float tensor
        # 轉換步驟在 DataLoader 中進行，可以利用多核，這裡直接返回 numpy array
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.labels)

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
    def process_raw_dataset(dataset_name: str, force_rebuild=False):
        """
        處理單個原始數據集文件夾。
        如果存在快取，則加載。如果不存在，則處理並保存快取。
        """
        processed_data_path = path.join(PROCESSED_DATA_DIR, f"{CURRENT_STACK_NUM}-{FINAL_PLAY_AREA_SIZE[0]}-{dataset_name}.npz")
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
        lookahead = 3 # 預測未來 3 個影格的狀態
        
        if len(all_stacked) > lookahead:
            # 圖像使用較早的影格，標籤使用較晚的影格
            images_final = all_stacked[:-lookahead]
            keys_final = all_keys[lookahead:]
            coords_final = all_coords[lookahead:]
        else:
            # 如果數據太少，無法應用 lookahead，則返回空
            print(f"Warning: Not enough frames in {dataset_name} to apply lookahead of {lookahead}. Skipping.")
            images_final, keys_final, coords_final = [], [], []

        images_np = np.array(images_final, dtype=np.float32)
        keys_np = np.array(keys_final, dtype=np.int64)
        coords_np = np.array(coords_final, dtype=np.float32)
        
        print(f"Saving processed dataset [{dataset_name}]...")
        # 使用壓縮存檔，更節省空間
        np.savez_compressed(processed_data_path, images=images_np, keys=keys_np, coords=coords_np)
        
        return images_np, keys_np, coords_np

class OsuDatasetBuilder:
    """
    負責構建和組織訓練/驗證數據集。
    這是解決數據洩漏的關鍵。
    """
    def __init__(self, datasets: list[str], label_type: EModelType, force_rebuild=False):
        self.label_type = label_type
        all_images, all_keys, all_coords = [], [], []
        
        for ds_name in datasets:
            images, keys, coords = OsuFrameProcessor.process_raw_dataset(ds_name, force_rebuild)
            if len(images) > 0:
                all_images.append(images)
                all_keys.append(keys)
                all_coords.append(coords)
            
        self.images = np.concatenate(all_images, axis=0) if all_images else np.array([])
        self.keys = np.concatenate(all_keys, axis=0) if all_keys else np.array([])
        self.coords = np.concatenate(all_coords, axis=0) if all_coords else np.array([])

    def __len__(self):
        return len(self.images)

    def get_train_val_datasets(self, val_split=0.1, random_seed=42):
        """
        核心方法：分割數據集，並只對訓練集進行平衡。
        """
        dataset_size = len(self)
        if dataset_size == 0:
            raise ValueError("Cannot create datasets. No data was loaded.")
            
        indices = list(range(dataset_size))
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        # 使用 torch.Generator 確保分割可重現
        generator = torch.Generator().manual_seed(random_seed)
        train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)

        # ------------------- 驗證集 (Validation Set) -------------------
        # 驗證集從不進行過採樣，它必須反映真實數據分佈
        val_images = self.images[val_indices]
        if self.label_type == EModelType.Actions:
            val_labels = self.keys[val_indices]
        elif self.label_type == EModelType.Aim:
            val_labels = self.coords[val_indices]
        elif self.label_type == EModelType.Combined:
            k1 = (self.keys[val_indices] == 2).astype(np.float32)[:, None]
            k2 = (self.keys[val_indices] == 1).astype(np.float32)[:, None]
            val_labels = np.hstack([self.coords[val_indices], k1, k2])
        
        validation_dataset = OsuDataset(val_images, val_labels, self.label_type)
        print(f"Validation set created with {len(validation_dataset)} samples.")

        # ------------------- 訓練集 (Training Set) -------------------
        train_images_unbalanced = self.images[train_indices]
        
        if self.label_type == EModelType.Aim:
            # Aim 模型通常不需要平衡（回歸任務）
            train_labels = self.coords[train_indices]
            training_dataset = OsuDataset(train_images_unbalanced, train_labels, self.label_type)
        else:
            # Actions 和 Combined 模型需要平衡
            print("Balancing training set...")
            y_to_balance = self.keys[train_indices]
            
            # imblearn 需要 X 是 2D 的，所以我們先 reshape
            n_samples, n_channels, height, width = train_images_unbalanced.shape
            X_reshaped = train_images_unbalanced.reshape((n_samples, -1))

            ros = RandomOverSampler(random_state=random_seed)
            X_resampled_flat, y_resampled_keys = ros.fit_resample(X_reshaped, y_to_balance)
            
            # 將 X reshape 回原來的形狀
            train_images_balanced = X_resampled_flat.reshape((-1, n_channels, height, width))
            
            if self.label_type == EModelType.Actions:
                train_labels = y_resampled_keys
            elif self.label_type == EModelType.Combined:
                # 當樣本被複製時，我們也需要找到它對應的滑鼠座標
                # 我們可以通過創建一個原始索引的副本來實現
                original_indices = np.arange(len(train_indices))
                _, resampled_indices = ros.fit_resample(original_indices.reshape(-1, 1), y_to_balance)
                resampled_indices = resampled_indices.flatten()
                
                # 從原始訓練數據中提取對應的座標
                coords_resampled = self.coords[train_indices][resampled_indices]
                
                k1 = (y_resampled_keys == 2).astype(np.float32)[:, None]
                k2 = (y_resampled_keys == 1).astype(np.float32)[:, None]
                train_labels = np.hstack([coords_resampled, k1, k2])

            training_dataset = OsuDataset(train_images_balanced, train_labels, self.label_type)
        
        print(f"Training set created with {len(training_dataset)} samples (after balancing).")
        
        # 打印數據平衡結果
        if self.label_type != EModelType.Aim:
            unique, counts = np.unique(y_resampled_keys, return_counts=True)
            balance_report = dict(zip(map(str, unique), counts))
            print("Final Training Dataset Balance:", balance_report)
            
        return training_dataset, validation_dataset