import os
import re
from os import path
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import deque
from torch.utils.data import Dataset

from ai.constants import CURRENT_STACK_NUM, FINAL_PLAY_AREA_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from ai.utils import derive_capture_params
from ai.enums import EModelType

from imblearn.over_sampling import RandomOverSampler


KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
    "11": 3,
}

LOOKAHEAD_BY_MODEL = {
    EModelType.Aim: 2,
    EModelType.Actions: 0,
    EModelType.Combined: 3,
}


class OsuLazyDataset(Dataset):
    """
    懶加載 Dataset：只存索引與路徑，__getitem__ 時從 chunk 中按需讀取。
    """

    def __init__(self, cumulative_lengths, image_paths, key_paths, coord_paths, indices, label_type: EModelType, lookahead: int):
        self._cumulative_lengths = cumulative_lengths
        self._image_paths = image_paths
        self._key_paths = key_paths
        self._coord_paths = coord_paths
        self._indices = list(indices)
        self.label_type = label_type
        self.lookahead = lookahead

        self._image_chunks = None
        self._key_chunks = None
        self._coord_chunks = None

    def _init_chunks(self):
        if self._image_chunks is None:
            self._image_chunks = [np.load(p, mmap_mode="r") for p in self._image_paths]
            self._key_chunks = [np.load(p, mmap_mode="r") for p in self._key_paths]
            self._coord_chunks = [np.load(p, mmap_mode="r") for p in self._coord_paths]

    def _get_chunk_and_index(self, global_idx):
        for chunk_idx in range(len(self._image_paths)):
            start = self._cumulative_lengths[chunk_idx]
            end = self._cumulative_lengths[chunk_idx + 1]
            if start <= global_idx < end:
                return chunk_idx, global_idx - start
        raise IndexError(f"Index {global_idx} out of range [0, {self._cumulative_lengths[-1]})")

    def _pressed(self, chunk_idx, local_idx):
        if local_idx < 0 or local_idx >= len(self._key_chunks[chunk_idx]):
            return False
        return self._key_chunks[chunk_idx][local_idx] != 0

    def _build_actions_dual_label(self, chunk_idx, local_idx, onset_radius=1):
        hold = np.float32(self._pressed(chunk_idx, local_idx))
        onset = np.float32(0.0)

        for off in range(-onset_radius, onset_radius + 1):
            idx = local_idx + off
            if idx < 0 or idx >= len(self._key_chunks[chunk_idx]):
                continue

            curr_pressed = self._pressed(chunk_idx, idx)
            prev_pressed = self._pressed(chunk_idx, idx - 1)

            if curr_pressed and not prev_pressed:
                d = abs(off)
                score = 1.0 if d == 0 else 0.75
                onset = max(onset, np.float32(score))

        return np.array([onset, hold], dtype=np.float32)

    def __getitem__(self, idx):
        self._init_chunks()
        global_idx = self._indices[idx]
        c, l = self._get_chunk_and_index(global_idx)

        start = l
        image = self._image_chunks[c][start:start + CURRENT_STACK_NUM].astype(np.float32)

        if self.label_type == EModelType.Aim:
            label = self._coord_chunks[c][l].copy()
        elif self.label_type == EModelType.Actions:
            label = self._build_actions_dual_label(c, l, onset_radius=1)
        elif self.label_type == EModelType.Combined:
            coords = self._coord_chunks[c][l]
            key = self._key_chunks[c][l]
            k1 = np.float32(key == 2)
            k2 = np.float32(key == 1)
            label = np.concatenate([coords, [k1, k2]])
        else:
            raise ValueError(f"Unsupported label type: {self.label_type}")

        if self.label_type == EModelType.Actions:
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.9, 1.1)
                image = np.clip(image * factor, 0.0, 1.0).astype(np.float32)

            if np.random.rand() > 0.8:
                noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0).astype(np.float32)


        if self.label_type in (EModelType.Aim, EModelType.Combined):
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                label[0] = 1.0 - label[0]

            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                label[1] = 1.0 - label[1]

        return image, label

    def __len__(self):
        return len(self._indices)


class OsuFrameProcessor:
    FILE_REG_EXPR = r"-([0-9]+),[0-1],[0-1],[-0-9.]+,[-0-9.]+\.(?:png|jpg)"

    @staticmethod
    def extract_info_from_state(state: str, dims: tuple[int, int]):
        width, height = dims
        capture_w, capture_h, offset_x, offset_y = derive_capture_params(width, height)

        _, k1, k2, x_str, y_str = state.split(",")

        key_state = KEY_STATES.get(f"{k1}{k2}".strip(), 0)

        x = float(x_str.strip())
        y = float(y_str.strip())

        x_rel = x - offset_x
        y_rel = y - offset_y

        x_norm = max(0.0, min(1.0, x_rel / capture_w)) if capture_w > 0 else 0.0
        y_norm = max(0.0, min(1.0, y_rel / capture_h)) if capture_h > 0 else 0.0

        mouse_state = np.array([x_norm, y_norm], dtype=np.float32)
        return key_state, mouse_state

    @staticmethod
    def process_and_stack_frame(frame: np.ndarray, state: str, original_dims: tuple[int, int], frame_queue: deque):
        capture_w, capture_h, offset_x, offset_y = derive_capture_params(original_dims[0], original_dims[1])
        frame_cropped = frame[offset_y:offset_y + capture_h, offset_x:offset_x + capture_w]

        resized_frame = cv2.resize(frame_cropped, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = (gray_frame / 255.0).astype(np.float32)

        key_state, mouse_state = OsuFrameProcessor.extract_info_from_state(state, original_dims)

        if len(frame_queue) < CURRENT_STACK_NUM - 1:
            frame_queue.append(normalized_frame)
            return None, None, None

        all_frames = list(frame_queue) + [normalized_frame]
        stacked_frames = np.stack(all_frames, axis=0)

        frame_queue.append(normalized_frame)
        return stacked_frames, key_state, mouse_state

    @staticmethod
    def process_raw_dataset(dataset_name: str, lookahead: int = 3, force_rebuild=False, frame_stride: int = 1):
        base_cache_name = path.join(PROCESSED_DATA_DIR, f"{CURRENT_STACK_NUM}-{FINAL_PLAY_AREA_SIZE[0]}-la{lookahead}-fs{frame_stride}-{dataset_name}")
        images_path = base_cache_name + "_images.npy"
        keys_path = base_cache_name + "_keys.npy"
        coords_path = base_cache_name + "_coords.npy"
        raw_data_path = path.join(RAW_DATA_DIR, dataset_name)

        if not force_rebuild and path.exists(images_path) and path.exists(keys_path) and path.exists(coords_path):
            print(f"Loading cached memory-mapped dataset [{dataset_name}]...")
            try:
                coords_data = np.load(coords_path, mmap_mode="r")
                return images_path, keys_path, coords_path, len(coords_data)
            except Exception as e:
                print(f"Failed to load cached files {base_cache_name}. Rebuilding... Error: {e}")

        print(f"Processing raw dataset [{dataset_name}]...")

        files_to_load = os.listdir(raw_data_path)
        if not files_to_load:
            print(f"Warning: Dataset directory {raw_data_path} is empty.")
            return None, None, None, 0

        files_to_load.sort(key=lambda x: int(re.search(OsuFrameProcessor.FILE_REG_EXPR, x).groups()[0]))

        if frame_stride > 1:
            files_to_load = files_to_load[::frame_stride]
            print(f"Applying frame_stride={frame_stride} for [{dataset_name}]...")

        first_frame = cv2.imread(path.join(raw_data_path, files_to_load[0]))
        original_dims = first_frame.shape[:2][::-1]

        def _worker(filename):
            try:
                frame = cv2.imread(path.join(raw_data_path, filename), cv2.IMREAD_COLOR)
                if frame is None:
                    return None

                capture_w, capture_h, offset_x, offset_y = derive_capture_params(original_dims[0], original_dims[1])
                frame_cropped = frame[offset_y:offset_y + capture_h, offset_x:offset_x + capture_w]

                resized_frame = cv2.resize(frame_cropped, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_AREA)
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                normalized_frame = (gray_frame / 255.0).astype(np.float32)

                state_str = filename[:-4].split(os.sep)[-1]
                return normalized_frame, state_str
            except Exception:
                return None

        print(f"Loading and resizing frames for [{dataset_name}]...")
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
            processed_frames_data = list(tqdm(executor.map(_worker, files_to_load), total=len(files_to_load)))

        print(f"Saving optimized sequential frames for [{dataset_name}]...")
        valid_data = [r for r in processed_frames_data if r is not None]
        num_valid = len(valid_data)

        n = num_valid - (CURRENT_STACK_NUM - 1) - lookahead

        if n <= 0:
            print(f"Warning: Not enough frames in {dataset_name} to apply lookahead of {lookahead}. Skipping.")
            return None, None, None, 0

        print("Allocating disk memory-map to prevent RAM OOM crash (single-frame storage)...")
        images_np = np.lib.format.open_memmap(
            images_path,
            mode="w+",
            dtype=np.float16,
            shape=(num_valid, FINAL_PLAY_AREA_SIZE[1], FINAL_PLAY_AREA_SIZE[0]),
        )

        all_keys = []
        all_coords = []

        for idx, result in enumerate(tqdm(valid_data, total=num_valid)):
            normalized_frame, state_str = result
            key_state, mouse_state = OsuFrameProcessor.extract_info_from_state(state_str, original_dims)

            images_np[idx] = normalized_frame.astype(np.float16)
            all_keys.append(key_state)
            all_coords.append(mouse_state)

        start_offset = (CURRENT_STACK_NUM - 1) + lookahead
        keys_final = all_keys[start_offset:start_offset + n]
        coords_final = all_coords[start_offset:start_offset + n]

        keys_np = np.array(keys_final, dtype=np.int64)
        coords_np = np.array(coords_final, dtype=np.float32)

        print(f"Saving uncompressed memory-map ready dataset [{dataset_name}]...")
        images_np.flush()
        np.save(keys_path, keys_np)
        np.save(coords_path, coords_np)

        return images_path, keys_path, coords_path, n


class OsuDatasetBuilder:
    def __init__(self, datasets: list[str], label_type: EModelType, force_rebuild=False):
        self.label_type = label_type
        self._image_paths = []
        self._key_paths = []
        self._coord_paths = []
        self._cumulative_lengths = [0]

        self.lookahead = LOOKAHEAD_BY_MODEL.get(label_type, 2)
        self.frame_stride = 2 if label_type == EModelType.Actions else 1  # Actions 用 50 FPS
        print(f"Using lookahead={self.lookahead} for {label_type.name} model")
        print(f"Using frame_stride={self.frame_stride} for {label_type.name} model")

        for ds_name in datasets:
            res = OsuFrameProcessor.process_raw_dataset(ds_name, lookahead=self.lookahead, force_rebuild=force_rebuild, frame_stride=self.frame_stride)
            if res[3] > 0:
                img_p, key_p, coord_p, length = res
                self._image_paths.append(img_p)
                self._key_paths.append(key_p)
                self._coord_paths.append(coord_p)
                self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)

        self._total_len = self._cumulative_lengths[-1]
        print(f"Total samples loaded: {self._total_len} (across {len(self._image_paths)} chunks)")

    def _get_chunk_and_index(self, global_idx):
        for chunk_idx in range(len(self._image_paths)):
            start = self._cumulative_lengths[chunk_idx]
            end = self._cumulative_lengths[chunk_idx + 1]
            if start <= global_idx < end:
                return chunk_idx, global_idx - start
        raise IndexError(f"Index {global_idx} out of range [0, {self._total_len})")

    def __len__(self):
        return self._total_len

    def get_train_val_datasets(self, val_split=0.1, random_seed=42):
        dataset_size = len(self)
        if dataset_size == 0:
            raise ValueError("Cannot create datasets. No data was loaded.")

        if len(self._image_paths) < 2:
            # 只有單一 chunk 時退回 random split
            indices = list(range(dataset_size))
            val_size = max(1, int(val_split * dataset_size))
            train_size = dataset_size - val_size
            generator = torch.Generator().manual_seed(random_seed)
            train_subset, val_subset = torch.utils.data.random_split(
                indices, [train_size, val_size], generator=generator
            )
            train_indices = list(train_subset)
            val_indices = list(val_subset)
        else:
            # Chunk-wise split：整顆 dataset chunk 分到 train 或 val，不打散時間相鄰樣本
            generator = torch.Generator().manual_seed(random_seed)
            num_chunks = len(self._image_paths)
            perm = torch.randperm(num_chunks, generator=generator).tolist()
            val_chunk_count = max(1, int(round(num_chunks * val_split)))
            val_chunk_count = min(val_chunk_count, num_chunks - 1)
            val_chunks = set(perm[:val_chunk_count])

            train_indices = []
            val_indices = []
            for chunk_idx in range(num_chunks):
                start = self._cumulative_lengths[chunk_idx]
                end = self._cumulative_lengths[chunk_idx + 1]
                if chunk_idx in val_chunks:
                    val_indices.extend(range(start, end))
                else:
                    train_indices.extend(range(start, end))

        validation_dataset = OsuLazyDataset(
            self._cumulative_lengths,
            self._image_paths,
            self._key_paths,
            self._coord_paths,
            val_indices,
            self.label_type,
            self.lookahead,
        )
        print(f"Validation set created with {len(validation_dataset)} samples.")

        if self.label_type == EModelType.Aim:
            training_dataset = OsuLazyDataset(
                self._cumulative_lengths,
                self._image_paths,
                self._key_paths,
                self._coord_paths,
                train_indices,
                self.label_type,
                self.lookahead,
            )
            print(f"Training set created with {len(training_dataset)} samples (Natural Distribution).")
        elif self.label_type == EModelType.Actions:
            training_dataset = OsuLazyDataset(
                self._cumulative_lengths,
                self._image_paths,
                self._key_paths,
                self._coord_paths,
                train_indices,
                self.label_type,
                self.lookahead,
            )
            print(f"Training set created with {len(training_dataset)} samples (no ROS for Actions).")
        else:
            print("Balancing training set (index-only, no image copy)...")

            temp_key_chunks = [np.load(p, mmap_mode="r") for p in self._key_paths]

            train_keys = []
            for idx in train_indices:
                c, l = self._get_chunk_and_index(idx)
                train_keys.append(temp_key_chunks[c][l])
            train_labels = (np.array(train_keys) > 0).astype(np.int64)

            ros = RandomOverSampler(random_state=random_seed)
            train_idx_array = np.array(train_indices).reshape(-1, 1)
            resampled_idx_array, _ = ros.fit_resample(train_idx_array, train_labels)
            resampled_global_indices = resampled_idx_array.flatten().tolist()

            training_dataset = OsuLazyDataset(
                self._cumulative_lengths,
                self._image_paths,
                self._key_paths,
                self._coord_paths,
                resampled_global_indices,
                self.label_type,
                self.lookahead,
            )

            resampled_keys = []
            for gidx in resampled_global_indices:
                c, l = self._get_chunk_and_index(gidx)
                resampled_keys.append(temp_key_chunks[c][l])

            unique, counts = np.unique(resampled_keys, return_counts=True)
            balance_report = dict(zip(map(str, unique), counts))
            print("Final Training Dataset Balance:", balance_report)

            if self.label_type == EModelType.Combined:
                print(f"Training set created with {len(training_dataset)} samples (after balancing).")

        return training_dataset, validation_dataset