# ai/convert.py (完整版，注意 start_convert 的變化)

import cv2
import json
import shutil
import os
import traceback
from tqdm import tqdm
from ai.utils import Cv2VideoContext, EventsSampler, playfield_coords_to_screen, derive_capture_params, get_validated_input
from ai.constants import RAW_DATA_DIR

# ReplayConverter 類保持不變，因為它已經是優化後的版本了
class ReplayConverter:
    """
    將 DANSER 渲染的影片和 replay JSON 轉換為 AI 訓練所需的幀數據集。
    採用簡化、穩定的單線程順序讀取流程。
    """

    def __init__(self, project_name: str, danser_video: str, replay_json: str,
                 save_dir: str = "", frame_offset_ms: int = 0,
                 replay_keys_json: str = None, remove_breaks: bool = True):
        
        self.project_name = project_name
        self.save_dir = os.path.join(save_dir, self.project_name)
        self.danser_video = danser_video
        self.replay_json = replay_json
        self.replay_keys_json = replay_keys_json
        self.frame_offset_ms = frame_offset_ms
        self.remove_breaks = remove_breaks
        
        # 準備保存目錄
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        
        # 立即開始轉換
        self.build_dataset()

    def _load_events(self):
        # ... (此方法保持不變)
        with open(self.replay_json, 'r') as f:
            replay_data = json.load(f)

        replay_keys_data = None
        if self.replay_keys_json:
            with open(self.replay_keys_json, 'r') as f:
                replay_keys_data = json.load(f)

        start_time = replay_data["objects"][0]["start"]
        time_offset = start_time + self.frame_offset_ms

        events_mouse = []
        total_time_mouse = 0
        for event in replay_data["events"]:
            total_time_mouse += event['diff']
            events_mouse.append({
                "x": event["x"],
                "y": event["y"],
                "time": total_time_mouse - time_offset,
            })

        events_keys = []
        if replay_keys_data is None:
            total_time_keys = 0
            for event in replay_data["events"]:
                total_time_keys += event['diff']
                events_keys.append({
                    "keys": [event['k1'], event['k2']],
                    "time": total_time_keys - time_offset,
                })
        else:
            total_time_keys = 0
            for event in replay_keys_data["events"]:
                total_time_keys += event['diff']
                events_keys.append({
                    "keys": [event['k1'], event['k2']],
                    "time": total_time_keys - time_offset,
                })

        breaks = []
        if self.remove_breaks:
            for b in replay_data.get('breaks', []):
                breaks.append({
                    "start": b["start"] - time_offset,
                    "end": b["end"] - time_offset
                })
        
        stop_time = max(events_mouse[-1]['time'], events_keys[-1]['time']) if events_mouse and events_keys else 0
        
        return events_mouse, events_keys, breaks, stop_time

    def build_dataset(self):
        # ... (此方法保持不變)
        try:
            print("Loading and processing replay events...")
            events_mouse, events_keys, breaks, stop_time = self._load_events()

            if not events_mouse or not events_keys:
                print("Error: Replay events could not be loaded. Aborting.")
                return

            mouse_sampler = EventsSampler(events_mouse)
            keys_sampler = EventsSampler(events_keys)
            
            print("Starting video frame processing...")
            with Cv2VideoContext(self.danser_video) as ctx:
                total_frames = int(ctx.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = ctx.cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    raise ValueError("Video FPS is 0, cannot process.")
                
                frame_duration_ms = 1000.0 / fps

                screen_h = int(ctx.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                screen_w = int(ctx.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                capture_w, capture_h, capture_dx, capture_dy = derive_capture_params(screen_w, screen_h)
                
                loading_bar = tqdm(total=total_frames, desc=f"Converting [{self.project_name}]")

                for frame_idx in range(total_frames):
                    read_success, frame = ctx.cap.read()
                    if not read_success:
                        loading_bar.update(1)
                        continue

                    current_time_ms = frame_idx * frame_duration_ms

                    if current_time_ms > stop_time + frame_duration_ms:
                        remaining_frames = total_frames - frame_idx
                        loading_bar.update(remaining_frames)
                        break

                    in_break = any(b["start"] <= current_time_ms <= b["end"] for b in breaks)
                    if self.remove_breaks and in_break:
                        loading_bar.update(1)
                        continue
                    
                    _, sampled_x, sampled_y = mouse_sampler.sample_mouse(current_time_ms)
                    _, keys_bool = keys_sampler.sample_keys(current_time_ms)

                    screen_x, screen_y, _, _ = playfield_coords_to_screen(
                        sampled_x, sampled_y, screen_w, screen_h, account_for_capture_params=True
                    )
                    
                    cropped_frame = frame[capture_dy : capture_dy + capture_h, capture_dx : capture_dx + capture_w]
                    
                    k1_str = "1" if keys_bool[0] else "0"
                    k2_str = "1" if keys_bool[1] else "0"
                    
                    image_file_name = f"{self.project_name}-{round(current_time_ms)},{k1_str},{k2_str},{screen_x:.2f},{screen_y:.2f}.png"
                    image_path = os.path.join(self.save_dir, image_file_name)
                    
                    cv2.imwrite(image_path, cropped_frame)
                    
                    loading_bar.update(1)

            loading_bar.close()
            print(f"Dataset '{self.project_name}' created successfully at '{self.save_dir}'")

        except Exception as e:
            print(f"\nAn error occurred during replay conversion:")
            traceback.print_exc()

# --- MODIFIED FUNCTION BELOW ---

def start_convert():
    """用戶交互界面，用於啟動轉換過程。"""
    try:
        project_name = get_validated_input(
            'What Would You Like To Name This Project?: ', 
            conversion_fn=lambda a: a.lower().strip()
        )
        rendered_path = get_validated_input(
            'Path to the rendered replay video: ', 
            validate_fn=lambda a: os.path.exists(a.strip()),
            conversion_fn=lambda a: a.strip(), 
            on_validation_error=lambda a: print("Invalid path!")
        )
        replay_json = get_validated_input(
            'Path to the replay JSON: ', 
            validate_fn=lambda a: os.path.exists(a.strip()),
            conversion_fn=lambda a: a.strip(), 
            on_validation_error=lambda a: print("Invalid path!")
        )

        has_keys_json_str = get_validated_input(
            'Do you have a separate keys-only replay JSON? (y/n): ',
            validate_fn=lambda a: a.strip().lower() in ['y', 'n'],
            conversion_fn=lambda a: a.strip().lower()
        )
        has_keys_json = has_keys_json_str.startswith('y')

        replay_keys_json = None
        if has_keys_json:
            replay_keys_json = get_validated_input(
                'Path to the keys-only replay JSON: ',
                validate_fn=lambda a: os.path.exists(a.strip()),
                conversion_fn=lambda a: a.strip(),
                on_validation_error=lambda a: print("Invalid path!")
            )

        offset_ms = get_validated_input(
            "Offset in ms to apply to the dataset (e.g., -100, default is 0): ",
            validate_fn=lambda a: not a.strip() or a.strip().lstrip('-+').isdigit(),
            conversion_fn=lambda a: int(a.strip()) if a.strip() else 0,
            on_validation_error=lambda a: print("It must be an integer or empty.")
        )
        
        # 創建 ReplayConverter 實例時，只傳入它需要的參數
        ReplayConverter(
            project_name=project_name, 
            danser_video=rendered_path, 
            replay_json=replay_json,
            save_dir=RAW_DATA_DIR, 
            frame_offset_ms=offset_ms,
            replay_keys_json=replay_keys_json
        )
    except Exception:
        # 確保在文件頂部有 import traceback
        print("\nAn error occurred during the conversion setup:")
        traceback.print_exc()