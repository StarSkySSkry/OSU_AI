# ai/utils.py

import json
import os
import time
import traceback
import cv2
import numpy as np
import bisect
import sys
import subprocess
from os import listdir, path
from datetime import datetime
from ai.constants import RAW_DATA_DIR, MODELS_DIR, CAPTURE_HEIGHT_PERCENT
from ai.enums import EModelType
from typing import TypeVar, Callable, Union

T = TypeVar("T")

class Cv2VideoContext:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)

    def __enter__(self):
        if not self.cap.isOpened():
            raise IOError(f"Error opening video stream or file {self.file_path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()

class EventsSampler:
    """
    一個高效的事件採樣器，使用二分搜索來快速定位時間戳。
    它結合了原有的滑鼠和鍵盤採樣邏輯。
    """
    def __init__(self, events: list[dict]):
        # 確保事件按時間排序
        self.events = sorted(events, key=lambda a: a['time'])
        # 預先提取時間列表，專用於二分搜索
        self.event_times = [e['time'] for e in self.events]
        self.events_num = len(self.events)
        if self.events_num == 0:
            raise ValueError("Cannot initialize EventsSampler with an empty list of events.")

    def _get_interp_indices(self, target_time_ms: float) -> tuple[int, int]:
        """
        使用二分搜索找到 target_time_ms 所在的索引區間 [i-1, i]。
        返回的元組代表用於插值的兩個事件的索引。
        """
        # 處理邊界情況
        if target_time_ms <= self.event_times[0]:
            return 0, 0
        if target_time_ms >= self.event_times[-1]:
            return self.events_num - 1, self.events_num - 1

        # bisect_left 找到插入點，確保 event_times[i-1] <= target_time_ms < event_times[i]
        # 這是O(log n)操作，遠快於線性搜索
        i = bisect.bisect_left(self.event_times, target_time_ms)
        
        # 如果時間完全相等，也返回 i, i (雖然插值時會處理，但明確返回更清晰)
        if self.event_times[i] == target_time_ms:
            return i, i

        return i - 1, i

    def sample_mouse(self, target_time_ms: float) -> tuple[float, float, float]:
        """
        在給定時間點對滑鼠位置進行線性插值採樣。
        返回 (時間, x, y)。
        """
        idx1, idx2 = self._get_interp_indices(target_time_ms)
        
        # 如果在邊界或時間完全匹配，直接返回事件值
        if idx1 == idx2:
            event = self.events[idx1]
            return event['time'], event['x'], event['y']

        a = self.events[idx1]
        b = self.events[idx2]
        
        cur_time = a['time']
        next_time = b['time']
        
        # 防止除以零
        events_dist = next_time - cur_time
        if events_dist == 0:
            return cur_time, a['x'], a['y']

        # 計算插值比例 alpha
        target_time_dist = target_time_ms - cur_time
        alpha = target_time_dist / events_dist
        
        # 線性插值
        interp_x = a["x"] + ((b['x'] - a['x']) * alpha)
        interp_y = a["y"] + ((b['y'] - a['y']) * alpha)
        
        return target_time_ms, interp_x, interp_y

    def sample_keys(self, target_time_ms: float) -> tuple[float, list[bool]]:
        """
        在給定時間點對按鍵狀態進行最近鄰採樣。
        返回 (時間, [k1_pressed, k2_pressed])。
        """
        idx1, idx2 = self._get_interp_indices(target_time_ms)

        if idx1 == idx2:
            event = self.events[idx1]
            return event['time'], event['keys']
            
        a = self.events[idx1]
        b = self.events[idx2]
        
        # 對於離散的按鍵事件，我們採用最近鄰採樣，而不是插值
        dist_to_a = target_time_ms - a['time']
        dist_to_b = b['time'] - target_time_ms

        if dist_to_a <= dist_to_b:
            return a['time'], a['keys']
        else:
            return b['time'], b['keys']


class FixedRuntime:
    """確保此上下文運行給定時間或更長時間。"""
    def __init__(self, target_time: float, debug_name: str = None):
        self.start_time = 0
        self.delay = target_time
        self.debug_name = debug_name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed = time.perf_counter() - self.start_time
        wait_time = self.delay - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        
        if self.debug_name:
            final_elapsed = time.perf_counter() - self.start_time
            print(f"Context [{self.debug_name}] elapsed: {final_elapsed:.4f}s")

# 全局變量用於緩存模型列表
_model_cache = {}

def refresh_model_list():
    """
    從模型目錄中刷新模型信息列表，並緩存在內存中。
    """
    global _model_cache
    _model_cache = {
        EModelType.Aim: [],
        EModelType.Actions: [],
        EModelType.Combined: []
    }
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        return
    
    for model_id in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_id)
        info_path = os.path.join(model_path, 'info.json')

        if os.path.isdir(model_path) and os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    data = json.load(f)
                    model_type = EModelType[data['type']]
                    
                    payload = {
                        'id': model_id,
                        'name': data.get('name', 'Unnamed Model'),
                        'date': datetime.fromisoformat(data['date']),
                        'channels': data.get('channels', 'N/A'),
                        'datasets': data.get('datasets', [])
                    }
                    _model_cache[model_type].append(payload)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not parse info.json for model {model_id}. Error: {e}")

    # 按日期降序排序
    for model_type in _model_cache:
        _model_cache[model_type].sort(key=lambda a: a['date'], reverse=True)

# 首次運行時自動刷新
refresh_model_list()

def get_models(model_type: EModelType) -> list[dict]:
    """獲取指定類型的模型列表。"""
    return _model_cache.get(model_type, [])

def get_datasets() -> list[str]:
    """獲取所有可用的原始數據集名稱。"""
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        return []
    return [d for d in listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

def get_validated_input(
    prompt: str = "You forgot to put your own prompt",
    validate_fn: Callable[[str], bool] = lambda a: len(a.strip()) != 0,
    conversion_fn: Callable[[str], T] = lambda a: a.strip(),
    on_validation_error: Callable[[str], None] = lambda a: print("Invalid input, please try again.")
) -> T:
    """
    一個健壯的函數，用於從用戶那裡獲取經過驗證和轉換的輸入。
    """
    while True:
        user_input = input(prompt)
        if validate_fn(user_input):
            return conversion_fn(user_input)
        else:
            on_validation_error(user_input)

def run_file(file_path: str):
    """在子進程中運行一個 Python 文件。"""
    try:
        process = subprocess.Popen([sys.executable, file_path], shell=False)
        process.communicate()
        return process.returncode
    except FileNotFoundError:
        print(f"Error: Python executable '{sys.executable}' not found.")
        return -1
    except Exception as e:
        print(f"An error occurred while running '{file_path}': {e}")
        return -1

def derive_capture_params(window_width=1920, window_height=1080):
    """
    計算遊戲區域的捕獲參數（寬度、高度、偏移量）。
    """
    osu_play_field_ratio = 4 / 3 # osu! 的遊戲區域是 4:3
    
    # 根據高度和比例計算寬度
    capture_height = int(window_height * CAPTURE_HEIGHT_PERCENT)
    capture_width = int(capture_height * osu_play_field_ratio)
    
    # 計算居中的偏移量
    offset_x = (window_width - capture_width) // 2
    offset_y = (window_height - capture_height) // 2
    
    return [capture_width, capture_height, offset_x, offset_y]

def playfield_coords_to_screen(
    playfield_x: float, playfield_y: float, 
    screen_w=1920, screen_h=1080, account_for_capture_params=False
) -> list[float]:
    """
    將 osu! 的內部 playfield 座標轉換為屏幕座標。
    這段邏輯比較複雜，基於 DANSER 的渲染方式，通常保持不變。
    """
    play_field_ratio = 4 / 3
    screen_ratio = screen_w / screen_h

    play_field_factory_width = 512
    play_field_factory_height = 384 # 512 / (4/3)

    # DANSER 為了保持比例，會在上下或左右添加黑邊
    # 這一步是在模擬 DANSER 的縮放行為
    if screen_ratio > play_field_ratio:
        # 屏幕更寬，上下有黑邊
        factory_h = play_field_factory_height
        factory_w = factory_h * screen_ratio
    else:
        # 屏幕更高，左右有黑邊
        factory_w = play_field_factory_width
        factory_h = factory_w / screen_ratio

    factory_dx = (factory_w - play_field_factory_width) / 2
    factory_dy = (factory_h - play_field_factory_height) / 2
    
    # 應用縮放和平移
    screen_x = (playfield_x * (screen_w / factory_w)) + (factory_dx * (screen_w / factory_w))
    screen_y = (playfield_y * (screen_h / factory_h)) + (factory_dy * (screen_h / factory_h))
    
    # 如果最終的 AI 模型只需要在裁剪後的遊戲區域內移動，
    # 我們需要減去裁剪區域的偏移量
    if account_for_capture_params:
        cap_w, cap_h, cap_dx, cap_dy = derive_capture_params(screen_w, screen_h)
        screen_x -= cap_dx
        screen_y -= cap_dy

    return [screen_x, screen_y, 0, 0] # 返回的 dx, dy 實際上已經被包含在 x, y 中了

class PID:
    """一個簡單的 PID 控制器。"""
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._last_error = 0
        self._integral = 0
        self._last_time = time.time()

    def update(self, measured_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt == 0:
            return 0

        error = self.setpoint - measured_value
        
        # P (Proportional)
        proportional_term = self.Kp * error
        
        # I (Integral)
        self._integral += error * dt
        integral_term = self.Ki * self._integral
        
        # D (Derivative)
        derivative = (error - self._last_error) / dt
        derivative_term = self.Kd * derivative
        
        output = proportional_term + integral_term + derivative_term
        
        self._last_error = error
        self._last_time = current_time
        
        return output

    def reset(self):
        self._last_error = 0
        self._integral = 0
        self._last_time = time.time()