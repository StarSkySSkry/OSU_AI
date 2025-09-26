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
        # 搜索找到 target_time_ms 所在的索引區間 [i-1, i]。
        # 返回的元組代表用於插值的兩個事件的索引。
        
    def _get_interp_indices(self, target_time_ms: float) -> tuple[int, int]:
        """
        使用 bisect_left 在排序的 event_times 中高效地找到 target_time_ms 的插值索引。
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

        if idx1 == idx2:
            event = self.events[idx1]
            return event['time'], event['x'], event['y']

        a = self.events[idx1]
        b = self.events[idx2]
        
        t = (target_time_ms - a['time']) / (b['time'] - a['time'])
        
        x = a['x'] + t * (b['x'] - a['x'])
        y = a['y'] + t * (b['y'] - a['y'])
        
        return target_time_ms, x, y

    def sample_keys(self, target_time_ms: float) -> tuple[float, list[bool]]:
        """
        在給定時間點對按鍵狀態進行採樣（最近鄰）。
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
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

_model_cache = {EModelType.Aim: [], EModelType.Actions: [], EModelType.Combined: []}

def refresh_model_list():
    """
    刷新模型列表緩存。
    """
    global _model_cache
    _model_cache = {EModelType.Aim: [], EModelType.Actions: [], EModelType.Combined: []}

    if not path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        return

    for model_id in listdir(MODELS_DIR):
        model_dir = path.join(MODELS_DIR, model_id)
        if path.isdir(model_dir):
            info_path = path.join(model_dir, 'info.json')
            if path.exists(info_path):
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
    return _model_cache[model_type]

def run_in_subprocess(file_path: str) -> int:
    """
    在子進程中運行指定的 Python 文件。
    """
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
    playfield_x, playfield_y,
    screen_w, screen_h,
    factory_w=512, factory_h=384,
    factory_dx=0, factory_dy=0,
    account_for_capture_params=False
):
    """
    將 osu! 遊戲區域座標轉換為螢幕座標。
    """
    # 縮放和平移
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
    """
    一個經過優化的 PID 控制器。
    - 增加了積分飽和保護 (Integral Windup Protection)。
    - 採用對測量值微分，以避免微分衝擊 (Derivative Kick)。
    - 增加了輸出限制。
    """
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-100, 100)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0
        self._last_error = 0
        self._last_measurement = 0
        self._last_time = time.perf_counter()

    def update(self, measured_value):
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        if dt == 0:
            return 0

        error = self.setpoint - measured_value
        
        # 比例項
        proportional_term = self.Kp * error
        
        # 積分項 (帶飽和保護)
        self._integral += self.Ki * error * dt
        self._integral = max(self.output_limits[0], min(self.output_limits[1], self._integral))
        integral_term = self._integral

        # 微分項 (對測量值微分)
        # 這樣可以避免目標點突變時引起的微分衝擊
        derivative = (measured_value - self._last_measurement) / dt
        derivative_term = self.Kd * -derivative # 注意這裡有個負號
        
        # 計算總輸出
        output = proportional_term + integral_term + derivative_term
        
        # 限制總輸出
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        # 更新狀態
        self._last_error = error
        self._last_measurement = measured_value
        self._last_time = current_time
        
        return output

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_measurement = 0
        self._last_time = time.perf_counter()

    def set_gains(self, Kp, Ki, Kd):
        """動態調整PID參數"""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

def get_validated_input(prompt: str, is_valid: Callable[[str], bool], convert: Callable[[str], T]) -> T:
    """
    Gets a validated input from the user.

    Args:
        prompt: The prompt to display to the user.
        is_valid: A function that returns True if the input is valid, False otherwise.
        convert: A function that converts the valid input string to the desired type.

    Returns:
        The validated and converted input.
    """
    while True:
        user_input = input(prompt)
        if is_valid(user_input):
            return convert(user_input)
        else:
            print("Invalid input, please try again.")
