# ai/eval.py

import json
import os
import time
import traceback
from collections import deque
from threading import Thread

import cv2
import keyboard
import mouse
import numpy as np
import torch
import win32api
import win32gui
from mss import mss
from torch import Tensor, device
from torch.nn import Module

from ai.constants import (DEFAULT_OSU_WINDOW, FINAL_PLAY_AREA_SIZE,
                          FRAME_DELAY, MODELS_DIR)
from ai.enums import EModelType, EPlayAreaIndices
from ai.utils import (PID, FixedRuntime, derive_capture_params,
                      playfield_coords_to_screen)


class EvalThread(Thread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__()
        self.daemon = True
        self.model_id = model_id
        self.game_window_name = game_window_name
        self.eval_key = eval_key
        self.eval = False
        self.capture_params = []

    def on_output(self, output: Tensor):
        raise NotImplementedError

    def on_eval_ready(self):
        raise NotImplementedError

    def _get_capture_params(self):
        hwnd = win32gui.FindWindow(None, self.game_window_name)
        if hwnd == 0:
            # Fallback to primary monitor if window not found
            s_width = win32api.GetSystemMetrics(0)
            s_height = win32api.GetSystemMetrics(1)
            client_left = 0
            client_top = 0
        else:
            # Get client area dimensions and position
            client_rect = win32gui.GetClientRect(hwnd)
            s_width = client_rect[2] - client_rect[0]
            s_height = client_rect[3] - client_rect[1]
            
            # Convert client area origin to screen coordinates
            client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))

        capture_width, capture_height, offset_x, offset_y = derive_capture_params(s_width, s_height)

        # Add client area origin to the offsets
        self.capture_params = [
            capture_width,
            capture_height,
            offset_x + client_left,
            offset_y + client_top
        ]

    def run(self):
        self._get_capture_params()
        
        model_path = os.path.join(MODELS_DIR, self.model_id, 'model.pth')
        info_path = os.path.join(MODELS_DIR, self.model_id, 'info.json')

        with open(info_path, 'r') as f:
            info = json.load(f)

        eval_model: Module = torch.load(model_path, map_location=device('cpu'))
        eval_model.eval()

        frame_buffer = deque(maxlen=eval_model.channels)
        
        keyboard.add_hotkey(self.eval_key, lambda: self.toggle_eval(), suppress=True)
        self.on_eval_ready()

        with mss() as sct:
            monitor = {"top": self.capture_params[EPlayAreaIndices.OffsetY.value],
                       "left": self.capture_params[EPlayAreaIndices.OffsetX.value],
                       "width": self.capture_params[EPlayAreaIndices.Width.value],
                       "height": self.capture_params[EPlayAreaIndices.Height.value]}

            while True:
                eval_this_frame = self.eval
                with FixedRuntime(target_time=FRAME_DELAY):
                    if eval_this_frame:
                        frame = np.array(sct.grab(monitor))
                        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), FINAL_PLAY_AREA_SIZE)

                        needed = eval_model.channels - len(frame_buffer)

                        if needed > 0:
                            for i in range(needed):
                                frame_buffer.append(frame)
                        else:
                            frame_buffer.append(frame)

                        stacked = np.stack(frame_buffer, axis=0)
                        
                        with torch.no_grad():
                            tensor = torch.from_numpy(stacked).unsqueeze(0).float()
                            output = eval_model(tensor)
                            self.on_output(output)

    def toggle_eval(self):
        self.eval = not self.eval
        print(f'Eval {"Enabled" if self.eval else "Disabled"}')


class ActionsThread(EvalThread):
    def on_eval_ready(self):
        print(f"Keypress Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        probs = torch.softmax(output, dim=1)
        predicated = torch.argmax(probs, dim=1)
        prob = probs[0][predicated.item()]
        if prob.item() > 0.7:
            state = predicated.item()
            if state == 0:
                keyboard.release('x')
                keyboard.release('z')
            elif state == 1:
                keyboard.release('z')
                keyboard.press('x')
            elif state == 2:
                keyboard.release('x')
                keyboard.press('z')


class AimThread(EvalThread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)
        
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pid_config.json')
        self._load_pid_config() # 初始載入設定

        # 為 X 和 Y 軸創建獨立的 PID 控制器
        self.pid_x = PID(self.pid_params['pid_x']['kp'], self.pid_params['pid_x']['ki'], self.pid_params['pid_x']['kd'],
                         output_limits=(self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max']))
        self.pid_y = PID(self.pid_params['pid_y']['kp'], self.pid_params['pid_y']['ki'], self.pid_params['pid_y']['kd'],
                         output_limits=(self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max']))

        # 設定熱重載
        keyboard.add_hotkey('ctrl+r', self._reload_pid_config)
        print("PID config loaded. Press 'Ctrl+R' to reload pid_config.json at any time.")

    def _load_pid_config(self):
        """從 JSON 檔案載入 PID 參數"""
        try:
            with open(self.config_path, 'r') as f:
                self.pid_params = json.load(f)
            print("PID config loaded successfully.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading PID config: {e}. Using default values.")
            self.pid_params = {
                "pid_x": {"kp": 0.2, "ki": 0.05, "kd": 0.1},
                "pid_y": {"kp": 0.2, "ki": 0.05, "kd": 0.1},
                "output_limits": {"min": -50, "max": 50}
            }
            
    def _reload_pid_config(self):
        """熱重載回呼函數"""
        print("\nReloading PID config...")
        self._load_pid_config()
        self.pid_x.set_gains(self.pid_params['pid_x']['kp'], self.pid_params['pid_x']['ki'], self.pid_params['pid_x']['kd'])
        self.pid_y.set_gains(self.pid_params['pid_y']['kp'], self.pid_params['pid_y']['ki'], self.pid_params['pid_y']['kd'])
        self.pid_x.output_limits = (self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max'])
        self.pid_y.output_limits = (self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max'])


    def on_eval_ready(self):
        print(f"Aim Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        target_x_percent, target_y_percent = output[0]
        width = self.capture_params[EPlayAreaIndices.Width.value]
        height = self.capture_params[EPlayAreaIndices.Height.value]
        offset_x = self.capture_params[EPlayAreaIndices.OffsetX.value]
        offset_y = self.capture_params[EPlayAreaIndices.OffsetY.value]

        # 設定 PID 目標點 (螢幕座標)
        self.pid_x.setpoint = (target_x_percent * width) + offset_x
        self.pid_y.setpoint = (target_y_percent * height) + offset_y

        # 獲取當前滑鼠位置
        current_x, current_y = mouse.get_position()

        # 計算 PID 輸出 (需要移動的距離)
        move_dx = self.pid_x.update(current_x)
        move_dy = self.pid_y.update(current_y)
        
        # 移動滑鼠
        mouse.move(current_x + move_dx, current_y + move_dy)


class CombinedThread(EvalThread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)
     
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pid_config.json')
        self._load_pid_config()

        self.pid_x = PID(self.pid_params['pid_x']['kp'], self.pid_params['pid_x']['ki'], self.pid_params['pid_x']['kd'],
                         output_limits=(self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max']))
        self.pid_y = PID(self.pid_params['pid_y']['kp'], self.pid_params['pid_y']['ki'], self.pid_params['pid_y']['kd'],
                         output_limits=(self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max']))

        keyboard.add_hotkey('ctrl+r', self._reload_pid_config)
        print("PID config loaded. Press 'Ctrl+R' to reload pid_config.json at any time.")

    def _load_pid_config(self):
        """從 JSON 檔案載入 PID 參數"""
        try:
            with open(self.config_path, 'r') as f:
                self.pid_params = json.load(f)
            print("PID config loaded successfully.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading PID config: {e}. Using default values.")
            self.pid_params = {
                "pid_x": {"kp": 0.2, "ki": 0.05, "kd": 0.1},
                "pid_y": {"kp": 0.2, "ki": 0.05, "kd": 0.1},
                "output_limits": {"min": -50, "max": 50}
            }

    def _reload_pid_config(self):
        """熱重載回呼函數"""
        print("\nReloading PID config...")
        self._load_pid_config()
        self.pid_x.set_gains(self.pid_params['pid_x']['kp'], self.pid_params['pid_x']['ki'], self.pid_params['pid_x']['kd'])
        self.pid_y.set_gains(self.pid_params['pid_y']['kp'], self.pid_params['pid_y']['ki'], self.pid_params['pid_y']['kd'])
        self.pid_x.output_limits = (self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max'])
        self.pid_y.output_limits = (self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max'])

    def on_eval_ready(self):
        print(f"Full AI Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        target_x_percent, target_y_percent, k1_prob, k2_prob = output[0]
        
        # --- Mouse control with PID ---
        width = self.capture_params[EPlayAreaIndices.Width.value]
        height = self.capture_params[EPlayAreaIndices.Height.value]
        offset_x = self.capture_params[EPlayAreaIndices.OffsetX.value]
        offset_y = self.capture_params[EPlayAreaIndices.OffsetY.value]

        self.pid_x.setpoint = (target_x_percent * width) + offset_x
        self.pid_y.setpoint = (target_y_percent * height) + offset_y

        current_x, current_y = mouse.get_position()

        move_dx = self.pid_x.update(current_x)
        move_dy = self.pid_y.update(current_y)
        
        mouse.move(current_x + move_dx, current_y + move_dy)

        # --- Keyboard control ---
        if k1_prob >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')

        if k2_prob >= 0.5:
            keyboard.press('x')
        else:
            keyboard.release('x')
