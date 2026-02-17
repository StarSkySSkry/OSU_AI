# ai/eval.py — GPU 加速 + FP16 + 預分配緩衝區 + PidMixin 優化版

import json
import os
import time
import traceback
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
                          FRAME_DELAY, MODELS_DIR, PYTORCH_DEVICE, USE_FP16)
from ai.enums import EModelType, EPlayAreaIndices
from ai.utils import (PID, FixedRuntime, derive_capture_params,
                      playfield_coords_to_screen)


class EvalThread(Thread):
    """推理基類：GPU 加速、FP16、預分配 tensor 緩衝區。"""

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
            s_width = win32api.GetSystemMetrics(0)
            s_height = win32api.GetSystemMetrics(1)
            client_left = 0
            client_top = 0
        else:
            client_rect = win32gui.GetClientRect(hwnd)
            s_width = client_rect[2] - client_rect[0]
            s_height = client_rect[3] - client_rect[1]
            client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))

        capture_width, capture_height, offset_x, offset_y = derive_capture_params(s_width, s_height)

        self.capture_params = [
            capture_width,
            capture_height,
            offset_x + client_left,
            offset_y + client_top
        ]

    def run(self):
        self._get_capture_params()

        model_path = os.path.join(MODELS_DIR, self.model_id, 'model.pt')
        info_path = os.path.join(MODELS_DIR, self.model_id, 'info.json')

        with open(info_path, 'r') as f:
            info = json.load(f)

        # --- 載入模型到 GPU ---
        print(f"Loading model from: {model_path}")
        print(f"Using device: {PYTORCH_DEVICE} | FP16: {USE_FP16}")
        eval_model: Module = torch.jit.load(model_path, map_location=PYTORCH_DEVICE)
        eval_model.eval()
        if USE_FP16:
            eval_model.half()

        num_channels = eval_model.channels
        h, w = FINAL_PLAY_AREA_SIZE[1], FINAL_PLAY_AREA_SIZE[0]
        dtype = torch.float16 if USE_FP16 else torch.float32

        # --- 預分配 tensor 緩衝區 (在 GPU 上) ---
        # shape: (1, channels, H, W)，避免每幀重新分配
        frame_buffer = torch.zeros((1, num_channels, h, w), dtype=dtype, device=PYTORCH_DEVICE)
        buffer_filled = 0  # 追蹤已填入多少幀

        # --- CUDA warmup：預熱 kernel，避免第一幀延遲 ---
        if PYTORCH_DEVICE.type == 'cuda':
            with torch.no_grad():
                _ = eval_model(frame_buffer)
            torch.cuda.synchronize()
            print("CUDA warmup complete.")

        keyboard.add_hotkey(self.eval_key, lambda: self.toggle_eval(), suppress=True)
        self.on_eval_ready()

        with mss() as sct:
            monitor = {
                "top": self.capture_params[EPlayAreaIndices.OffsetY.value],
                "left": self.capture_params[EPlayAreaIndices.OffsetX.value],
                "width": self.capture_params[EPlayAreaIndices.Width.value],
                "height": self.capture_params[EPlayAreaIndices.Height.value],
            }

            while True:
                eval_this_frame = self.eval
                with FixedRuntime(target_time=FRAME_DELAY):
                    if eval_this_frame:
                        # 1. 截圖 → numpy 灰度
                        raw = np.array(sct.grab(monitor))
                        gray = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY), FINAL_PLAY_AREA_SIZE)

                        # 2. numpy → tensor，歸一化，送到 GPU
                        frame_tensor = torch.from_numpy(gray).to(
                            device=PYTORCH_DEVICE, dtype=dtype
                        ).div_(255.0)  # in-place /255，shape: (H, W)

                        # 3. 寫入預分配緩衝區
                        if buffer_filled < num_channels:
                            # 初始填充：用同一幀填滿
                            for i in range(num_channels):
                                frame_buffer[0, i] = frame_tensor
                            buffer_filled = num_channels
                        else:
                            # 滾動：左移一幀，新幀寫入最後一層
                            frame_buffer[0, :-1] = frame_buffer[0, 1:].clone()
                            frame_buffer[0, -1] = frame_tensor

                        # 4. 推理
                        with torch.no_grad():
                            output = eval_model(frame_buffer)
                            # 結果搬回 CPU 給 on_output 處理（滑鼠/鍵盤控制需要 CPU 數值）
                            self.on_output(output.cpu().float())

    def toggle_eval(self):
        self.eval = not self.eval
        print(f'Eval {"Enabled" if self.eval else "Disabled"}')


# ─────────────────────────────────────────────
# PidMixin：AimThread 和 CombinedThread 共用的 PID 邏輯
# ─────────────────────────────────────────────

class PidMixin:
    """提取 PID 控制器的初始化、載入、熱重載邏輯。"""

    def _init_pid(self):
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pid_config.json')
        self._load_pid_config()

        self.pid_x = PID(
            self.pid_params['pid_x']['kp'],
            self.pid_params['pid_x']['ki'],
            self.pid_params['pid_x']['kd'],
            output_limits=(self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max'])
        )
        self.pid_y = PID(
            self.pid_params['pid_y']['kp'],
            self.pid_params['pid_y']['ki'],
            self.pid_params['pid_y']['kd'],
            output_limits=(self.pid_params['output_limits']['min'], self.pid_params['output_limits']['max'])
        )

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

    def _move_mouse_with_pid(self, target_x_percent, target_y_percent):
        """用 PID 控制器平滑移動滑鼠到目標位置"""
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


# ─────────────────────────────────────────────
# 三種模型的 EvalThread 子類
# ─────────────────────────────────────────────

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


class AimThread(PidMixin, EvalThread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)
        self._init_pid()

    def on_eval_ready(self):
        print(f"Aim Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        target_x_percent, target_y_percent = output[0]
        self._move_mouse_with_pid(target_x_percent.item(), target_y_percent.item())


class CombinedThread(PidMixin, EvalThread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__(model_id, game_window_name, eval_key)
        self._init_pid()

    def on_eval_ready(self):
        print(f"Full AI Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        target_x_percent, target_y_percent, k1_prob, k2_prob = output[0]

        # 滑鼠控制
        self._move_mouse_with_pid(target_x_percent.item(), target_y_percent.item())

        # 鍵盤控制
        if k1_prob >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')

        if k2_prob >= 0.5:
            keyboard.press('x')
        else:
            keyboard.release('x')
