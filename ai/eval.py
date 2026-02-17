# ai/eval.py — GPU 加速 + FP16 + 預分配緩衝區 + 零延遲優化版

import json
import os
import time
import traceback
import ctypes
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
                          MODELS_DIR, PYTORCH_DEVICE, USE_FP16)
from ai.enums import EModelType, EPlayAreaIndices
from ai.utils import (derive_capture_params,
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
                if not self.eval:
                    time.sleep(0.001)  # 未啟用時低功耗等待
                    continue

                # 1. 截圖 → numpy 灰度（mss 輸出 BGRA，直接轉灰度）
                raw = np.array(sct.grab(monitor))
                gray = cv2.resize(
                    cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY),
                    FINAL_PLAY_AREA_SIZE
                )

                # 2. numpy → GPU tensor，歸一化
                frame_tensor = torch.from_numpy(gray).to(
                    device=PYTORCH_DEVICE, dtype=dtype
                ).div_(255.0)

                # 3. 寫入預分配緩衝區（GPU 上 in-place 操作）
                if buffer_filled < num_channels:
                    for i in range(num_channels):
                        frame_buffer[0, i] = frame_tensor
                    buffer_filled = num_channels
                else:
                    # torch.roll 比 slice+clone 更快（GPU kernel 級操作）
                    frame_buffer = torch.roll(frame_buffer, shifts=-1, dims=1)
                    frame_buffer[0, -1] = frame_tensor

                # 4. 推理 → 結果搬回 CPU
                with torch.no_grad():
                    output = eval_model(frame_buffer)
                self.on_output(output.detach().cpu().float())

    def toggle_eval(self):
        self.eval = not self.eval
        print(f'Eval {"Enabled" if self.eval else "Disabled"}')


def _move_mouse_absolute(capture_params, target_x_percent, target_y_percent):
    """直接將滑鼠移動到目標螢幕座標（無 PID，零延遲）。"""
    width = capture_params[EPlayAreaIndices.Width.value]
    height = capture_params[EPlayAreaIndices.Height.value]
    offset_x = capture_params[EPlayAreaIndices.OffsetX.value]
    offset_y = capture_params[EPlayAreaIndices.OffsetY.value]

    target_x = int((target_x_percent * width) + offset_x)
    target_y = int((target_y_percent * height) + offset_y)

    # 使用 win32api 直接設定絕對座標，比 mouse.move 更快更精準
    ctypes.windll.user32.SetCursorPos(target_x, target_y)


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


class AimThread(EvalThread):
    def on_eval_ready(self):
        print(f"Aim Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        target_x_percent, target_y_percent = output[0]
        _move_mouse_absolute(self.capture_params, target_x_percent.item(), target_y_percent.item())


class CombinedThread(EvalThread):
    def on_eval_ready(self):
        print(f"Full AI Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        target_x_percent, target_y_percent, k1_prob, k2_prob = output[0]

        # 滑鼠控制：直接跳到目標座標
        _move_mouse_absolute(self.capture_params, target_x_percent.item(), target_y_percent.item())

        # 鍵盤控制
        if k1_prob >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')

        if k2_prob >= 0.5:
            keyboard.press('x')
        else:
            keyboard.release('x')
