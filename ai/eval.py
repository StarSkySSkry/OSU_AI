# ai/eval.py — 穩定版：CPU 推理 + 直接滑鼠控制

import json
import os
import time
import traceback
import ctypes
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
                          MODELS_DIR)
from ai.enums import EModelType, EPlayAreaIndices
from ai.utils import derive_capture_params


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

        # --- 載入模型（CPU，與原始訓練環境一致） ---
        print(f"Loading model from: {model_path}")
        eval_model: Module = torch.jit.load(model_path, map_location=device('cpu'))
        eval_model.eval()

        frame_buffer = deque(maxlen=eval_model.channels)

        keyboard.add_hotkey(self.eval_key, lambda: self.toggle_eval(), suppress=True)
        self.on_eval_ready()

        # --- FPS 計數器 ---
        fps_counter = 0
        fps_timer = time.perf_counter()

        with mss() as sct:
            monitor = {"top": self.capture_params[EPlayAreaIndices.OffsetY.value],
                       "left": self.capture_params[EPlayAreaIndices.OffsetX.value],
                       "width": self.capture_params[EPlayAreaIndices.Width.value],
                       "height": self.capture_params[EPlayAreaIndices.Height.value]}

            while True:
                if not self.eval:
                    time.sleep(0.001)
                    continue

                frame = np.array(sct.grab(monitor))
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), FINAL_PLAY_AREA_SIZE)

                needed = eval_model.channels - len(frame_buffer)

                if needed > 0:
                    for i in range(needed):
                        frame_buffer.append(frame)
                else:
                    frame_buffer.append(frame)

                stacked = np.stack(frame_buffer, axis=0)

                with torch.no_grad():
                    tensor = torch.from_numpy(stacked).unsqueeze(0).float().div_(255.0)
                    output = eval_model(tensor)
                    self.on_output(output)

                # FPS 報告
                fps_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed >= 2.0:
                    print(f"[Eval] FPS: {fps_counter / elapsed:.1f}")
                    fps_counter = 0
                    fps_timer = now

    def toggle_eval(self):
        self.eval = not self.eval
        print(f'Eval {"Enabled" if self.eval else "Disabled"}')


def _move_mouse_absolute(capture_params, target_x_percent, target_y_percent):
    """直接將滑鼠移動到目標螢幕座標。"""
    width = capture_params[EPlayAreaIndices.Width.value]
    height = capture_params[EPlayAreaIndices.Height.value]
    offset_x = capture_params[EPlayAreaIndices.OffsetX.value]
    offset_y = capture_params[EPlayAreaIndices.OffsetY.value]

    target_x = int((target_x_percent * width) + offset_x)
    target_y = int((target_y_percent * height) + offset_y)

    ctypes.windll.user32.SetCursorPos(target_x, target_y)


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

        _move_mouse_absolute(self.capture_params, target_x_percent.item(), target_y_percent.item())

        if k1_prob >= 0.5:
            keyboard.press('z')
        else:
            keyboard.release('z')

        if k2_prob >= 0.5:
            keyboard.press('x')
        else:
            keyboard.release('x')
