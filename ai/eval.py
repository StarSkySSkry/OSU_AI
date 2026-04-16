# ai/eval.py — v0.7.1: 回歸串行推理（穩定 ~30 FPS）

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
import dxcam
from torch import Tensor, device
from torch.nn import Module

BASE_EVAL_FPS = 100
ACTIONS_INFER_EVERY = 2  # Actions 每 2 幀推理一次，等效 50 FPS，與訓練 frame_stride=2 對齊

from ai.constants import (DEFAULT_OSU_WINDOW, FINAL_PLAY_AREA_SIZE,
                          MODELS_DIR)
from ai.enums import EModelType, EPlayAreaIndices
from ai.utils import derive_capture_params


def find_osu_window(base_name: str) -> int:
    """Find the game window even if its title changes during gameplay (e.g. 'osu!  - Artist - Title')"""
    hwnd = win32gui.FindWindow(None, base_name)
    if hwnd != 0 and not win32gui.IsIconic(hwnd):
        return hwnd
    
    found_hwnd = 0
    def enum_windows_callback(h, lparam):
        nonlocal found_hwnd
        if win32gui.IsWindowVisible(h) and not win32gui.IsIconic(h):
            title = win32gui.GetWindowText(h)
            if title.startswith(base_name):
                found_hwnd = h
                return False
        return True
    
    try:
        win32gui.EnumWindows(enum_windows_callback, 0)
    except Exception:
        pass
    
    return found_hwnd

def get_dxcam_monitor_and_region(capture_params):
    v_left, v_top = capture_params[EPlayAreaIndices.OffsetX.value], capture_params[EPlayAreaIndices.OffsetY.value]
    v_width, v_height = capture_params[EPlayAreaIndices.Width.value], capture_params[EPlayAreaIndices.Height.value]
    
    monitors = win32api.EnumDisplayMonitors()
    target_idx = 0
    m_left, m_top = 0, 0
    
    # 找到視窗中心點所在的 Monitor
    center_x = v_left + v_width // 2
    center_y = v_top + v_height // 2
    
    for i, (hMonitor, hdcMonitor, monitorRect) in enumerate(monitors):
        ml, mt, mr, mb = monitorRect
        if ml <= center_x <= mr and mt <= center_y <= mb:
            target_idx = i
            m_left, m_top = ml, mt
            break
            
    intra_left = int(v_left - m_left)
    intra_top = int(v_top - m_top)
    region = (intra_left, intra_top, intra_left + v_width, intra_top + v_height)
    
    return target_idx, region

class EvalThread(Thread):
    def __init__(self, model_id: str, game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__()
        self.daemon = True
        self.model_id = model_id
        self.game_window_name = game_window_name
        self.eval_key = eval_key
        self.eval = False
        self.capture_params = []

    def get_eval_fps(self):
        return BASE_EVAL_FPS

    def on_output(self, output: Tensor):
        raise NotImplementedError

    def on_eval_ready(self):
        raise NotImplementedError

    def _get_capture_params(self):
        hwnd = find_osu_window(self.game_window_name)
        while hwnd == 0:
            print(f"[Wait] Waiting for '{self.game_window_name}' to be open and un-minimized on screen...")
            time.sleep(2)
            hwnd = find_osu_window(self.game_window_name)

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
        self.model_info = info

        print(f"Loading model from: {model_path} onto CUDA")
        eval_model: Module = torch.jit.load(model_path, map_location=torch.device('cuda'))
        eval_model = eval_model.to('cuda')
        eval_model.eval()

        frame_buffer = deque(maxlen=eval_model.channels)

        keyboard.add_hotkey(self.eval_key, lambda: self.toggle_eval(), suppress=True)
        self.on_eval_ready()

        fps_counter = 0
        fps_timer = time.perf_counter()

        monitor_idx, capture_region = get_dxcam_monitor_and_region(self.capture_params)
        camera = dxcam.create(output_idx=monitor_idx, output_color="GRAY", region=capture_region)

        target_frame_time = 1.0 / self.get_eval_fps()
        last_process_time = time.perf_counter()

        try:
            is_camera_running = False

            while True:
                if not self.eval:
                    if is_camera_running:
                        camera.stop()
                        is_camera_running = False

                    time.sleep(0.01)
                    last_process_time = time.perf_counter()
                    continue
                else:
                    if not is_camera_running:
                        camera.start(target_fps=0, video_mode=False)
                        is_camera_running = True
                        last_process_time = time.perf_counter()

                now = time.perf_counter()
                if now - last_process_time < target_frame_time:
                    continue

                frame = camera.get_latest_frame()
                if frame is None:
                    continue

                last_process_time = time.perf_counter()
                frame = cv2.resize(frame, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_AREA)

                needed = eval_model.channels - len(frame_buffer)
                if needed > 0:
                    for _ in range(needed):
                        frame_buffer.append(frame)
                else:
                    frame_buffer.append(frame)

                stacked = np.stack(frame_buffer, axis=0)

                with torch.inference_mode():
                    tensor = torch.as_tensor(stacked, dtype=torch.uint8, device='cuda').unsqueeze(0)
                    tensor = tensor.to(dtype=torch.float32).div_(255.0)
                    output = eval_model(tensor)

                self.on_output(output.cpu())

                fps_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed >= 2.0:
                    fps = fps_counter / elapsed if elapsed > 0 else 0
                    print(f"[Eval] FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_timer = now
        finally:
            camera.stop()
            del camera

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

def _extract_heatmap_coords(heatmap: torch.Tensor) -> tuple[float, float]:
    """
    從 30x40 的熱點機率圖中提取精確座標 (x, y)。
    使用局部加權平均 (Soft-Argmax) 來獲得次像素級精度。
    """
    h, w = heatmap.shape[-2], heatmap.shape[-1]
    
    # 如果是 V2 (3 channels)，使用偏移量補償
    if heatmap.shape[0] == 3:
        # Ch0: Heatmap, Ch1: Offset_X, Ch2: Offset_Y
        scores = torch.sigmoid(heatmap[0])
        offset_x = heatmap[1]
        offset_y = heatmap[2]
        
        max_idx = torch.argmax(scores)
        y_idx = max_idx // w
        x_idx = max_idx % w
        
        # 讀取對應網格的偏移量並還原
        # 網格座標範圍 [0, w-1]，偏移量 [-0.5, 0.5]
        refined_x = (x_idx.float() + offset_x[y_idx, x_idx]) / (w - 1)
        refined_y = (y_idx.float() + offset_y[y_idx, x_idx]) / (h - 1)
        
        return torch.clamp(refined_x, 0, 1).item(), torch.clamp(refined_y, 0, 1).item()
    
    # 否則使用舊版的 Soft-Argmax
    heatmap = torch.sigmoid(heatmap.squeeze()) 
    
    # 1. 找到最大值像素
    max_idx = torch.argmax(heatmap)
    y_idx = max_idx // w
    x_idx = max_idx % w
    
    # 2. 局部加權平均 (取 3x3 窗口)
    y_start = max(0, y_idx - 1)
    y_end = min(h, y_idx + 2)
    x_start = max(0, x_idx - 1)
    x_end = min(w, x_idx + 2)
    
    window = heatmap[y_start:y_end, x_start:x_end]
    
    # 生成窗口內的相對座標
    grid_y, grid_x = torch.meshgrid(
        torch.arange(y_start, y_end, device=heatmap.device, dtype=torch.float32),
        torch.arange(x_start, x_end, device=heatmap.device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 加權平均
    sum_weight = window.sum()
    if sum_weight <= 0:
        return x_idx.item() / (w - 1), y_idx.item() / (h - 1)
        
    refined_y = (grid_y * window).sum() / sum_weight
    refined_x = (grid_x * window).sum() / sum_weight
    
    # 歸一化到 [0, 1]
    return refined_x.item() / (w - 1), refined_y.item() / (h - 1)

class ActionsThread(EvalThread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_key = 'z'
        self._is_clicking = False
        self._click_start_time = 0.0
        self._onset_thresh = None
        self._hold_exit_thresh = None
        self._min_press_ms = None
        self._slope_prob = None
        self._slope_value = None
        self._prob_log = []
        self._prob_timer = 0.0
        self._onset_history = deque(maxlen=2)  # 2-frame slope（比 3-frame 早 20ms 觸發）
        self._frame_skip = 0  # 主回圈 100 FPS，每 ACTIONS_INFER_EVERY 幀才真正推理
        self._smooth_hold = 0.0  # EMA-smoothed hold_prob（防 premature release）

    def on_eval_ready(self):
        runtime = getattr(self, "model_info", {}).get("runtime", {})
        self._onset_thresh = runtime.get("onset_thresh", 0.50)
        self._hold_exit_thresh = runtime.get("hold_exit_thresh", 0.20)
        self._min_press_ms = runtime.get("min_press_ms", 30.0)
        self._slope_prob = runtime.get("slope_prob", 0.33)
        self._slope_value = runtime.get("slope_value", 0.10)

        print(
            f"Dual-Head Actions Ready "
            f"(onset>{self._onset_thresh:.2f}, hold_exit<{self._hold_exit_thresh:.2f}, "
            f"min_press={self._min_press_ms:.1f}ms, slope>={self._slope_prob:.2f}/{self._slope_value:.2f})"
        )
        print(f" Press '{self.eval_key}' To Toggle")
        self._prob_timer = time.perf_counter()

    def on_output(self, output: Tensor):
        # 主回圈 100 FPS，但每 ACTIONS_INFER_EVERY 幀才執行推理邏輯
        self._frame_skip = (self._frame_skip + 1) % ACTIONS_INFER_EVERY
        if self._frame_skip != 0:
            return

        probs = torch.sigmoid(output[0])
        onset_prob = probs[0].item()
        hold_prob_raw = probs[1].item()

        # EMA 平滑 hold_prob（alpha=0.5，輕度平滑），防止滑條中間瞬降誤放
        HOLD_EMA_ALPHA = 0.5
        self._smooth_hold = HOLD_EMA_ALPHA * self._smooth_hold + (1.0 - HOLD_EMA_ALPHA) * hold_prob_raw
        hold_prob = self._smooth_hold

        self._prob_log.append((onset_prob, hold_prob))
        now = time.perf_counter()
        if now - self._prob_timer >= 2.0 and self._prob_log:
            onset_vals = [p[0] for p in self._prob_log]
            hold_vals = [p[1] for p in self._prob_log]
            print(
                f" [Action] onset avg={sum(onset_vals)/len(onset_vals):.3f} max={max(onset_vals):.3f} "
                f"| hold avg={sum(hold_vals)/len(hold_vals):.3f} max={max(hold_vals):.3f}"
            )
            self._prob_log.clear()
            self._prob_timer = now

        now_ms = time.perf_counter() * 1000.0

        self._onset_history.append(onset_prob)
        slope_trigger = False

        # 2-frame slope trigger：只看最近一段上升，比 3-frame rising 早 20ms 觸發
        if len(self._onset_history) >= 2:
            p1 = self._onset_history[-2]
            p2 = self._onset_history[-1]
            slope = p2 - p1

            if p2 >= self._slope_prob and slope > self._slope_value:
                slope_trigger = True

        if not self._is_clicking:
            if onset_prob > self._onset_thresh or slope_trigger:
                self._is_clicking = True
                self._click_start_time = now_ms

                if self._last_key == 'z':
                    keyboard.release('z')
                    keyboard.press('x')
                    self._last_key = 'x'
                else:
                    keyboard.release('x')
                    keyboard.press('z')
                    self._last_key = 'z'
        else:
            elapsed_ms = now_ms - self._click_start_time
            if hold_prob < self._hold_exit_thresh and elapsed_ms >= self._min_press_ms:
                self._is_clicking = False
                keyboard.release('x')
                keyboard.release('z')


class AimThread(EvalThread):
    def get_eval_fps(self):
        return BASE_EVAL_FPS

    def on_eval_ready(self):
        print(f"Aim Model Ready, Press '{self.eval_key}' To Toggle")

    def on_output(self, output: Tensor):
        out = output[0]
        if out.dim() == 3:
            target_x_percent, target_y_percent = _extract_heatmap_coords(out)
        else:
            target_x_percent, target_y_percent = out[0].item(), out[1].item()

        _move_mouse_absolute(self.capture_params, target_x_percent, target_y_percent)


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


class DualEvalThread(Thread):
    def __init__(self, aim_model_id: str, actions_model_id: str,
                 game_window_name: str = DEFAULT_OSU_WINDOW, eval_key: str = '\\'):
        super().__init__()
        self.daemon = True
        self.aim_model_id = aim_model_id
        self.actions_model_id = actions_model_id
        self.game_window_name = game_window_name
        self.eval_key = eval_key
        self.eval = False
        self.capture_params = []

        self._smooth_x = 0.5
        self._smooth_y = 0.5
        self._last_key = 'z'

        self._is_clicking = False
        self._click_start_time = 0.0
        self._onset_thresh = None
        self._hold_exit_thresh = None
        self._min_press_ms = None
        self._slope_prob = None
        self._slope_value = None
        self._onset_history = deque(maxlen=2)  # 2-frame slope trigger
        self._smooth_hold = 0.0  # EMA-smoothed hold_prob

    def _get_capture_params(self):
        hwnd = find_osu_window(self.game_window_name)
        while hwnd == 0:
            print(f"[Wait] Waiting for '{self.game_window_name}' to be open and un-minimized on screen...")
            time.sleep(2)
            hwnd = find_osu_window(self.game_window_name)

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

    def toggle_eval(self):
        self.eval = not self.eval
        print(f'Eval {"Enabled" if self.eval else "Disabled"}')

    def run(self):
        self._get_capture_params()

        aim_path = os.path.join(MODELS_DIR, self.aim_model_id, 'model.pt')
        actions_path = os.path.join(MODELS_DIR, self.actions_model_id, 'model.pt')
        actions_info_path = os.path.join(MODELS_DIR, self.actions_model_id, 'info.json')

        try:
            with open(actions_info_path, 'r') as f:
                actions_info = json.load(f)
            runtime = actions_info.get("runtime", {})
        except Exception:
            runtime = {}

        self._onset_thresh = runtime.get("onset_thresh", 0.50)
        self._hold_exit_thresh = runtime.get("hold_exit_thresh", 0.20)
        self._min_press_ms = runtime.get("min_press_ms", 30.0)
        self._slope_prob = runtime.get("slope_prob", 0.33)
        self._slope_value = runtime.get("slope_value", 0.10)

        print(f"Loading Aim model from: {aim_path}")
        aim_model: Module = torch.jit.load(aim_path, map_location=device('cuda'))
        aim_model = aim_model.to('cuda')
        aim_model.eval()

        print(f"Loading Actions model from: {actions_path}")
        actions_model: Module = torch.jit.load(actions_path, map_location=device('cuda'))
        actions_model = actions_model.to('cuda')
        actions_model.eval()

        aim_buffer = deque(maxlen=aim_model.channels)
        actions_buffer = deque(maxlen=actions_model.channels)

        keyboard.add_hotkey(self.eval_key, lambda: self.toggle_eval(), suppress=True)
        print(
            f"[Dual Mode] Aim + Dual-Head Actions Ready "
            f"(onset>{self._onset_thresh:.2f}, hold_exit<{self._hold_exit_thresh:.2f}, "
            f"min_press={self._min_press_ms:.1f}ms)"
        )
        print(f" Press '{self.eval_key}' To Toggle")

        fps_counter = 0
        fps_timer = time.perf_counter()

        monitor_idx, capture_region = get_dxcam_monitor_and_region(self.capture_params)
        camera = dxcam.create(output_idx=monitor_idx, output_color="GRAY", region=capture_region)

        target_frame_time = 1.0 / BASE_EVAL_FPS
        actions_infer_interval = ACTIONS_INFER_EVERY
        last_process_time = time.perf_counter()
        loop_count = 0

        prob_log = []

        try:
            is_camera_running = False

            while True:
                if not self.eval:
                    if is_camera_running:
                        camera.stop()
                        is_camera_running = False

                    time.sleep(0.01)
                    last_process_time = time.perf_counter()
                    continue
                else:
                    if not is_camera_running:
                        camera.start(target_fps=0, video_mode=False)
                        is_camera_running = True
                        last_process_time = time.perf_counter()

                now = time.perf_counter()
                if now - last_process_time < target_frame_time:
                    time.sleep(0.001)
                    continue

                frame = camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                last_process_time = time.perf_counter()
                loop_count += 1

                frame = cv2.resize(frame, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_AREA)

                needed = aim_model.channels - len(aim_buffer)
                if needed > 0:
                    for _ in range(needed):
                        aim_buffer.append(frame)
                else:
                    aim_buffer.append(frame)

                aim_stacked = np.stack(aim_buffer, axis=0)
                with torch.inference_mode():
                    aim_tensor = torch.as_tensor(aim_stacked, dtype=torch.uint8, device='cuda').unsqueeze(0)
                    aim_tensor = aim_tensor.to(dtype=torch.float32).div_(255.0)
                    aim_output = aim_model(aim_tensor)

                out_cpu = aim_output[0].cpu()
                if out_cpu.dim() == 3:
                    target_x, target_y = _extract_heatmap_coords(out_cpu)
                else:
                    target_x, target_y = out_cpu[0].item(), out_cpu[1].item()

                _move_mouse_absolute(self.capture_params, target_x, target_y)

                if loop_count % actions_infer_interval == 0:
                    needed = actions_model.channels - len(actions_buffer)
                    if needed > 0:
                        for _ in range(needed):
                            actions_buffer.append(frame)
                    else:
                        actions_buffer.append(frame)

                    actions_stacked = np.stack(actions_buffer, axis=0)
                    with torch.inference_mode():
                        actions_tensor = torch.as_tensor(actions_stacked, dtype=torch.uint8, device='cuda').unsqueeze(0)
                        actions_tensor = actions_tensor.to(dtype=torch.float32).div_(255.0)
                        actions_output = actions_model(actions_tensor)

                    probs = torch.sigmoid(actions_output[0].cpu())
                    onset_prob = probs[0].item()
                    hold_prob_raw = probs[1].item()

                    # EMA 平滑 hold_prob，防 premature release
                    HOLD_EMA_ALPHA = 0.5
                    self._smooth_hold = HOLD_EMA_ALPHA * self._smooth_hold + (1.0 - HOLD_EMA_ALPHA) * hold_prob_raw
                    hold_prob = self._smooth_hold

                    prob_log.append((onset_prob, hold_prob))
                    now_ms = time.perf_counter() * 1000.0

                    self._onset_history.append(onset_prob)
                    slope_trigger = False

                    # 2-frame slope trigger，比 3-frame 早 20ms 觸發
                    if len(self._onset_history) >= 2:
                        p1 = self._onset_history[-2]
                        p2 = self._onset_history[-1]
                        slope = p2 - p1

                        if p2 >= self._slope_prob and slope > self._slope_value:
                            slope_trigger = True

                    if not self._is_clicking:
                        if onset_prob > self._onset_thresh or slope_trigger:
                            self._is_clicking = True
                            self._click_start_time = now_ms

                            if self._last_key == 'z':
                                keyboard.release('z')
                                keyboard.press('x')
                                self._last_key = 'x'
                            else:
                                keyboard.release('x')
                                keyboard.press('z')
                                self._last_key = 'z'
                    else:
                        elapsed_ms = now_ms - self._click_start_time
                        if hold_prob < self._hold_exit_thresh and elapsed_ms >= self._min_press_ms:
                            self._is_clicking = False
                            keyboard.release('x')
                            keyboard.release('z')

                fps_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed >= 2.0:
                    fps = fps_counter / elapsed if elapsed > 0 else 0
                    if prob_log:
                        onset_vals = [p[0] for p in prob_log]
                        hold_vals = [p[1] for p in prob_log]
                        print(
                            f"[Dual] FPS:{fps:.0f} | onset avg={sum(onset_vals)/len(onset_vals):.3f} "
                            f"max={max(onset_vals):.3f} | hold avg={sum(hold_vals)/len(hold_vals):.3f} "
                            f"max={max(hold_vals):.3f}"
                        )
                        prob_log.clear()
                    else:
                        print(f"[Dual] FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_timer = now
        finally:
            camera.stop()
            del camera
