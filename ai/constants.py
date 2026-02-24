import torch
from os import getcwd, path, makedirs

ASSETS_DIR = path.normpath(path.join(getcwd(), 'assets'))
MODELS_DIR = path.normpath(path.join(
    getcwd(), 'models'))
RAW_DATA_DIR = path.normpath(path.join(
    getcwd(), 'data', 'raw'))
PROCESSED_DATA_DIR = path.normpath(path.join(
    getcwd(), 'data', 'processed'))

CAPTURE_HEIGHT_PERCENT = 1.0


PLAY_AREA_RATIO = 4 / 3

FINAL_IMAGE_WIDTH = 160  # Increased from 80 to quadruple AI visual fidelity
# FINAL_IMAGE_WIDTH = 80
FINAL_IMAGE_HEIGHT = int(FINAL_IMAGE_WIDTH / PLAY_AREA_RATIO)

FINAL_PLAY_AREA_SIZE = (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT)

PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()  # FP16 半精度加速（僅 CUDA 有效）

CURRENT_STACK_NUM = 10

FRAME_DELAY = 0.01

MAX_THREADS_FOR_RESIZING = 20

DEFAULT_OSU_WINDOW = "osu!"

if not path.exists(RAW_DATA_DIR):
    makedirs(RAW_DATA_DIR)

if not path.exists(PROCESSED_DATA_DIR):
    makedirs(PROCESSED_DATA_DIR)