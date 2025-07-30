# ai/models.py

import torch
import torch.nn as nn
import uuid
import os
import json
import timm
from typing import Callable, Optional
from datetime import datetime
from ai.constants import CURRENT_STACK_NUM
from ai.utils import refresh_model_list
from ai.enums import EModelType

def get_timm_model(
    model_name: str, 
    out_features: int,
    channels: int = 3, 
    pretrained: bool = False
) -> nn.Module:
    """
    從 timm 庫創建一個模型，並替換其最終的分類層。

    Args:
        model_name (str): 要創建的模型的名稱 (e.g., 'resnet18', 'efficientnet_b0').
        out_features (int): 最終輸出層的特徵數量。
        channels (int): 輸入圖像的通道數。
        pretrained (bool): 是否加載預訓練權重。

    Returns:
        nn.Module: 配置好的 timm 模型。
    """
    model = timm.create_model(
        model_name=model_name, 
        pretrained=pretrained, 
        in_chans=channels, 
        num_classes=out_features  # 直接在這裡設置輸出大小
    )
    return model

class OsuAiModel(nn.Module):
    """
    所有 AI模型的基類，提供了通用的保存和加載功能。
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM, model_type: EModelType = EModelType.Unknown):
        super().__init__()
        self.channels = channels
        self.model_type = model_type
        # 將模型名稱作為一個可配置的屬性
        self.backbone_name = "efficientnet_b0" # 選擇一個高效且性能良好的模型

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基類中的 forward 應該被子類重寫
        raise NotImplementedError("Each model must implement its own forward pass.")

    def save(self, project_name: str, datasets: list[str], epochs: int, learning_rate: float, path: str = './models', weights: Optional[dict] = None):
        """保存模型狀態、腳本化模型和元數據到磁碟。"""
        model_id = str(uuid.uuid4())
        save_dir = os.path.join(path, model_id)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 保存權重
        weights_to_save = weights if weights is not None else self.state_dict()
        weights_path = os.path.join(save_dir, 'weights.pt')
        torch.save(weights_to_save, weights_path)

        # 2. 保存 TorchScript 模型 (用於部署)
        # 確保模型在評估模式下進行腳本化
        self.eval()
        model_scripted = torch.jit.script(self)
        model_scripted.save(os.path.join(save_dir, 'model.pt'))
        
        # 3. 保存元數據
        config = {
            "id": model_id,
            "name": project_name,
            "channels": self.channels,
            "date": datetime.utcnow().isoformat(), # 使用 ISO 8601 格式，更標準
            "datasets": datasets,
            "type": self.model_type.name,
            "epochs_trained": epochs,
            "learning_rate": learning_rate,
            "architecture": {
                "backbone": self.backbone_name,
                "model_class": self.__class__.__name__
            }
        }
        with open(os.path.join(save_dir, 'info.json'), 'w') as f:
            json.dump(config, f, indent=4)

        print(f"Model saved successfully with ID: {model_id}")
        refresh_model_list()

    @classmethod
    def load(cls, model_id: str, models_dir: str = './models') -> 'OsuAiModel':
        """加載一個已保存的模型。"""
        model_path = os.path.join(models_dir, model_id)
        config_path = os.path.join(model_path, 'info.json')
        weights_path = os.path.join(model_path, 'weights.pt')

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model files not found for ID {model_id} in {model_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 從配置中獲取參數來實例化模型
        model_channels = config.get('channels', CURRENT_STACK_NUM)
        model = cls(channels=model_channels) # 使用 cls 關鍵字，這樣子類調用時會創建正確的實例
        
        # 加載狀態字典
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print(f"Loaded {model.model_type.name} model '{config['name']}' from epoch {config.get('epochs_trained', 'N/A')}")
        return model

class AimNet(OsuAiModel):
    """預測滑鼠座標 (x, y) 的模型。"""
    def __init__(self, channels=CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Aim)
        # 使用一個簡單的線性層作為回歸頭
        self.backbone = get_timm_model(
            model_name=self.backbone_name, 
            out_features=2, # 輸出 (x, y)
            channels=self.channels
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 使用 sigmoid 將輸出限制在 [0, 1] 範圍內，以匹配標籤的正規化方式
        return torch.sigmoid(self.backbone(images))

class ActionsNet(OsuAiModel):
    """預測按鍵動作 (Idle, K1, K2) 的模型。"""
    def __init__(self, channels=CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Actions)
        self.backbone = get_timm_model(
            model_name=self.backbone_name,
            out_features=3, # 輸出 3 個類別的 logits
            channels=self.channels
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss 會在內部應用 softmax，所以這裡直接返回 logits
        return self.backbone(images)

class CombinedNet(OsuAiModel):
    """同時預測滑鼠座標和按鍵動作的模型。"""
    def __init__(self, channels=CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Combined)
        # 共享的主幹網絡
        self.backbone = timm.create_model(
            model_name=self.backbone_name,
            pretrained=False,
            in_chans=self.channels,
            num_classes=0  # 移除分類頭，我們將自己創建
        )
        num_bottleneck_features = self.backbone.num_features

        # 專門用於預測座標的頭
        self.aim_head = nn.Sequential(
            nn.Linear(num_bottleneck_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
        # 專門用於預測按鍵的頭 (兩個獨立的二元分類器)
        self.keys_head = nn.Sequential(
            nn.Linear(num_bottleneck_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2) # 輸出 k1, k2 的 logits
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 前向傳播通過共享的主幹
        features = self.backbone(images)
        
        # 分別通過各自的頭
        aim_output = torch.sigmoid(self.aim_head(features)) # (batch, 2)
        keys_output = self.keys_head(features) # (batch, 2)
        
        # 將結果合併為一個 tensor，以匹配標籤格式 (x, y, k1, k2)
        # 使用 sigmoid 將按鍵 logits 轉換為 [0, 1] 之間的機率
        return torch.cat([aim_output, torch.sigmoid(keys_output)], dim=1)