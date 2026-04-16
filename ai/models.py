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

class CoordAtt(nn.Module):
    """
    Coordinate Attention: 考慮了水平和垂直方向的空間信息，
    能幫助模型更精確地定位圓圈中心。
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 移除 AdaptiveAvgPool2d，因為它在 TorchScript 下不支援 (None, 1) 的 Tuple 寫法
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 使用 torch.mean 來模擬 AdaptiveAvgPool2d((h, 1)) 和 ((1, w))
        # 這對 TorchScript 更加親和且效能一致
        x_h = torch.mean(x, dim=3, keepdim=True) # (n, c, h, 1)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2) # (n, c, 1, w) -> (n, c, w, 1)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

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
        self.backbone_name = "efficientnet_b2" # 選擇一個特徵提取更強的模型

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基類中的 forward 應該被子類重寫
        raise NotImplementedError("Each model must implement its own forward pass.")

    def save(self, project_name: str, datasets: list[str], epochs: int, learning_rate: float, path: str = './models', weights: Optional[dict] = None, runtime_config: Optional[dict] = None):
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
            },
            "runtime": runtime_config or {
                "onset_thresh": 0.52,
                "hold_exit_thresh": 0.20,
                "min_press_ms": 32.0,
                "slope_prob": 0.36,
                "slope_value": 0.14
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

class AimNetGRU(OsuAiModel):
    """
    帶有 GRU 時序記憶的 Aim 模型。
    
    與 AimNet 的差異：
    - AimNet 把 10 幀疊成 10 個 channel → CNN 把時間當顏色處理，不理解先後順序
    - AimNetGRU 把每幀單獨送入共享 CNN → 得到 10 個特徵向量 → GRU 按時間順序處理
    
    GRU 能學會的時序模式：
    - 「Approach Circle 正在縮小 → 即將要點擊 → 準備移向下一個圈圈」
    - 「游標正在沿 Slider 軌跡移動 → 繼續跟蹤」
    - 「大跳前需要提前加速」
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Aim)
        self.backbone_name = "mobilenetv3_small_100"
        
        # 共享的逐幀特徵提取器（輕量級，每幀獨立處理）
        self.frame_encoder = timm.create_model(
            self.backbone_name, pretrained=False, in_chans=1, num_classes=0
        )
        # mobilenetv3 的 num_features 屬性與實際輸出維度不一致，用 dummy forward 取得真實維度
        self.frame_encoder.eval()  # BatchNorm 在 train 模式下不接受 batch_size=1
        with torch.no_grad():
            feat_dim = self.frame_encoder(torch.zeros(1, 1, 8, 8)).shape[-1]
        self.frame_encoder.train()
        
        # GRU 時序建模（將 10 個特徵向量視為一段時間序列）
        self.gru = nn.GRU(feat_dim, 128, num_layers=1, batch_first=True)
        
        # 座標預測頭
        self.head = nn.Linear(128, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        輸入: (batch, num_frames, H, W) — 與 AimNet 完全相同的格式
        輸出: (batch, 2) — 預測的 (x%, y%) 座標
        """
        batch_size = images.shape[0]
        num_frames = images.shape[1]
        
        # 1. 把所有幀攤平成一個大 batch，一次通過 CNN
        frames = images.reshape(batch_size * num_frames, 1, images.shape[2], images.shape[3])
        features = self.frame_encoder(frames)  # (B*10, feat_dim)
        
        # 2. 重新組合成時間序列
        features = features.reshape(batch_size, num_frames, -1)
        
        # 3. GRU 按時間順序處理特徵序列
        gru_out, _ = self.gru(features)  # (B, 10, 128)
        
        # 4. 取最後一個時間步的輸出
        last_output = gru_out[:, -1, :]  # (B, 128)
        
        # 5. 預測座標
        return torch.sigmoid(self.head(last_output))  # (B, 2)

class AimNetHeatmap(OsuAiModel):
    """
    熱點預測 (Heatmap Prediction) 模型。
    不直接回歸座標，而是輸出 30x40 的空間機率圖。
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Aim)
        # 1. 強大的特徵提取骨幹
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=False, in_chans=self.channels, num_classes=0
        )
        num_features = self.backbone.num_features # B2 是 1408
        
        # 2. 上採樣解碼器 (Decoding Head)
        # 把 1408 維度的向量投影回空間網格，並逐步上採樣到 30x40
        self.decoder_input = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 4 * 5) # 初始小尺寸 4x5
        )
        
        self.decoder = nn.Sequential(
            # 4x5 -> 8x10
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 8x10 -> 16x20
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 16x20 -> 30x40 (最終輸出熱點)
            # 使用 Upsample + Conv 避免 Checkerboard 效應
            nn.Upsample(size=(30, 40), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        輸入: (batch, 10, 120, 160)
        輸出: (batch, 1, 30, 40) 的 Heatmap Logits
        """
        # 1. 提取全域特徵
        features = self.backbone(images) # (B, 1408)
        
        # 2. 映射回空間維度
        x = self.decoder_input(features) # (B, 128*4*5)
        x = x.view(-1, 128, 4, 5)        # (B, 128, 4, 5)
        
        # 3. 上採樣重建熱點
        heatmap = self.decoder(x)        # (B, 1, 30, 40)
        
        return heatmap # 注意：輸出是 Logits，訓練時配合 BCEWithLogitsLoss 使用

class AimNetHeatmapV2(OsuAiModel):
    """
    熱點預測 V2：Heatmap + Offset Regression + Coordinate Attention。
    輸出 3 個 channel：
    - ch0: Heatmap (物體中心機率)
    - ch1: X-Offset (相對於網格中心的偏移 [-0.5, 0.5])
    - ch2: Y-Offset (相對於網格中心的偏移 [-0.5, 0.5])
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Aim)
        # 1. 骨幹
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=False, in_chans=self.channels, num_classes=0
        )
        num_features = self.backbone.num_features
        
        # 2. 空間注意力強化 (在進入解碼器前)
        self.attention = CoordAtt(128, 128)
        
        # 3. 解碼器
        self.decoder_input = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 4 * 5)
        )
        
        self.decoder = nn.Sequential(
            # 4x5 -> 8x10
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 8x10 -> 16x20
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 16x20 -> 30x40
            nn.Upsample(size=(30, 40), mode='bilinear', align_corners=False),
            # 最終輸出 3 個通道：Heatmap, Offset_X, Offset_Y
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        x = self.decoder_input(features)
        x = x.view(-1, 128, 4, 5)
        
        # 套用 Coordinate Attention
        x = self.attention(x)
        
        # 上採樣
        out = self.decoder(x)
        return out 

class ActionsNet(OsuAiModel):
    """預測按鍵動作 (Idle, K1, K2) 的模型。含 Dropout 正則化。"""
    def __init__(self, channels=CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Actions)
        # 提取特徵（不含分類頭）
        self.backbone = timm.create_model(
            model_name=self.backbone_name,
            pretrained=False,
            in_chans=self.channels,
            num_classes=0  # 移除分類頭
        )
        num_features = self.backbone.num_features
        # Dropout + 分類頭
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(num_features, 2)  # onset_logit, hold_logit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.dropout(features)
        return self.head(features)

class ActionsNetGRU(OsuAiModel):
    """
    帶有 GRU 時序記憶的 Actions 模型。
    
    與 ActionsNet 的差異：
    - ActionsNet 把 10 幀疊成 10 個 channel → CNN 不理解時間先後順序
    - ActionsNetGRU 逐幀送入 CNN → GRU 按時間順序處理 → 能學會節奏
    
    GRU 能學會的時序模式：
    - 「Approach Circle 正在縮小 → 現在該點擊了」
    - 「剛點完一個圈 → 接下來是 Idle」
    - 「連打節奏 → K1 K2 交替」
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Actions)
        self.backbone_name = "mobilenetv3_small_100"
        
        # 共享的逐幀特徵提取器（輕量級，每幀獨立處理）
        self.frame_encoder = timm.create_model(
            self.backbone_name, pretrained=False, in_chans=1, num_classes=0
        )
        # mobilenetv3 的 num_features 屬性與實際輸出維度不一致，用 dummy forward 取得真實維度
        self.frame_encoder.eval()  # BatchNorm 在 train 模式下不接受 batch_size=1
        with torch.no_grad():
            feat_dim = self.frame_encoder(torch.zeros(1, 1, 8, 8)).shape[-1]
        self.frame_encoder.train()
        
        # GRU 時序建模（2 層 GRU + 內建 Dropout）
        self.gru = nn.GRU(feat_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Dropout + 分類頭
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(128, 2)  # 輸出 Idle/Click logits (二分類)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        輸入: (batch, num_frames, H, W) — 與 AimNetGRU 完全相同的格式
        輸出: (batch, 3) — 預測的 Idle/K1/K2 logits
        """
        batch_size = images.shape[0]
        num_frames = images.shape[1]
        
        # 1. 把所有幀攤平成一個大 batch，一次通過 CNN
        frames = images.reshape(batch_size * num_frames, 1, images.shape[2], images.shape[3])
        features = self.frame_encoder(frames)  # (B*10, feat_dim)
        
        # 2. 重新組合成時間序列
        features = features.reshape(batch_size, num_frames, -1)
        
        # 3. GRU 按時間順序處理特徵序列
        gru_out, _ = self.gru(features)  # (B, 10, 128)
        
        # 4. 取最後一個時間步的輸出
        last_output = gru_out[:, -1, :]  # (B, 128)
        
        # 5. Dropout + 分類
        last_output = self.dropout(last_output)
        return self.head(last_output)  # (B, 3) — logits, CrossEntropyLoss 內部會 softmax

class ActionsNet3D(OsuAiModel):
    """
    原生 3D CNN (Three-dimensional Convolutional Network) 模型。
    
    專為處理「動態軌跡」設計，卷積核同時在 X、Y 空間軸以及 T 時間軸上滑動：
    - 能直接捕捉 Approach Circle 隨著時間逐漸縮小的物理過程。
    - 比 GRU 速度快非常多（使用底層 C++ 3D Conv 加速，不需迴圈）。
    - 輸入形狀需求：(Batch, Channel=1, Depth=10, H, W)
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Actions)
        
        # 輸入形狀: (B, 1, 10, 120, 160)
        self.features = nn.Sequential(
            # --- 終極加速核心 ---
            # 點擊判讀不需要看超高畫質，一開始直接在空間上縮小一半 (120x160 -> 60x80)
            # 保留完整的 10 幀時間 (t=10)，但這一步直接把全體 VRAM 和計算量砍掉 75%
            nn.AvgPool3d(kernel_size=(1, 2, 2)), 
            
            # Conv1: 輕量級 16 通道
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # -> (B, 16, 10, 30, 40)
            
            # Conv2: 32 通道
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # 時間減半 -> (B, 32, 5, 15, 20)
            
            # Conv3: 64 通道
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # 時間減半 -> (B, 64, 2, 7, 10)
            
            # Conv4: 128 通道
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1) # -> (B, 128, 1, 1, 1)
        )
        
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(128, 2)  # onset_logit, hold_logit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images 原本形狀: (batch, 10, H, W)
        3D CNN 需要: (batch, Channel, Depth, H, W)
        所以我們擴展通道維度: (batch, 1, 10, H, W)
        """
        # unsqueeze(1) 在通道維度(C)增加維度 1，變成 (B, 1, T, H, W)
        x = images.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten -> (B, 256)
        x = self.dropout(x)
        return self.head(x)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.pad,
            dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.pad > 0:
            x = x[:, :, :-self.pad]
        return x


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation Block (1D)。
    讓模型自動學會哪些 feature channel 對 onset 最重要，
    適應性地重新權重各通道。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=2)  # Global Average Pooling over time -> (B, C)
        w = self.fc(w).unsqueeze(2)  # (B, C, 1)
        return x * w


class ActionsNetTemporal(OsuAiModel):
    """
    強化版 1D Temporal Conv 模型。
    
    改進：
    - TCN dilation 從 [1,2,4] 擴大到 [1,2,4,8,16]，receptive field 從 150ms → 630ms
    - 加入 SE Attention 讓模型自動學習哪些 channel 對 onset 最重要
    - Dropout 0.15 → 0.2 減少過擬合
    """
    def __init__(self, channels: int = CURRENT_STACK_NUM):
        super().__init__(channels, EModelType.Actions)
        self.backbone_name = "mobilenetv3_small_100"

        self.frame_encoder = timm.create_model(
            self.backbone_name,
            pretrained=False,
            in_chans=1,
            num_classes=0
        )

        self.frame_encoder.eval()
        with torch.no_grad():
            feat_dim = self.frame_encoder(torch.zeros(1, 1, 8, 8)).shape[-1]
        self.frame_encoder.train()

        # 擴大 receptive field: dilation [1,2,4,8,16] → RF = 1 + 2*(1+2+4+8+16) = 63 frames
        # 在 10 幀輸入下，高 dilation 層會自動在可用幀內重複利用特徵，強化時序建模
        self.temporal = nn.Sequential(
            TemporalResidualBlock(feat_dim, 128, kernel_size=3, dilation=1, dropout=0.2),
            TemporalResidualBlock(128, 128, kernel_size=3, dilation=2, dropout=0.2),
            TemporalResidualBlock(128, 128, kernel_size=3, dilation=4, dropout=0.2),
            TemporalResidualBlock(128, 128, kernel_size=3, dilation=8, dropout=0.2),
            TemporalResidualBlock(128, 128, kernel_size=3, dilation=16, dropout=0.2),
        )

        # SE Attention: 讓模型自動學習通道重要性
        self.se = SEBlock1d(128, reduction=4)

        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)   # [onset_logit, hold_logit]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        num_frames = images.shape[1]

        frames = images.reshape(batch_size * num_frames, 1, images.shape[2], images.shape[3])
        features = self.frame_encoder(frames)                         # (B*T, F)
        features = features.reshape(batch_size, num_frames, -1)      # (B, T, F)
        features = features.permute(0, 2, 1)                         # (B, F, T)

        temporal_out = self.temporal(features)                       # (B, 128, T)
        temporal_out = self.se(temporal_out)                         # SE Attention 重新權重通道
        last = temporal_out[:, :, -3:].mean(dim=2)
        return self.head(last)                                       # (B, 2)

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