# ai/train.py

import copy
import torch
import torch.nn as nn
import traceback
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from ai.models import ActionsNet, AimNet, CombinedNet, OsuAiModel
from ai.dataset import OsuDatasetBuilder
from ai.utils import get_datasets, get_validated_input, get_models
from ai.constants import PYTORCH_DEVICE
from ai.enums import EModelType

@torch.jit.script
def get_acc(predicted: torch.Tensor, truth: torch.Tensor, thresh: int = 60, is_combined: bool = False):
    """
    計算 Aim 或 Combined 模型的準確率。
    """
    predicted = predicted.detach().clone()
    truth = truth.detach().clone()

    predicted[:, 0] *= 1920
    predicted[:, 1] *= 1080
    truth[:, 0] *= 1920
    truth[:, 1] *= 1080

    if is_combined:
        diff = predicted[:, :-2] - truth[:, :-2]
    else:
        diff = predicted - truth

    dist = torch.sqrt((diff ** 2).sum(dim=1))
    
    # 距離小於閾值的視為正確
    dist_acc = (dist < thresh).float().mean().item()

    if not is_combined:
        return dist_acc

    # 對於 Combined 模型，還需要計算按鍵的準確率
    predicted_keys = predicted[:, 2:]
    truth_keys = truth[:, 2:]

    predicted_keys[predicted_keys >= 0.5] = 1
    truth_keys[truth_keys >= 0.5] = 1
    predicted_keys[predicted_keys < 0.5] = 0
    truth_keys[truth_keys < 0.5] = 0

    key_acc = torch.all(predicted_keys == truth_keys, dim=1).float().mean().item()

    # 最終準確率是兩者的平均
    return (dist_acc + key_acc) / 2

def _train_model(project_name: str,
                 datasets: list[str],
                 model_type: EModelType,
                 model_class: OsuAiModel,
                 criterion: nn.Module,
                 force_rebuild=False,
                 checkpoint_model_id=None,
                 batch_size=64,
                 epochs=1,
                 learning_rate=0.0001):
    """
    一個通用的模型訓練函數，包含了完整的訓練、驗證、學習率調整和早停邏輯。
    """
    writer = SummaryWriter(f'runs/{model_type.name.lower()}/{project_name}')

    # 1. 數據加載和分割 (使用新的 DatasetBuilder)
    print("Building and splitting dataset...")
    builder = OsuDatasetBuilder(datasets, model_type, force_rebuild)
    training_dataset, validation_dataset = builder.get_train_val_datasets(val_split=0.1, random_seed=42)
    
    # 創建 DataLoader
    # num_workers > 0 可以利用多核心加載數據，pin_memory=True 在使用 GPU 時可以加速數據轉移
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 2. 模型、優化器、排程器初始化
    model = model_class.load(checkpoint_model_id).to(PYTORCH_DEVICE) if checkpoint_model_id else model_class().to(PYTORCH_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 30  # 早停的耐心值，應大於 scheduler 的耐心值
    patience_count = 0

    try:
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # --- 訓練階段 ---
            model.train()
            running_train_loss = 0.0
            train_acc_sum = 0
            train_steps = 0
            
            train_bar = tqdm(train_loader, desc="Training")
            for images, labels in train_bar:
                images = images.to(PYTORCH_DEVICE, non_blocking=True)
                labels = labels.to(PYTORCH_DEVICE, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * images.size(0)
                
                # 計算準確率
                if model_type == EModelType.Actions:
                    train_acc_sum += (outputs.argmax(1) == labels).sum().item()
                    train_steps += labels.size(0)
                else:
                    train_acc_sum += get_acc(outputs, labels, is_combined=(model_type == EModelType.Combined))
                    train_steps += 1 # get_acc 返回的是 batch 的平均值

            epoch_train_loss = running_train_loss / len(training_dataset)
            epoch_train_acc = (train_acc_sum / train_steps if model_type == EModelType.Actions else (train_acc_sum / train_steps) * 100)

            # --- 驗證階段 ---
            model.eval()
            running_val_loss = 0.0
            val_acc_sum = 0
            val_steps = 0
            
            val_bar = tqdm(val_loader, desc="Validating")
            with torch.no_grad():
                for images, labels in val_bar:
                    images = images.to(PYTORCH_DEVICE, non_blocking=True)
                    labels = labels.to(PYTORCH_DEVICE, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    running_val_loss += loss.item() * images.size(0)
                    
                    if model_type == EModelType.Actions:
                        val_acc_sum += (outputs.argmax(1) == labels).sum().item()
                        val_steps += labels.size(0)
                    else:
                        val_acc_sum += get_acc(outputs, labels, is_combined=(model_type == EModelType.Combined))
                        val_steps += 1
            
            epoch_val_loss = running_val_loss / len(validation_dataset)
            epoch_val_acc = (val_acc_sum / val_steps if model_type == EModelType.Actions else (val_acc_sum / val_steps) * 100)

            print(f"Train Loss: {epoch_train_loss:.6f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.6f}, Val Acc: {epoch_val_acc:.4f}")

            # --- 更新 Scheduler & Tensorboard ---
            scheduler.step(epoch_val_loss)
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            # --- 早停與模型保存 ---
            if epoch_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.6f} to {epoch_val_loss:.6f}. Saving model...")
                best_val_loss = epoch_val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1
                print(f"Validation loss did not improve. Patience: {patience_count}/{patience}")

            if patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if get_validated_input("Would you like to save the best model found so far?\n", lambda a: True, lambda a: a.strip().lower()).startswith("y"):
            print(f"Saving model from epoch {best_epoch+1} with validation loss {best_val_loss:.6f}.")
            model.save(project_name, datasets, best_epoch, learning_rate, weights=best_state)
        writer.close()

def train_action_net(**kwargs):
    print("Setting up training for ActionsNet...")
    _train_model(model_type=EModelType.Actions,
                 model_class=ActionsNet,
                 criterion=nn.CrossEntropyLoss(),
                 **kwargs)

def train_aim_net(**kwargs):
    print("Setting up training for AimNet...")
    _train_model(model_type=EModelType.Aim,
                 model_class=AimNet,
                 criterion=nn.MSELoss(),
                 **kwargs)

def train_combined_net(**kwargs):
    print("Setting up training for CombinedNet...")
    _train_model(model_type=EModelType.Combined,
                 model_class=CombinedNet,
                 criterion=nn.MSELoss(),
                 **kwargs)

def get_train_data(data_type: EModelType, all_datasets: list[str], models: list[dict]):
    """從用戶獲取訓練任務的配置。"""
    project_name = get_validated_input("What would you like to name this project? ")
    
    # 選擇數據集
    dataset_prompt = "Please select datasets from below, separated by a comma (e.g., 0,2):\n"
    for i, ds_name in enumerate(all_datasets):
        dataset_prompt += f"    [{i}] {ds_name}\n"
    
    def validate_datasets(s: str):
        try:
            indices = [int(i.strip()) for i in s.split(',') if i.strip()]
            return all(0 <= i < len(all_datasets) for i in indices)
        except ValueError:
            return False
            
    selected_indices = get_validated_input(dataset_prompt, validate_datasets, lambda s: [int(i.strip()) for i in s.split(',') if i.strip()])
    selected_datasets = [all_datasets[i] for i in selected_indices]
    
    # 選擇 Checkpoint
    checkpoint_id = None
    if models:
        if get_validated_input("Would you like to use a checkpoint? (y/n) ", lambda a: a.lower() in ['y', 'n'], lambda a: a.lower()).startswith("y"):
            models_prompt = "Please select a model to use as a checkpoint:\n"
            for i, model_info in enumerate(models):
                models_prompt += f"    [{i}] {model_info['name']} ({model_info['id'][:8]}...)\n"
            
            def validate_model_idx(s: str):
                return s.strip().isdigit() and 0 <= int(s.strip()) < len(models)

            checkpoint_index = get_validated_input(models_prompt, validate_model_idx, lambda s: int(s.strip()))
            checkpoint_id = models[checkpoint_index]['id']

    epochs = get_validated_input("Max epochs to train for? ", lambda s: s.strip().isdigit() and int(s.strip()) > 0, lambda s: int(s.strip()))
    
    return {
        "project_name": project_name,
        "datasets": selected_datasets,
        "checkpoint_model_id": checkpoint_id,
        "epochs": epochs
    }

def start_train():
    try:
        all_datasets = get_datasets()
        if not all_datasets:
            print("No datasets found in 'data/raw'. Please run the converter first.")
            return

        prompt = """What type of training would you like to do?
    [0] Train Aim 
    [1] Train Actions
    [2] Train Combined
"""
        user_choice = get_validated_input(prompt, lambda a: a.strip().isdigit() and (0 <= int(a.strip()) <= 2), lambda a: int(a.strip()))

        if user_choice == 0:
            model_type = EModelType.Aim
            train_fn = train_aim_net
        elif user_choice == 1:
            model_type = EModelType.Actions
            train_fn = train_action_net
        else:
            model_type = EModelType.Combined
            train_fn = train_combined_net

        available_models = get_models(model_type)
        train_config = get_train_data(model_type, all_datasets, available_models)
        
        # 這裡可以添加更多配置，如 batch_size, learning_rate 等
        train_config['batch_size'] = 64
        train_config['learning_rate'] = 0.0001
        train_config['force_rebuild'] = False # 默認不強制重建

        train_fn(**train_config)

    except Exception:
        print("\nAn error occurred during the training setup:")
        traceback.print_exc()