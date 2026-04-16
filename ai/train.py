# ai/train.py

import os
import copy
import time
import keyboard
import torch
import torch.nn as nn
import traceback
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ai.models import (
    ActionsNet,
    ActionsNetGRU,
    ActionsNet3D,
    ActionsNetTemporal,
    AimNet,
    AimNetGRU,
    AimNetHeatmap,
    AimNetHeatmapV2,
    CombinedNet,
    OsuAiModel,
)
from ai.dataset import OsuDatasetBuilder
from ai.utils import get_datasets, get_validated_input, get_models
from ai.constants import PYTORCH_DEVICE
from ai.enums import EModelType


is_training_paused = False


def _toggle_pause():
    global is_training_paused
    is_training_paused = not is_training_paused

    if is_training_paused:
        print("\n\n⏸️ [PAUSE] 訓練已暫停！(Training Paused) 按 F9 繼續。")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        print("\n\n▶️ [RESUME] 訓練繼續！(Training Resumed)")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


class DualHeadActionLoss(nn.Module):
    def __init__(self, onset_pos_weight=6.5, hold_pos_weight=1.2, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "pos_weight",
            torch.tensor([onset_pos_weight, hold_pos_weight], dtype=torch.float32),
        )

    def forward(self, logits, targets):
        targets = targets.float()

        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )


def _dual_head_exact_match(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = probs >= 0.5
    truth = labels >= 0.5
    return (preds == truth).all(dim=1).float().mean().item()


def _binary_prf_from_logits(logits: torch.Tensor, labels: torch.Tensor, head_idx: int, thresh: float = 0.5):
    probs = torch.sigmoid(logits[:, head_idx])
    preds = probs >= thresh
    truth = labels[:, head_idx] >= 0.5

    tp = (preds & truth).sum().item()
    fp = (preds & (~truth)).sum().item()
    fn = ((~preds) & truth).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def _simulate_action_runtime(
    onset_probs: torch.Tensor,
    hold_probs: torch.Tensor,
    onset_labels: torch.Tensor,
    hold_labels: torch.Tensor,
    onset_thresh: float,
    hold_exit_thresh: float,
    min_press_frames: int,
    slope_prob: float,
    slope_value: float,
):
    is_clicking = False
    click_start = -999999
    pred_onset = torch.zeros_like(onset_labels, dtype=torch.bool)
    pred_hold = torch.zeros_like(hold_labels, dtype=torch.bool)
    history: list[float] = []
    smooth_hold = 0.0  # EMA-smoothed hold，對齊 eval.py
    HOLD_EMA_ALPHA = 0.5

    for i in range(len(onset_probs)):
        op = float(onset_probs[i].item())
        hp_raw = float(hold_probs[i].item())
        smooth_hold = HOLD_EMA_ALPHA * smooth_hold + (1.0 - HOLD_EMA_ALPHA) * hp_raw
        hp = smooth_hold

        history.append(op)

        slope_trigger = False

        # 2-frame slope trigger，完全對齊 eval.py
        if len(history) >= 2:
            p1 = history[-2]
            p2 = history[-1]
            slope = p2 - p1

            if p2 >= slope_prob and slope > slope_value:
                slope_trigger = True

        if not is_clicking:
            if op > onset_thresh or slope_trigger:
                is_clicking = True
                click_start = i
                pred_onset[i] = True
                pred_hold[i] = True
        else:
            pred_hold[i] = True
            if hp < hold_exit_thresh and (i - click_start) >= min_press_frames:
                is_clicking = False

    onset_truth = onset_labels >= 0.5
    hold_truth = hold_labels >= 0.5

    onset_tp = (pred_onset & onset_truth).sum().item()
    onset_fp = (pred_onset & (~onset_truth)).sum().item()
    onset_fn = ((~pred_onset) & onset_truth).sum().item()

    hold_tp = (pred_hold & hold_truth).sum().item()
    hold_fp = (pred_hold & (~hold_truth)).sum().item()
    hold_fn = ((~pred_hold) & hold_truth).sum().item()

    onset_precision = onset_tp / (onset_tp + onset_fp) if (onset_tp + onset_fp) > 0 else 0.0
    onset_recall = onset_tp / (onset_tp + onset_fn) if (onset_tp + onset_fn) > 0 else 0.0
    onset_f1 = (
        2 * onset_precision * onset_recall / (onset_precision + onset_recall)
        if (onset_precision + onset_recall) > 0 else 0.0
    )
    hold_precision = hold_tp / (hold_tp + hold_fp) if (hold_tp + hold_fp) > 0 else 0.0
    hold_recall = hold_tp / (hold_tp + hold_fn) if (hold_tp + hold_fn) > 0 else 0.0
    hold_f1 = (
        2 * hold_precision * hold_recall / (hold_precision + hold_recall)
        if (hold_precision + hold_recall) > 0 else 0.0
    )
    return {
        "onset_precision": onset_precision, "onset_recall": onset_recall, "onset_f1": onset_f1,
        "hold_precision": hold_precision, "hold_recall": hold_recall, "hold_f1": hold_f1,
        "onset_fp": onset_fp, "onset_fn": onset_fn,
    }


def _find_best_action_runtime(all_logits: torch.Tensor, all_labels: torch.Tensor):
    probs = torch.sigmoid(all_logits)
    onset_probs = probs[:, 0].cpu()
    hold_probs = probs[:, 1].cpu()
    onset_labels = all_labels[:, 0].cpu()
    hold_labels = all_labels[:, 1].cpu()

    best_runtime = None
    best_score = -1e9

    for onset_thresh in [0.46, 0.48, 0.50, 0.52, 0.54]:
        for hold_exit_thresh in [0.16, 0.18, 0.20, 0.22]:
            for min_press_frames in [3, 4]:
                for slope_prob in [0.32, 0.34, 0.36, 0.38]:
                    for slope_value in [0.10, 0.12, 0.14, 0.16]:
                        m = _simulate_action_runtime(
                            onset_probs, hold_probs, onset_labels, hold_labels,
                            onset_thresh=onset_thresh, hold_exit_thresh=hold_exit_thresh,
                            min_press_frames=min_press_frames,
                            slope_prob=slope_prob, slope_value=slope_value,
                        )
                        score = (
                            2.2 * m["onset_precision"]
                            + 1.2 * m["onset_f1"]
                            + 0.6 * m["hold_f1"]
                            - 0.00035 * m["onset_fp"]
                            - 0.00015 * m["onset_fn"]
                        )
                        if score > best_score:
                            best_score = score
                            best_runtime = {
                                "onset_thresh": float(onset_thresh),
                                "hold_exit_thresh": float(hold_exit_thresh),
                                "min_press_ms": float(min_press_frames * 20.0),
                                "slope_prob": float(slope_prob),
                                "slope_value": float(slope_value),
                            }
    return best_runtime


def get_acc(predicted: torch.Tensor, truth: torch.Tensor, thresh: int = 60, is_combined: bool = False):
    predicted = predicted.detach().clone()
    truth = truth.detach().clone()

    predicted[:, 0] *= 1440
    predicted[:, 1] *= 1080
    truth[:, 0] *= 1440
    truth[:, 1] *= 1080

    if is_combined:
        diff = predicted[:, :2] - truth[:, :2]
    else:
        diff = predicted - truth

    dist = torch.sqrt((diff ** 2).sum(dim=1))
    dist_acc = (dist < thresh).float().mean().item()

    if not is_combined:
        return dist_acc

    predicted_keys = predicted[:, 2:]
    truth_keys = truth[:, 2:]
    key_acc = torch.all((predicted_keys >= 0.5) == (truth_keys >= 0.5), dim=1).float().mean().item()
    return (dist_acc + key_acc) / 2


def get_heatmap_acc(predicted_heatmaps: torch.Tensor, truth_coords: torch.Tensor, thresh: int = 60):
    from ai.eval import _extract_heatmap_coords

    batch = predicted_heatmaps.shape[0]
    acc_sum = 0

    for i in range(batch):
        pred_x, pred_y = _extract_heatmap_coords(predicted_heatmaps[i])
        true_x, true_y = truth_coords[i][0].item(), truth_coords[i][1].item()

        dist = ((pred_x * 1440 - true_x * 1440) ** 2 + (pred_y * 1080 - true_y * 1080) ** 2) ** 0.5
        if dist < thresh:
            acc_sum += 1

    return acc_sum / batch


def _train_model(
    project_name: str,
    datasets: list[str],
    model_type: EModelType,
    model_class: OsuAiModel,
    criterion: nn.Module,
    force_rebuild=False,
    checkpoint_model_id=None,
    batch_size=64,
    epochs=1,
    learning_rate=0.0001,
):
    writer = SummaryWriter(f"runs/{model_type.name.lower()}/{project_name}")

    try:
        keyboard.add_hotkey("f9", _toggle_pause, suppress=True)
        print("\n💡 提示：訓練期間可按 F9 暫停/繼續。")
    except Exception as e:
        print(f"\n⚠️ 無法註冊 F9 暫停熱鍵: {e}")

    print("Building and splitting dataset...")
    builder = OsuDatasetBuilder(datasets, model_type, force_rebuild)
    training_dataset, validation_dataset = builder.get_train_val_datasets(val_split=0.1, random_seed=42)

    workers = min(6, max(4, (os.cpu_count() or 8) // 2))
    val_workers = max(2, workers // 2)

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
    )

    model = model_class.load(checkpoint_model_id).to(PYTORCH_DEVICE) if checkpoint_model_id else model_class().to(PYTORCH_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10
    patience_count = 0
    best_runtime = None
    best_runtime_for_best_state = None  # 和 best_state 同一 epoch 的 runtime

    scaler = torch.amp.GradScaler("cuda")

    try:
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

            model.train()
            running_train_loss = 0.0
            train_acc_sum = 0
            train_steps = 0

            train_bar = tqdm(train_loader, desc="Training")
            for images, labels in train_bar:
                while is_training_paused:
                    time.sleep(1)

                images = images.to(PYTORCH_DEVICE, non_blocking=True)
                if model_type == EModelType.Actions:
                    labels = labels.to(PYTORCH_DEVICE, non_blocking=True).float()
                else:
                    labels = labels.to(PYTORCH_DEVICE, non_blocking=True)

                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if isinstance(model, (AimNetHeatmap, AimNetHeatmapV2)):
                        b = labels.shape[0]
                        h, w = 30, 40

                        grid_y, grid_x = torch.meshgrid(
                            torch.linspace(0, 1, h, device=PYTORCH_DEVICE),
                            torch.linspace(0, 1, w, device=PYTORCH_DEVICE),
                            indexing="ij",
                        )

                        target_coords = labels.view(b, 1, 1, 2)
                        dist_sq = (grid_x - target_coords[..., 0]) ** 2 + (grid_y - target_coords[..., 1]) ** 2
                        sigma = 0.05
                        heatmap_labels = torch.exp(-dist_sq / (2 * sigma ** 2)).unsqueeze(1)

                        outputs = model(images)

                        if isinstance(model, AimNetHeatmapV2):
                            dx_label = (target_coords[..., 0] - grid_x) * (w - 1)
                            dy_label = (target_coords[..., 1] - grid_y) * (h - 1)
                            offset_labels = torch.stack([dx_label, dy_label], dim=1)

                            bce_loss = nn.BCEWithLogitsLoss()(outputs[:, 0:1], heatmap_labels)
                            offset_loss = (
                                nn.SmoothL1Loss(reduction="none")(outputs[:, 1:3], offset_labels) * heatmap_labels
                            ).mean()
                            loss = bce_loss + 2.0 * offset_loss
                        else:
                            loss = criterion(outputs, heatmap_labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_train_loss += loss.item() * images.size(0)

                if model_type == EModelType.Actions:
                    batch_exact = ((torch.sigmoid(outputs) >= 0.5) == (labels >= 0.5)).all(dim=1).sum().item()
                    train_acc_sum += batch_exact
                    train_steps += labels.size(0)
                elif isinstance(model, (AimNetHeatmap, AimNetHeatmapV2)):
                    train_acc_sum += 0
                    train_steps += 1
                else:
                    train_acc_sum += get_acc(outputs, labels, is_combined=(model_type == EModelType.Combined))
                    train_steps += 1

            epoch_train_loss = running_train_loss / len(training_dataset)
            epoch_train_acc = train_acc_sum / train_steps if model_type == EModelType.Actions else (train_acc_sum / train_steps) * 100

            torch.cuda.empty_cache()

            model.eval()
            running_val_loss = 0.0
            val_acc_sum = 0
            val_steps = 0

            val_bar = tqdm(val_loader, desc="Validating")
            with torch.no_grad():
                for images, labels in val_bar:
                    while is_training_paused:
                        time.sleep(1)

                    images = images.to(PYTORCH_DEVICE, non_blocking=True)
                    if model_type == EModelType.Actions:
                        labels = labels.to(PYTORCH_DEVICE, non_blocking=True).float()
                    else:
                        labels = labels.to(PYTORCH_DEVICE, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        if isinstance(model, (AimNetHeatmap, AimNetHeatmapV2)):
                            b = labels.shape[0]
                            h, w = 30, 40

                            grid_y, grid_x = torch.meshgrid(
                                torch.linspace(0, 1, h, device=PYTORCH_DEVICE),
                                torch.linspace(0, 1, w, device=PYTORCH_DEVICE),
                                indexing="ij",
                            )

                            target_coords = labels.view(b, 1, 1, 2)
                            dist_sq = (grid_x - target_coords[..., 0]) ** 2 + (grid_y - target_coords[..., 1]) ** 2
                            sigma = 0.05
                            heatmap_labels = torch.exp(-dist_sq / (2 * sigma ** 2)).unsqueeze(1)

                            outputs = model(images)

                            if isinstance(model, AimNetHeatmapV2):
                                bce_loss = nn.BCEWithLogitsLoss()(outputs[:, 0:1], heatmap_labels)
                                dx_label = (target_coords[..., 0] - grid_x) * (w - 1)
                                dy_label = (target_coords[..., 1] - grid_y) * (h - 1)
                                offset_labels = torch.stack([dx_label, dy_label], dim=1)
                                offset_loss = (
                                    nn.SmoothL1Loss(reduction="none")(outputs[:, 1:3], offset_labels) * heatmap_labels
                                ).mean()
                                loss = bce_loss + 2.0 * offset_loss
                            else:
                                loss = criterion(outputs, heatmap_labels)
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)

                    running_val_loss += loss.item() * images.size(0)

                    if model_type == EModelType.Actions:
                        batch_exact = ((torch.sigmoid(outputs) >= 0.5) == (labels >= 0.5)).all(dim=1).sum().item()
                        val_acc_sum += batch_exact
                        val_steps += labels.size(0)
                    elif isinstance(model, (AimNetHeatmap, AimNetHeatmapV2)):
                        val_acc_sum += get_heatmap_acc(outputs, labels)
                        val_steps += 1
                    else:
                        val_acc_sum += get_acc(outputs, labels, is_combined=(model_type == EModelType.Combined))
                        val_steps += 1

            epoch_val_loss = running_val_loss / len(validation_dataset)
            epoch_val_acc = val_acc_sum / val_steps if model_type == EModelType.Actions else (val_acc_sum / val_steps) * 100

            print(
                f"Train Loss: {epoch_train_loss:.6f}, Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {epoch_val_loss:.6f}, Val Acc: {epoch_val_acc:.4f}"
            )

            if model_type == EModelType.Actions:
                all_logits = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(PYTORCH_DEVICE, non_blocking=True)
                        labels = labels.to(PYTORCH_DEVICE, non_blocking=True).float()
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model(images)
                        all_logits.append(out.float().cpu())
                        all_labels.append(labels.float().cpu())

                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                o_p, o_r, o_f1, o_tp, o_fp, o_fn = _binary_prf_from_logits(all_logits, all_labels, head_idx=0)
                h_p, h_r, h_f1, h_tp, h_fp, h_fn = _binary_prf_from_logits(all_logits, all_labels, head_idx=1)

                print(f" Onset → Precision: {o_p:.4f}, Recall: {o_r:.4f}, F1: {o_f1:.4f} (TP={o_tp}, FP={o_fp}, FN={o_fn})")
                print(f" Hold  → Precision: {h_p:.4f}, Recall: {h_r:.4f}, F1: {h_f1:.4f} (TP={h_tp}, FP={h_fp}, FN={h_fn})")

                writer.add_scalar("Onset/precision", o_p, epoch)
                writer.add_scalar("Onset/recall", o_r, epoch)
                writer.add_scalar("Onset/f1", o_f1, epoch)
                writer.add_scalar("Hold/precision", h_p, epoch)
                writer.add_scalar("Hold/recall", h_r, epoch)
                writer.add_scalar("Hold/f1", h_f1, epoch)

                best_runtime = _find_best_action_runtime(all_logits, all_labels)
                print(
                    "[Auto Runtime] "
                    f"onset_thresh={best_runtime['onset_thresh']:.2f}, "
                    f"hold_exit_thresh={best_runtime['hold_exit_thresh']:.2f}, "
                    f"min_press_ms={best_runtime['min_press_ms']:.1f}, "
                    f"slope_prob={best_runtime['slope_prob']:.2f}, "
                    f"slope_value={best_runtime['slope_value']:.2f}"
                )
            else:
                best_runtime = None

            scheduler.step(epoch_val_loss)
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_train_acc, epoch)
            writer.add_scalar("Loss/val", epoch_val_loss, epoch)
            writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

            if epoch_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.6f} to {epoch_val_loss:.6f}. Saving model...")
                best_val_loss = epoch_val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_runtime_for_best_state = copy.deepcopy(best_runtime)  # 鎖定此時的 runtime
                patience_count = 0
            else:
                patience_count += 1
                print(f"Validation loss did not improve. Patience: {patience_count}/{patience}")

            if patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

            if (epoch + 1) % 5 == 0:
                print(f"[Checkpoint] Saving periodic checkpoint at epoch {epoch + 1}...")
                model.save(
                    f"{project_name}_ckpt_ep{epoch + 1}",
                    datasets, epoch, learning_rate,
                    weights=best_state,
                    runtime_config=best_runtime_for_best_state if model_type == EModelType.Actions else None,
                )

            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining crashed: {e}")
    finally:
        print(f"\n=== Auto-saving best model (epoch {best_epoch + 1}, val_loss={best_val_loss:.6f}) ===")
        model.save(
            project_name,
            datasets, best_epoch, learning_rate,
            weights=best_state,
            runtime_config=best_runtime_for_best_state if model_type == EModelType.Actions else None,
        )
        print("Model saved successfully!")
        writer.close()


def train_action_net(**kwargs):
    print("Setting up training for ActionsNet (Dual-Head BCE)...")
    _train_model(
        model_type=EModelType.Actions,
        model_class=ActionsNet,
        criterion=DualHeadActionLoss(onset_pos_weight=10.0, hold_pos_weight=1.5, label_smoothing=0.05),
        **kwargs,
    )


def train_action_gru_net(**kwargs):
    print("Setting up training for ActionsNetGRU (Dual-Head BCE)...")
    if kwargs.get("batch_size", 128) > 64:
        kwargs["batch_size"] = 64
        print(" Batch size auto-reduced to 64 (10x per-frame CNN passes per sample)")
    _train_model(
        model_type=EModelType.Actions,
        model_class=ActionsNetGRU,
        criterion=DualHeadActionLoss(onset_pos_weight=10.0, hold_pos_weight=1.5, label_smoothing=0.05),
        **kwargs,
    )


def train_action_3d_net(**kwargs):
    print("Setting up training for ActionsNet3D (Dual-Head BCE)...")
    _train_model(
        model_type=EModelType.Actions,
        model_class=ActionsNet3D,
        criterion=DualHeadActionLoss(onset_pos_weight=10.0, hold_pos_weight=1.5, label_smoothing=0.05),
        **kwargs,
    )


def train_action_temporal_net(**kwargs):
    print("Setting up training for ActionsNetTemporal (Dual-Head BCE + SE Attention)...")
    if kwargs.get("batch_size", 128) > 64:
        kwargs["batch_size"] = 64
        print(" Batch size auto-reduced to 64 (10x per-frame CNN passes per sample)")
    _train_model(
        model_type=EModelType.Actions,
        model_class=ActionsNetTemporal,
        criterion=DualHeadActionLoss(onset_pos_weight=5.5, hold_pos_weight=1.2, label_smoothing=0.0),
        **kwargs,
    )


def train_aim_net(**kwargs):
    print("Setting up training for AimNet...")
    _train_model(
        model_type=EModelType.Aim,
        model_class=AimNet,
        criterion=nn.SmoothL1Loss(),
        **kwargs,
    )


def train_aim_gru_net(**kwargs):
    print("Setting up training for AimNetGRU (CNN + GRU 時序模型)...")
    if kwargs.get("batch_size", 128) > 64:
        kwargs["batch_size"] = 64
        print(" Batch size auto-reduced to 64 (10x per-frame CNN passes per sample)")
    _train_model(
        model_type=EModelType.Aim,
        model_class=AimNetGRU,
        criterion=nn.SmoothL1Loss(),
        **kwargs,
    )


def train_aim_heatmap_net_v2(**kwargs):
    print("Setting up training for AimNetHeatmapV2 (Heatmap + Offset + Attention)...")
    _train_model(
        model_type=EModelType.Aim,
        model_class=AimNetHeatmapV2,
        criterion=None,
        **kwargs,
    )


def train_aim_heatmap_net(**kwargs):
    print("Setting up training for AimNetHeatmap (Heatmap)...")
    _train_model(
        model_type=EModelType.Aim,
        model_class=AimNetHeatmap,
        criterion=nn.BCEWithLogitsLoss(),
        **kwargs,
    )


def train_combined_net(**kwargs):
    print("Setting up training for CombinedNet...")
    _train_model(
        model_type=EModelType.Combined,
        model_class=CombinedNet,
        criterion=nn.SmoothL1Loss(),
        **kwargs,
    )


def get_train_data(data_type: EModelType, all_datasets: list[str], models: list[dict]):
    project_name = get_validated_input(
        "What would you like to name this project? ",
        lambda s: len(s.strip()) > 0,
        lambda s: s.strip(),
    )

    dataset_prompt = "Please select datasets from below, separated by a comma (e.g., 0,2) or type 'all':\n"
    for i, ds_name in enumerate(all_datasets):
        dataset_prompt += f" [{i}] {ds_name}\n"

    def validate_datasets(s: str):
        if s.strip().lower() == "all":
            return True
        try:
            indices = [int(i.strip()) for i in s.split(",") if i.strip()]
            return all(0 <= i < len(all_datasets) for i in indices)
        except ValueError:
            return False

    def convert_datasets(s: str):
        if s.strip().lower() == "all":
            return list(range(len(all_datasets)))
        return [int(i.strip()) for i in s.split(",") if i.strip()]

    selected_indices = get_validated_input(dataset_prompt, validate_datasets, convert_datasets)
    selected_datasets = [all_datasets[i] for i in selected_indices]

    checkpoint_id = None
    if models:
        use_ckpt = get_validated_input(
            "Would you like to use a checkpoint? (y/n) ",
            lambda a: a.lower() in ["y", "n"],
            lambda a: a.lower(),
        )
        if use_ckpt.startswith("y"):
            models_prompt = "Please select a model to use as a checkpoint:\n"
            for i, model_info in enumerate(models):
                models_prompt += f"    [{i}] {model_info['name']}  ({model_info['date'].strftime('%Y-%m-%d %H:%M')})\n"

            def validate_model_idx(s: str):
                return s.strip().isdigit() and 0 <= int(s.strip()) < len(models)

            checkpoint_index = get_validated_input(models_prompt, validate_model_idx, lambda s: int(s.strip()))
            checkpoint_id = models[checkpoint_index]["id"]

    epochs = get_validated_input(
        "Max epochs to train for? ",
        lambda s: s.strip().isdigit() and int(s.strip()) > 0,
        lambda s: int(s.strip()),
    )

    return {
        "project_name": project_name,
        "datasets": selected_datasets,
        "checkpoint_model_id": checkpoint_id,
        "epochs": epochs,
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
[3] Train Aim (GRU Temporal)
[4] Train Aim (Heatmap)
[5] Train Aim (Heatmap V2 - Offset + Attention)
[6] Train Actions (GRU Temporal)
[7] Train Actions (3D CNN Temporal)
[8] Train Actions (1D Temporal Conv)
"""
        user_choice = get_validated_input(
            prompt,
            lambda a: a.strip().isdigit() and (0 <= int(a.strip()) <= 8),
            lambda a: int(a.strip()),
        )

        if user_choice == 0:
            model_type = EModelType.Aim
            train_fn = train_aim_net
        elif user_choice == 1:
            model_type = EModelType.Actions
            train_fn = train_action_temporal_net
        elif user_choice == 2:
            model_type = EModelType.Combined
            train_fn = train_combined_net
        elif user_choice == 3:
            model_type = EModelType.Aim
            train_fn = train_aim_gru_net
        elif user_choice == 4:
            model_type = EModelType.Aim
            train_fn = train_aim_heatmap_net
        elif user_choice == 5:
            model_type = EModelType.Aim
            train_fn = train_aim_heatmap_net_v2
        elif user_choice == 6:
            model_type = EModelType.Actions
            train_fn = train_action_gru_net
        elif user_choice == 7:
            model_type = EModelType.Actions
            train_fn = train_action_3d_net
        else:
            model_type = EModelType.Actions
            train_fn = train_action_temporal_net

        available_models = get_models(model_type)
        train_config = get_train_data(model_type, all_datasets, available_models)

        train_config["batch_size"] = 256
        train_config["learning_rate"] = 0.0003
        train_config["force_rebuild"] = False

        train_fn(**train_config)

    except Exception:
        print("\nAn error occurred during the training setup:")
        traceback.print_exc()