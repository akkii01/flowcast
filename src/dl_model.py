"""M6: Deep Learning Sequence Forecasting Engine (PyTorch LSTM & BiLSTM from scratch)."""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluate import (
    compute_multiclass_metrics,
    compute_regression_metrics,
    save_model_card,
)
from src.utils import load_config, set_global_seed, setup_logger

logger = setup_logger("M6_DeepLearning")

# Bounded vs Unbounded sequence features (PRD Section 10.1: Min-Max + Z-score)
BOUNDED_SEQ_COLS = ["occupancy", "visibility"]
UNBOUNDED_SEQ_COLS = [
    "traffic_volume",
    "avg_speed",
    "travel_time",
    "vc_ratio",
    "temperature",
    "rainfall",
]
PASSTHROUGH_SEQ_COLS = ["hour_sin", "hour_cos", "is_rain", "is_fog"]
ALL_SEQ_COLS = BOUNDED_SEQ_COLS + UNBOUNDED_SEQ_COLS + PASSTHROUGH_SEQ_COLS


class FlowCastSequenceNet(nn.Module):
    """Multi-task Recurrent Sequence Forecaster (LSTM or Bidirectional LSTM).

    PRD Section 12.1:
      - Input: Sliding window of previous N=12 steps per road segment + road embedding + context
      - Core: 2-layer LSTM (or BiLSTM) with inter-layer dropout + temporal attention pooling
      - Heads: Dual output heads for Volume Regression and 4-Class Congestion Classification
    """

    def __init__(
        self,
        seq_input_dim: int,
        ctx_input_dim: int,
        num_roads: int = 25,
        road_emb_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        rnn_out_dim = hidden_size * self.num_directions

        self.road_emb = nn.Embedding(num_roads, road_emb_dim)

        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Learned temporal attention over the N sequence steps
        self.attn_proj = nn.Linear(rnn_out_dim, 1)

        # Target-time known context branch (weather, calendar, seasonal lags, baselines)
        self.ctx_branch = nn.Sequential(
            nn.Linear(ctx_input_dim + road_emb_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion trunk combining last hidden state + attention context + known context
        fusion_dim = rnn_out_dim * 2 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(96, 48),
            nn.GELU(),
        )

        # Multi-task prediction heads
        self.volume_head = nn.Linear(48, 1)
        self.congestion_head = nn.Linear(48, 4)

    def forward(
        self, x_seq: torch.Tensor, x_ctx: torch.Tensor, road_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_out, _ = self.lstm(x_seq)  # (B, N, rnn_out_dim)
        last_hidden = rnn_out[:, -1, :]

        attn_weights = torch.softmax(self.attn_proj(rnn_out), dim=1)  # (B, N, 1)
        attn_pooled = torch.sum(attn_weights * rnn_out, dim=1)

        r_emb = self.road_emb(road_idx)
        ctx_feat = self.ctx_branch(torch.cat([x_ctx, r_emb], dim=1))

        fused = self.fusion(torch.cat([last_hidden, attn_pooled, ctx_feat], dim=1))
        vol_pred = self.volume_head(fused).squeeze(1)
        cong_logits = self.congestion_head(fused)
        return vol_pred, cong_logits


def prepare_sequence_tensors(
    clean_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    road_to_idx: Dict[str, int],
    seq_length: int,
) -> Dict:
    """Build leakage-free (N-step past sequence, context, road_id, targets) tensors.

    Guarantees exact 1-to-1 row alignment with M5 train/val/test splits.
    """
    logger.info(f"Building leakage-free {seq_length}-step sliding windows across 25 segments...")

    # Ensure deterministic sorting on both tables
    clean = clean_df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
    clean = clean.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    feat = feat_df.copy()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)
    feat = feat.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    # Derive sequence helper signals on clean timeline
    hour_frac = clean["timestamp"].dt.hour + clean["timestamp"].dt.minute / 60.0
    clean["hour_sin"] = np.sin(2.0 * np.pi * hour_frac / 24.0)
    clean["hour_cos"] = np.cos(2.0 * np.pi * hour_frac / 24.0)
    clean["vc_ratio"] = clean["traffic_volume"] / (clean["road_capacity"] / 2.0)
    clean["is_rain"] = ((clean["weather_condition"] == "Rain") | (clean["rainfall"] > 0.0)).astype(float)
    clean["is_fog"] = (clean["weather_condition"] == "Fog").astype(float)

    # Strictly fit sequence scalers on timestamps belonging to train period only
    train_end_ts = feat.loc[feat["split"] == "train", "timestamp"].max()
    train_mask_clean = clean["timestamp"] <= train_end_ts

    minmax_scaler = MinMaxScaler()
    zscore_scaler = StandardScaler()

    clean.loc[:, BOUNDED_SEQ_COLS] = minmax_scaler.fit_transform(
        clean.loc[train_mask_clean, BOUNDED_SEQ_COLS]
    ).mean(axis=0)  # placeholder init before full transform
    clean[BOUNDED_SEQ_COLS] = minmax_scaler.transform(clean[BOUNDED_SEQ_COLS])
    zscore_scaler.fit(clean.loc[train_mask_clean, UNBOUNDED_SEQ_COLS])
    clean[UNBOUNDED_SEQ_COLS] = zscore_scaler.transform(clean[UNBOUNDED_SEQ_COLS])

    # Reshape clean timeline into (num_roads, num_timesteps, num_seq_features)
    unique_ts = np.sort(clean["timestamp"].unique())
    ts_to_pos = {ts: i for i, ts in enumerate(unique_ts)}
    num_roads = len(road_to_idx)
    num_ts = len(unique_ts)
    num_seq_feats = len(ALL_SEQ_COLS)

    grid = np.zeros((num_roads, num_ts, num_seq_feats), dtype=np.float32)
    r_indices = clean["road_id"].map(road_to_idx).to_numpy(dtype=np.int64)
    t_indices = clean["timestamp"].map(ts_to_pos).to_numpy(dtype=np.int64)
    grid[r_indices, t_indices, :] = clean[ALL_SEQ_COLS].to_numpy(dtype=np.float32)

    # Fit context & target scalers strictly on train split
    train_df = feat[feat["split"] == "train"].reset_index(drop=True)
    val_df = feat[feat["split"] == "val"].reset_index(drop=True)
    test_df = feat[feat["split"] == "test"].reset_index(drop=True)

    ctx_scaler = StandardScaler()
    ctx_scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    y_mean = float(train_df["target_volume"].mean())
    y_std = float(train_df["target_volume"].std())

    def _slice_split(split_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        r_idx = split_df["road_id"].map(road_to_idx).to_numpy(dtype=np.int64)
        t_pos = split_df["timestamp"].map(ts_to_pos).to_numpy(dtype=np.int64)

        # Strictly past windows: [t - seq_length, ..., t - 1] (zero leakage)
        offsets = np.arange(-seq_length, 0, dtype=np.int64)  # [-12, ..., -1]
        win_t = t_pos[:, None] + offsets[None, :]  # (N_samples, seq_length)
        assert win_t.min() >= 0, "Insufficient history for sequence slicing!"
        assert np.all(win_t[:, -1] == t_pos - 1), "Leakage guard failed: last step != t-1!"

        X_seq = grid[r_idx[:, None], win_t, :]  # (N_samples, seq_length, num_seq_feats)
        X_ctx = ctx_scaler.transform(
            split_df[feature_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        y_vol = split_df["target_volume"].to_numpy(dtype=np.float32)
        y_vol_scaled = ((y_vol - y_mean) / y_std).astype(np.float32)
        y_cong = split_df["target_congestion_id"].to_numpy(dtype=np.int64)

        return {
            "X_seq": X_seq,
            "X_ctx": X_ctx,
            "road_idx": r_idx,
            "y_vol": y_vol,
            "y_vol_scaled": y_vol_scaled,
            "y_cong": y_cong,
        }

    data_bundle = {
        "train": _slice_split(train_df),
        "val": _slice_split(val_df),
        "test": _slice_split(test_df),
        "test_df_ref": test_df,
        "scalers": {
            "minmax_seq_scaler": minmax_scaler,
            "zscore_seq_scaler": zscore_scaler,
            "ctx_scaler": ctx_scaler,
            "y_mean": y_mean,
            "y_std": y_std,
            "seq_cols": ALL_SEQ_COLS,
            "ctx_cols": feature_cols,
            "seq_length": seq_length,
        },
    }
    logger.info(
        f"Sequence tensors ready -> Train: {data_bundle['train']['X_seq'].shape} | "
        f"Val: {data_bundle['val']['X_seq'].shape} | Test: {data_bundle['test']['X_seq'].shape}"
    )
    return data_bundle


def train_single_sequence_model(
    model_label: str,
    bidirectional: bool,
    bundle: Dict,
    dl_cfg: Dict,
    seed: int,
    device: torch.device,
) -> Dict:
    """Train one PyTorch LSTM/BiLSTM model with Early Stopping & LR scheduling."""
    set_global_seed(seed)
    tr, va, te = bundle["train"], bundle["val"], bundle["test"]
    y_mean, y_std = bundle["scalers"]["y_mean"], bundle["scalers"]["y_std"]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(tr["X_seq"]),
            torch.from_numpy(tr["X_ctx"]),
            torch.from_numpy(tr["road_idx"]),
            torch.from_numpy(tr["y_vol_scaled"]),
            torch.from_numpy(tr["y_cong"]),
        ),
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(va["X_seq"]),
            torch.from_numpy(va["X_ctx"]),
            torch.from_numpy(va["road_idx"]),
            torch.from_numpy(va["y_vol_scaled"]),
            torch.from_numpy(va["y_cong"]),
        ),
        batch_size=512,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(te["X_seq"]),
            torch.from_numpy(te["X_ctx"]),
            torch.from_numpy(te["road_idx"]),
        ),
        batch_size=512,
        shuffle=False,
    )

    model = FlowCastSequenceNet(
        seq_input_dim=tr["X_seq"].shape[2],
        ctx_input_dim=tr["X_ctx"].shape[1],
        num_roads=25,
        hidden_size=dl_cfg["hidden_size"],
        num_layers=dl_cfg["num_layers"],
        dropout=dl_cfg["dropout"],
        bidirectional=bidirectional,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=dl_cfg["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    vol_criterion = nn.MSELoss()
    cong_criterion = nn.CrossEntropyLoss()

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []

    logger.info(f"--- Training {model_label} on {device} (max {dl_cfg['epochs']} epochs) ---")
    for epoch in range(1, dl_cfg["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        n_tr_batches = 0

        for bx_seq, bx_ctx, b_road, by_vol, by_cong in train_loader:
            bx_seq = bx_seq.to(device)
            bx_ctx = bx_ctx.to(device)
            b_road = b_road.to(device)
            by_vol = by_vol.to(device)
            by_cong = by_cong.to(device)

            optimizer.zero_grad()
            pred_vol_sc, cong_logits = model(bx_seq, bx_ctx, b_road)
            loss = vol_criterion(pred_vol_sc, by_vol) + 0.12 * cong_criterion(cong_logits, by_cong)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_tr_batches += 1

        # Validation evaluation
        model.eval()
        val_loss_sum = 0.0
        n_va_batches = 0
        val_vol_preds = []
        val_cong_preds = []
        with torch.no_grad():
            for bx_seq, bx_ctx, b_road, by_vol, by_cong in val_loader:
                bx_seq = bx_seq.to(device)
                bx_ctx = bx_ctx.to(device)
                b_road = b_road.to(device)
                by_vol = by_vol.to(device)
                by_cong = by_cong.to(device)

                pred_vol_sc, cong_logits = model(bx_seq, bx_ctx, b_road)
                v_loss = vol_criterion(pred_vol_sc, by_vol) + 0.12 * cong_criterion(
                    cong_logits, by_cong
                )
                val_loss_sum += float(v_loss.item())
                n_va_batches += 1

                val_vol_preds.append((pred_vol_sc.cpu().numpy() * y_std) + y_mean)
                val_cong_preds.append(torch.argmax(cong_logits, dim=1).cpu().numpy())

        val_vol_preds_arr = np.clip(np.concatenate(val_vol_preds), 0.0, None)
        val_cong_preds_arr = np.concatenate(val_cong_preds)

        va_vol_m = compute_regression_metrics(va["y_vol"], val_vol_preds_arr)
        va_cong_m, _ = compute_multiclass_metrics(va["y_cong"], val_cong_preds_arr)
        train_loss = train_loss_sum / max(n_tr_batches, 1)
        val_loss = val_loss_sum / max(n_va_batches, 1)
        curr_lr = float(optimizer.param_groups[0]["lr"])

        scheduler.step(va_vol_m["RMSE"])

        history.append(
            {
                "model": model_label,
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_rmse": va_vol_m["RMSE"],
                "val_mape": va_vol_m["MAPE"],
                "val_macro_f1": va_cong_m["Macro_F1"],
                "lr": curr_lr,
            }
        )

        logger.info(
            f"[{model_label}] Epoch {epoch:02d}/{dl_cfg['epochs']} | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"ValRMSE={va_vol_m['RMSE']:.3f} | ValMAPE={va_vol_m['MAPE']:.2f}% | "
            f"ValMacroF1={va_cong_m['Macro_F1']:.4f} | LR={curr_lr:.5f}"
        )

        if va_vol_m["RMSE"] < best_val_rmse - 0.02:
            best_val_rmse = va_vol_m["RMSE"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= dl_cfg["patience"]:
                logger.info(
                    f"[{model_label}] Early stopping triggered at epoch {epoch} (best ValRMSE={best_val_rmse:.3f})."
                )
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Final validation residuals for 90% empirical confidence interval
    val_preds_best = []
    val_cong_best = []
    with torch.no_grad():
        for bx_seq, bx_ctx, b_road, _, _ in val_loader:
            p_sc, c_log = model(bx_seq.to(device), bx_ctx.to(device), b_road.to(device))
            val_preds_best.append((p_sc.cpu().numpy() * y_std) + y_mean)
            val_cong_best.append(torch.argmax(c_log, dim=1).cpu().numpy())

    val_preds_best = np.clip(np.concatenate(val_preds_best), 0.0, None)
    val_cong_best = np.concatenate(val_cong_best)
    val_q90_hw = float(np.percentile(np.abs(va["y_vol"] - val_preds_best), 90))
    final_val_vol_m = compute_regression_metrics(va["y_vol"], val_preds_best)
    final_val_cong_m, _ = compute_multiclass_metrics(va["y_cong"], val_cong_best)

    # Held-out test evaluation
    test_vol_preds = []
    test_cong_probs = []
    with torch.no_grad():
        for bx_seq, bx_ctx, b_road in test_loader:
            p_sc, c_log = model(bx_seq.to(device), bx_ctx.to(device), b_road.to(device))
            test_vol_preds.append((p_sc.cpu().numpy() * y_std) + y_mean)
            test_cong_probs.append(torch.softmax(c_log, dim=1).cpu().numpy())

    test_vol_preds = np.clip(np.concatenate(test_vol_preds), 0.0, None)
    test_cong_probs = np.concatenate(test_cong_probs)
    test_cong_preds = np.argmax(test_cong_probs, axis=1)
    test_cong_conf = np.max(test_cong_probs, axis=1)

    test_vol_m = compute_regression_metrics(te["y_vol"], test_vol_preds)
    test_cong_m, test_cong_cm = compute_multiclass_metrics(te["y_cong"], test_cong_preds)

    return {
        "model": model,
        "history": history,
        "val_vol_metrics": final_val_vol_m,
        "val_cong_metrics": final_val_cong_m,
        "test_vol_metrics": test_vol_m,
        "test_cong_metrics": test_cong_m,
        "test_cong_cm": test_cong_cm,
        "val_q90_halfwidth": val_q90_hw,
        "test_vol_preds": test_vol_preds,
        "test_cong_preds": test_cong_preds,
        "test_cong_conf": test_cong_conf,
    }


def run_deep_learning_pipeline(config_path: str = "config.yaml") -> pd.DataFrame:
    """Train LSTM and BiLSTM from scratch and run head-to-head benchmark vs M5."""
    start_time = time.time()
    config = load_config(config_path)
    seed = config["project"]["random_seed"]
    set_global_seed(seed)

    dl_cfg = config["deep_learning"]
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    clean_df = pd.read_parquet(config["paths"]["cleaned_merged"])
    feat_df = pd.read_parquet(config["paths"]["processed_features"])
    meta_path = Path(config["paths"]["processed_features"]).parent / "feature_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]
    road_to_idx = metadata["road_to_idx"]
    train_window = [metadata["splits"]["train"]["start"], metadata["splits"]["train"]["end"]]
    test_window = [metadata["splits"]["test"]["start"], metadata["splits"]["test"]["end"]]

    bundle = prepare_sequence_tensors(
        clean_df=clean_df,
        feat_df=feat_df,
        feature_cols=feature_cols,
        road_to_idx=road_to_idx,
        seq_length=dl_cfg["seq_length"],
    )
    joblib.dump(bundle["scalers"], models_dir / "dl_scalers.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train Primary Multi-Task LSTM
    lstm_res = train_single_sequence_model(
        model_label="LSTM (Multi-Task)",
        bidirectional=False,
        bundle=bundle,
        dl_cfg=dl_cfg,
        seed=seed,
        device=device,
    )
    torch.save(lstm_res["model"].state_dict(), models_dir / "lstm_sequence_model.pt")

    # 2. Train Comparison Bidirectional LSTM
    bilstm_res = train_single_sequence_model(
        model_label="BiLSTM (Multi-Task)",
        bidirectional=True,
        bundle=bundle,
        dl_cfg=dl_cfg,
        seed=seed + 1,
        device=device,
    )
    torch.save(bilstm_res["model"].state_dict(), models_dir / "bilstm_sequence_model.pt")

    # Save combined training curves (PRD Section 12.3)
    history_df = pd.DataFrame(lstm_res["history"] + bilstm_res["history"])
    history_df.to_csv(models_dir / "dl_training_history.csv", index=False)

    # Select winning DL model by Validation RMSE
    if bilstm_res["val_vol_metrics"]["RMSE"] < lstm_res["val_vol_metrics"]["RMSE"]:
        dl_winner_name = "BiLSTM (Multi-Task)"
        dl_winner = bilstm_res
    else:
        dl_winner_name = "LSTM (Multi-Task)"
        dl_winner = lstm_res

    # Save DL model cards (PRD Section 13.4)
    for m_id, label, res in [
        ("lstm_sequence_model", "PyTorch Multi-Task LSTM (From Scratch)", lstm_res),
        ("bilstm_sequence_model", "PyTorch Multi-Task BiLSTM (From Scratch)", bilstm_res),
    ]:
        save_model_card(
            model_name=m_id,
            task_name="Multi-Task Sequence Forecasting (Volume + Congestion)",
            algorithm=label,
            train_window=train_window,
            test_window=test_window,
            features=ALL_SEQ_COLS + feature_cols,
            hyperparameters={
                "seq_length": dl_cfg["seq_length"],
                "hidden_size": dl_cfg["hidden_size"],
                "num_layers": dl_cfg["num_layers"],
                "dropout": dl_cfg["dropout"],
                "learning_rate": dl_cfg["learning_rate"],
                "batch_size": dl_cfg["batch_size"],
                "val_q90_halfwidth": round(res["val_q90_halfwidth"], 3),
            },
            val_metrics={
                "volume": res["val_vol_metrics"],
                "congestion": res["val_cong_metrics"],
            },
            test_metrics={
                "volume": res["test_vol_metrics"],
                "congestion": res["test_cong_metrics"],
            },
            selection_rationale=f"Trained from scratch on {dl_cfg['seq_length']}-step temporal trajectories with Min-Max + Z-score scaling, dropout, early stopping, and LR plateau scheduling.",
            models_dir=str(models_dir),
        )

    # Save DL test predictions aligned with classical test predictions
    idx_to_cong = {0: "Free-flow", 1: "Moderate", 2: "Heavy", 3: "Severe"}
    test_ref = bundle["test_df_ref"]
    dl_preds_df = pd.DataFrame(
        {
            "road_id": test_ref["road_id"].values,
            "timestamp": test_ref["timestamp"].values,
            "actual_volume": bundle["test"]["y_vol"],
            "dl_pred_volume": np.round(dl_winner["test_vol_preds"], 2),
            "dl_vol_lower_90": np.round(
                np.clip(dl_winner["test_vol_preds"] - dl_winner["val_q90_halfwidth"], 0.0, None), 2
            ),
            "dl_vol_upper_90": np.round(
                dl_winner["test_vol_preds"] + dl_winner["val_q90_halfwidth"], 2
            ),
            "dl_pred_congestion": [idx_to_cong[i] for i in dl_winner["test_cong_preds"]],
            "dl_congestion_confidence": np.round(dl_winner["test_cong_conf"], 4),
            "lstm_pred_volume": np.round(lstm_res["test_vol_preds"], 2),
            "bilstm_pred_volume": np.round(bilstm_res["test_vol_preds"], 2),
            "dl_winner_model": dl_winner_name,
        }
    )
    dl_preds_df.to_parquet(models_dir / "dl_test_predictions.parquet", index=False)

    # Build Head-to-Head Classical vs Deep Learning Benchmark Table (PRD Section 12.4)
    classical_sb_path = models_dir / "classical_scoreboard.csv"
    benchmark_rows = []
    if classical_sb_path.exists():
        c_sb = pd.read_csv(classical_sb_path)
        vol_c = c_sb[c_sb["Task"] == "Traffic Volume"]
        cong_c = c_sb[c_sb["Task"] == "Congestion (4-Class)"]
        for _, row in vol_c.iterrows():
            benchmark_rows.append(
                {
                    "Family": "Classical ML",
                    "Task": "Traffic Volume",
                    "Model": row["Model"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "MAPE": row["MAPE"],
                    "R2": row["R2"],
                    "Macro_F1": np.nan,
                    "Accuracy": np.nan,
                }
            )
        for _, row in cong_c.iterrows():
            benchmark_rows.append(
                {
                    "Family": "Classical ML",
                    "Task": "Congestion (4-Class)",
                    "Model": row["Model"],
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "R2": np.nan,
                    "Macro_F1": row["Macro_F1"],
                    "Accuracy": row["Accuracy"],
                }
            )

    for label, res in [("LSTM (Multi-Task)", lstm_res), ("BiLSTM (Multi-Task)", bilstm_res)]:
        benchmark_rows.append(
            {
                "Family": "Deep Learning (PyTorch)",
                "Task": "Traffic Volume",
                "Model": label,
                "RMSE": res["test_vol_metrics"]["RMSE"],
                "MAE": res["test_vol_metrics"]["MAE"],
                "MAPE": res["test_vol_metrics"]["MAPE"],
                "R2": res["test_vol_metrics"]["R2"],
                "Macro_F1": np.nan,
                "Accuracy": np.nan,
            }
        )
        benchmark_rows.append(
            {
                "Family": "Deep Learning (PyTorch)",
                "Task": "Congestion (4-Class)",
                "Model": label,
                "RMSE": np.nan,
                "MAE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
                "Macro_F1": res["test_cong_metrics"]["Macro_F1"],
                "Accuracy": res["test_cong_metrics"]["Accuracy"],
            }
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(models_dir / "classical_vs_dl_benchmark.csv", index=False)

    # Check PRD Section 3.3 head-to-head criterion
    best_c_vol_rmse = float(
        benchmark_df.loc[
            (benchmark_df["Family"] == "Classical ML") & (benchmark_df["Task"] == "Traffic Volume"),
            "RMSE",
        ].min()
    )
    best_dl_vol_rmse = float(dl_winner["test_vol_metrics"]["RMSE"])
    beats_classical = best_dl_vol_rmse < best_c_vol_rmse

    elapsed = time.time() - start_time
    print("\n" + "=" * 98)
    print(" FLOWCAST M6: CLASSICAL ML vs DEEP LEARNING (PYTORCH) HEAD-TO-HEAD BENCHMARK (TEST SET)")
    print("=" * 98)
    print(benchmark_df.astype(object).fillna("-").to_string(index=False))
    print("=" * 98)
    print(f"Best Classical Volume RMSE : {best_c_vol_rmse:.4f}")
    print(
        f"Best DL Volume RMSE        : {best_dl_vol_rmse:.4f} ({dl_winner_name}) | "
        f"MAPE={dl_winner['test_vol_metrics']['MAPE']:.2f}% | R2={dl_winner['test_vol_metrics']['R2']:.4f}"
    )
    print(
        f"DL Congestion Performance  : Macro-F1={dl_winner['test_cong_metrics']['Macro_F1']:.4f} | "
        f"Accuracy={dl_winner['test_cong_metrics']['Accuracy']:.4f}"
    )
    print(f"DL Beats Best Classical?   : {'YES' if beats_classical else 'NO (See analysis)'}")
    print(f"Total M6 Training Time     : {elapsed:.1f}s")
    print("=" * 98 + "\n")

    logger.info("M6 Deep Learning Sequence Engine completed successfully.")
    return benchmark_df


if __name__ == "__main__":
    run_deep_learning_pipeline()