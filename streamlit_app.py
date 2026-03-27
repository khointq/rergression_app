from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer


# ==============================
# Cau hinh duong dan artifact
# ==============================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_SUBDIR = Path("deep_multimodal_price")
ARTIFACT_DIR_CANDIDATES = [
    BASE_DIR / "models" / ARTIFACT_SUBDIR,
    BASE_DIR / "notebooks" / "models" / ARTIFACT_SUBDIR,
]
PHOBERT_DIR_CANDIDATES = [
    BASE_DIR / "models" / "phobert-base",
    BASE_DIR / "notebooks" / "models" / "phobert-base",
]
MODEL_WEIGHTS_NAME = "best_model_state.pt"
MODEL_META_NAME = "inference_meta.pkl"
DEFAULT_REPO_ID = "vinai/phobert-base"


def inject_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

        :root {
            --brand: #ef6a5b;
            --brand-soft: #fff1ed;
            --ink: #1f2430;
            --muted: #5e6470;
            --card-border: rgba(36, 42, 58, 0.10);
            --card-shadow: 0 14px 32px rgba(17, 24, 39, 0.08);
        }

        html, body, [class*="css"] {
            font-family: 'Be Vietnam Pro', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 95% 5%, rgba(255, 196, 138, 0.18), transparent 35%),
                radial-gradient(circle at 5% 95%, rgba(239, 106, 91, 0.10), transparent 30%);
        }

        .app-hero {
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 18px 22px;
            background: linear-gradient(120deg, #fff7f4 0%, #ffffff 100%);
            box-shadow: var(--card-shadow);
            margin-bottom: 18px;
        }

        .app-hero h2 {
            margin: 0 0 6px 0;
            color: var(--ink);
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .app-hero p {
            margin: 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .section-title {
            margin: 8px 0 12px 0;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--ink);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
            border-color: var(--card-border);
            box-shadow: var(--card-shadow);
            background: rgba(255, 255, 255, 0.78);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 8px 12px;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            min-height: 48px;
            border: none;
            background: linear-gradient(120deg, #ef6a5b, #e05549);
            color: white;
            box-shadow: 0 10px 20px rgba(224, 85, 73, 0.25);
        }

        .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        section[data-testid="stSidebar"] .stRadio > div {
            gap: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_existing_file(base_dirs, filename):
    for base_dir in base_dirs:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return None


def get_artifact_paths():
    weights_path = find_existing_file(ARTIFACT_DIR_CANDIDATES, MODEL_WEIGHTS_NAME)
    meta_path = find_existing_file(ARTIFACT_DIR_CANDIDATES, MODEL_META_NAME)
    return weights_path, meta_path


def get_phobert_dir():
    for candidate in PHOBERT_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def load_phobert_assets():
    phobert_dir = get_phobert_dir()
    model_name = str(phobert_dir) if phobert_dir else DEFAULT_REPO_ID
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    text_backbone = AutoModel.from_pretrained(model_name)
    return tokenizer, text_backbone


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class MultiModalPriceModel(nn.Module):
    def __init__(self, text_backbone, num_tabular_features, train_last_n_layers=4):
        super().__init__()
        self.text_backbone = text_backbone
        hidden_size = self.text_backbone.config.hidden_size

        for param in self.text_backbone.parameters():
            param.requires_grad = False

        if hasattr(self.text_backbone, "encoder") and hasattr(self.text_backbone.encoder, "layer"):
            for layer in self.text_backbone.encoder.layer[-train_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            if hasattr(self.text_backbone, "pooler") and self.text_backbone.pooler is not None:
                for param in self.text_backbone.pooler.parameters():
                    param.requires_grad = True

        self.text_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.25),
            ResidualMLPBlock(384, dropout=0.2),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.tabular_stem = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.tabular_block1 = ResidualMLPBlock(128, dropout=0.15)
        self.tabular_block2 = ResidualMLPBlock(128, dropout=0.15)
        self.tabular_projection = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(256 + 96, 128),
            nn.GELU(),
            nn.Linear(128, 256 + 96),
            nn.Sigmoid(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256 + 96, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualMLPBlock(192, dropout=0.2),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def mean_pool(self, last_hidden_state, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = last_hidden_state * expanded_mask
        pooled = masked_hidden.sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def forward(self, input_ids, attention_mask, tabular):
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]
        mean_embedding = self.mean_pool(text_outputs.last_hidden_state, attention_mask)
        text_features = self.text_head(torch.cat([cls_embedding, mean_embedding], dim=1))

        tabular_features = self.tabular_stem(tabular)
        tabular_features = self.tabular_block1(tabular_features)
        tabular_features = self.tabular_block2(tabular_features)
        tabular_features = self.tabular_projection(tabular_features)

        combined = torch.cat([text_features, tabular_features], dim=1)
        gated = combined * self.gate(combined)
        return self.regressor(gated)


@st.cache_resource(show_spinner=False)
def load_saved_price_model(weights_path=None, meta_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights_path is None or meta_path is None:
        auto_weights_path, auto_meta_path = get_artifact_paths()
        weights_path = weights_path or auto_weights_path
        meta_path = meta_path or auto_meta_path

    if weights_path is None or meta_path is None:
        searched_dirs = "\n".join([f"- {d}" for d in ARTIFACT_DIR_CANDIDATES])
        raise FileNotFoundError(
            "Khong tim thay artifact model. Hay chay cell train trong notebook truoc de tao file luu.\n"
            f"Da tim trong cac thu muc:\n{searched_dirs}"
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    tokenizer, text_backbone = load_phobert_assets()
    model = MultiModalPriceModel(
        text_backbone,
        len(meta["numeric_cols"]),
        train_last_n_layers=meta.get("train_last_n_layers", 4),
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, meta, device, Path(weights_path), Path(meta_path)


def predict_house_price(sample, model, tokenizer, meta, device):
    numeric_cols = meta["numeric_cols"]
    title_col = meta["text_title_col"]
    desc_col = meta["text_desc_col"]
    addr_col = meta["text_addr_col"]
    max_len = meta["max_len"]

    numeric_row = pd.DataFrame([{col: sample.get(col, np.nan) for col in numeric_cols}])
    numeric_row = meta["numeric_imputer"].transform(numeric_row[numeric_cols])
    numeric_row = meta["numeric_scaler"].transform(numeric_row)
    numeric_tensor = torch.tensor(numeric_row.astype(np.float32), device=device)

    title = str(sample.get(title_col, "") or "")
    description = str(sample.get(desc_col, "") or "")
    address = str(sample.get(addr_col, "") or "")
    full_text = f"{title} [SEP] {description} [SEP] {address}"

    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
        verbose=False,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        pred_log = model(input_ids, attention_mask, numeric_tensor).item()

    pred_ty = float(np.exp(pred_log))
    return pred_log, pred_ty


def build_default_numeric_values(numeric_cols):
    defaults = {
        "dien_tich_num": 60.0,
        "dien_tich_dat_num": 60.0,
        "dien_tich_su_dung_num": 100.0,
        "so_phong_ngu": 3.0,
        "so_phong_ve_sinh": 2.0,
        "tong_so_tang": 2.0,
        "chieu_ngang_num": 4.0,
        "chieu_dai_num": 15.0,
    }
    return {col: defaults.get(col, 0.0) for col in numeric_cols}


def get_feature_display_label(col_name):
    label_map = {
        "dien_tich_num": "Diện tích (m²)",
        "dien_tich_dat_num": "Diện tích đất (m²)",
        "dien_tich_su_dung_num": "Diện tích sử dụng (m²)",
        "so_phong_ngu": "Số phòng ngủ",
        "so_phong_ve_sinh": "Số phòng vệ sinh",
        "tong_so_tang": "Tổng số tầng",
        "chieu_ngang_num": "Chiều ngang (m)",
        "chieu_dai_num": "Chiều dài (m)",
    }

    if col_name in label_map:
        return label_map[col_name]

    cleaned = col_name.replace("_num", "")
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned.capitalize()


def get_display_column_label(col_name):
    label_map = {
        "tieu_de": "Tiêu đề",
        "dia_chi": "Địa chỉ",
        "mo_ta": "Mô tả",
        "so_luong_nguyen_nhan": "Số nguyên nhân bất thường",
        "ly_do_bat_thuong": "Nguyên nhân bất thường",
        "gia_ban_num": "Giá đăng (tỷ đồng)",
        "gia_de_xuat_ty": "Giá đề xuất (tỷ đồng)",
        "gia_de_xuat_vnd": "Giá đề xuất (VND)",
        "chenh_lech_ty": "Chênh lệch (tỷ đồng)",
        "chenh_lech_pct": "Chênh lệch (%)",
        "nhan_dinh_gia": "Nhận định giá",
        "anomaly_votes": "Số phương pháp đồng thuận",
        "ly_do_gia_bat_thuong": "Nguyên nhân bất thường giá",
    }

    if col_name in label_map:
        return label_map[col_name]

    return get_feature_display_label(col_name)


def localize_display_columns(df_input):
    rename_map = {col: get_display_column_label(col) for col in df_input.columns}
    return df_input.rename(columns=rename_map)


@st.cache_data(show_spinner=False)
def load_anomaly_source_data():
    candidate_paths = [
        BASE_DIR / "data" / "processed" / "outlier_samples_6cols.csv",
        BASE_DIR / "data" / "processed" / "non_outlier_samples_6cols.csv",
        BASE_DIR / "notebooks" / "outlier_samples_6cols.csv",
    ]

    data_path = None
    for path in candidate_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError("Không tìm thấy file dữ liệu để phát hiện bất thường.")

    return pd.read_csv(data_path), data_path


def detect_anomalies_with_reasons(df_input):
    df = df_input.copy()

    cols_to_check = [
        "chieu_ngang_num",
        "chieu_dai_num",
        "tong_so_tang",
        "so_phong_ngu",
        "so_phong_ve_sinh",
        "dien_tich_num",
    ]

    for col in cols_to_check:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["calc_area"] = df["chieu_ngang_num"] * df["chieu_dai_num"]
    mask_dim_mismatch = (df["calc_area"] > 0) & (
        (df["calc_area"] > 1.5 * df["dien_tich_num"])
        | (df["calc_area"] < 0.5 * df["dien_tich_num"])
    )

    df["total_est_area"] = df["dien_tich_num"] * df["tong_so_tang"]
    df["total_rooms"] = df["so_phong_ngu"] + df["so_phong_ve_sinh"]
    mask_room_density = (df["total_rooms"] > 0) & (df["total_est_area"] / df["total_rooms"] < 5)

    mask_illogical_dim = (df["chieu_ngang_num"] > df["dien_tich_num"]) | (
        df["chieu_dai_num"] > df["dien_tich_num"]
    )
    mask_extreme_floors = (df["tong_so_tang"] > 10) & (df["dien_tich_num"] < 50)

    reasons = pd.Series("", index=df.index, dtype="object")
    reasons.loc[mask_dim_mismatch] += "Kích thước (ngang*dài) không khớp diện tích; "
    reasons.loc[mask_room_density] += "Mật độ phòng quá cao so với diện tích; "
    reasons.loc[mask_illogical_dim] += "Ngang hoặc dài lớn hơn cả diện tích; "
    reasons.loc[mask_extreme_floors] += "Số tầng quá cao so với diện tích nhỏ; "

    df["ly_do_bat_thuong"] = reasons.str.rstrip("; ")
    anomalies = df[df["ly_do_bat_thuong"].str.len() > 0].copy()
    anomalies["so_luong_nguyen_nhan"] = anomalies["ly_do_bat_thuong"].str.count(";") + 1

    return anomalies.sort_values("so_luong_nguyen_nhan", ascending=False)


def detect_price_anomalies_ensemble(df_input):
    """Detect price anomalies using ensemble voting and return anomalous rows."""
    df = df_input.copy()

    cols_required = [
        "dien_tich_num",
        "so_phong_ngu",
        "so_phong_ve_sinh",
        "tong_so_tang",
        "chieu_ngang_num",
        "chieu_dai_num",
    ]

    for col in cols_required:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "log_gia_ban" not in df.columns:
        if "gia_ban_num" in df.columns:
            df["gia_ban_num"] = pd.to_numeric(df["gia_ban_num"], errors="coerce")
            df["log_gia_ban"] = np.log1p(df["gia_ban_num"])
        else:
            return pd.DataFrame()
    else:
        df["log_gia_ban"] = pd.to_numeric(df["log_gia_ban"], errors="coerce")
        if "gia_ban_num" not in df.columns:
            df["gia_ban_num"] = np.expm1(df["log_gia_ban"])

    available_features = [c for c in cols_required if c in df.columns]
    X_raw = df[available_features].copy()

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X_raw), columns=available_features, index=df.index
    )

    valid_mask = df["log_gia_ban"].notna()
    X_valid = X.loc[valid_mask].copy()
    y_valid = df.loc[valid_mask, "log_gia_ban"].copy()

    if len(X_valid) < 50:
        return pd.DataFrame()

    df_result = df.loc[valid_mask].copy()

    try:
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        rf.fit(X_valid, y_valid)

        pred_all = rf.predict(X_valid)
        residual = y_valid.values - pred_all
        residual_abs = np.abs(residual)

        res_mu = residual_abs.mean()
        res_sigma = residual_abs.std()
        res_z = (residual_abs - res_mu) / (res_sigma + 1e-9)

        df_result["residual_z_score"] = res_z
        df_result["anomaly_residual"] = (res_z > 2.5).astype(int)
    except Exception:
        df_result["anomaly_residual"] = 0
        df_result["residual_z_score"] = 0.0

    try:
        q1 = df_result["gia_ban_num"].quantile(0.25)
        q3 = df_result["gia_ban_num"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_result["anomaly_iqr"] = (
            (df_result["gia_ban_num"] < lower_bound) | (df_result["gia_ban_num"] > upper_bound)
        ).astype(int)
    except Exception:
        df_result["anomaly_iqr"] = 0

    try:
        iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
        iso.fit(X_valid)

        iso_pred = iso.predict(X_valid)
        iso_score = -iso.score_samples(X_valid)

        df_result["iso_score"] = iso_score
        df_result["anomaly_iso"] = (iso_pred == -1).astype(int)
    except Exception:
        df_result["anomaly_iso"] = 0
        df_result["iso_score"] = 0.0

    df_result["anomaly_votes"] = (
        df_result["anomaly_residual"] + df_result["anomaly_iqr"] + df_result["anomaly_iso"]
    )
    df_result["is_price_anomaly"] = (df_result["anomaly_votes"] >= 2).astype(int)

    df_result["ly_do_gia_bat_thuong"] = ""
    if "anomaly_residual" in df_result.columns:
        mask_residual = df_result["anomaly_residual"] == 1
        df_result.loc[mask_residual, "ly_do_gia_bat_thuong"] = (
            df_result.loc[mask_residual, "ly_do_gia_bat_thuong"] + "Giá dự đoán lệch rất xa so với thực tế (Z-score > 2.5); "
        )
    if "anomaly_iqr" in df_result.columns:
        mask_iqr = df_result["anomaly_iqr"] == 1
        df_result.loc[mask_iqr, "ly_do_gia_bat_thuong"] = (
            df_result.loc[mask_iqr, "ly_do_gia_bat_thuong"] + "Giá vượt quá khoảng IQR chuẩn [Q1-1.5*IQR, Q3+1.5*IQR]; "
        )
    if "anomaly_iso" in df_result.columns:
        mask_iso = df_result["anomaly_iso"] == 1
        df_result.loc[mask_iso, "ly_do_gia_bat_thuong"] = (
            df_result.loc[mask_iso, "ly_do_gia_bat_thuong"] + "Kết hợp đặc trưng (diện tích, phòng, tầng) tạo nên điểm cô lập cao; "
        )

    df_result["ly_do_gia_bat_thuong"] = df_result["ly_do_gia_bat_thuong"].str.rstrip("; ")

    price_anomalies = df_result[df_result["is_price_anomaly"] == 1].copy()
    return price_anomalies.sort_values("anomaly_votes", ascending=False)


def load_anomaly_input_data():
    source_option = st.radio(
        "Nguồn dữ liệu",
        options=["Dữ liệu mặc định", "Tải file CSV/Excel", "Nhập thủ công"],
        horizontal=True,
        key="anomaly_data_source",
    )

    if source_option == "Dữ liệu mặc định":
        try:
            df_source, data_path = load_anomaly_source_data()
            return df_source, str(data_path)
        except Exception as exc:
            st.error("Không thể tải dữ liệu mặc định.")
            st.exception(exc)
            return None, None

    if source_option == "Tải file CSV/Excel":
        uploaded_file = st.file_uploader(
            "Chọn file dữ liệu",
            type=["csv", "xlsx", "xls"],
            key="anomaly_upload_file",
        )
        if uploaded_file is None:
            st.info("Hãy tải lên một file CSV hoặc Excel để bắt đầu.")
            return None, None

        file_name = uploaded_file.name.lower()
        try:
            if file_name.endswith(".csv"):
                df_source = pd.read_csv(uploaded_file)
            else:
                df_source = pd.read_excel(uploaded_file)
            return df_source, uploaded_file.name
        except Exception as exc:
            st.error("Không đọc được file đã tải lên.")
            st.exception(exc)
            return None, None

    manual_template = pd.DataFrame(
        [
            {
                "tieu_de": "",
                "dia_chi": "",
                "mo_ta": "",
                "gia_ban_num": 8.0,
                "dien_tich_num": 60.0,
                "so_phong_ngu": 3.0,
                "so_phong_ve_sinh": 2.0,
                "tong_so_tang": 2.0,
                "chieu_ngang_num": 4.0,
                "chieu_dai_num": 15.0,
            }
        ]
    )

    st.caption("Bạn có thể thêm/xóa dòng trực tiếp trong bảng dưới đây.")
    manual_column_config = {
        "tieu_de": st.column_config.TextColumn("Tiêu đề"),
        "dia_chi": st.column_config.TextColumn("Địa chỉ"),
        "mo_ta": st.column_config.TextColumn("Mô tả"),
        "gia_ban_num": st.column_config.NumberColumn("Giá đăng (tỷ đồng)", format="%.3f"),
        "dien_tich_num": st.column_config.NumberColumn("Diện tích (m²)", format="%.2f"),
        "so_phong_ngu": st.column_config.NumberColumn("Số phòng ngủ", format="%.0f"),
        "so_phong_ve_sinh": st.column_config.NumberColumn("Số phòng vệ sinh", format="%.0f"),
        "tong_so_tang": st.column_config.NumberColumn("Tổng số tầng", format="%.0f"),
        "chieu_ngang_num": st.column_config.NumberColumn("Chiều ngang (m)", format="%.2f"),
        "chieu_dai_num": st.column_config.NumberColumn("Chiều dài (m)", format="%.2f"),
    }

    edited_df = st.data_editor(
        manual_template,
        num_rows="dynamic",
        column_config=manual_column_config,
        use_container_width=True,
        key="manual_anomaly_editor",
    )
    if edited_df is None or edited_df.empty:
        st.info("Hãy nhập ít nhất 1 dòng dữ liệu để phát hiện bất thường.")
        return None, None

    return edited_df.copy(), "Nhập thủ công"


def add_recommended_price_columns(df_input):
    if df_input.empty:
        return df_input

    try:
        model, tokenizer, meta, device, _, _ = load_saved_price_model()
    except Exception:
        df_out = df_input.copy()
        df_out["gia_de_xuat_ty"] = np.nan
        df_out["gia_de_xuat_vnd"] = np.nan
        df_out["nhan_dinh_gia"] = "Không nạp được mô hình để tính giá đề xuất"
        return df_out

    pred_ty_values = []
    for _, row in df_input.iterrows():
        try:
            sample = row.to_dict()
            _, pred_ty = predict_house_price(sample, model, tokenizer, meta, device)
            pred_ty_values.append(float(pred_ty))
        except Exception:
            pred_ty_values.append(np.nan)

    df_out = df_input.copy()
    df_out["gia_de_xuat_ty"] = pred_ty_values
    df_out["gia_de_xuat_vnd"] = df_out["gia_de_xuat_ty"] * 1_000_000_000

    if "gia_ban_num" in df_out.columns:
        actual = pd.to_numeric(df_out["gia_ban_num"], errors="coerce")
        delta_ty = df_out["gia_de_xuat_ty"] - actual
        delta_pct = np.where(actual > 0, (delta_ty / actual) * 100, np.nan)
        df_out["chenh_lech_ty"] = delta_ty
        df_out["chenh_lech_pct"] = delta_pct

        labels = np.where(
            delta_pct >= 0,
            "Giá đăng thấp hơn giá đề xuất",
            "Giá đăng cao hơn giá đề xuất",
        )
        df_out["nhan_dinh_gia"] = np.where(
            np.isnan(delta_pct),
            "Thiếu giá đăng để so sánh",
            labels + " khoảng " + np.abs(delta_pct).round(2).astype(str) + "%",
        )
    else:
        df_out["nhan_dinh_gia"] = "Không có cột gia_ban_num để so sánh"

    return df_out


def render_intro_tab():
    st.title("Ứng dụng dự đoán giá nhà")
    st.subheader("Giới thiệu ứng dụng")
    st.write(
        "Ứng dụng sử dụng mô hình deep learning kết hợp PhoBERT và đặc trưng số "
        "để dự đoán giá bán nhà theo đơn vị tỷ đồng."
    )

    st.markdown("### Chức năng chính")
    st.markdown("- Nhập thông tin văn bản: tiêu đề, mô tả, địa chỉ")
    st.markdown("- Nhập các đặc trưng số của bất động sản")
    st.markdown("- Dự đoán giá nhà từ mô hình đã huấn luyện và đã lưu")

    st.markdown("### Quy trình sử dụng")
    st.markdown("1. Chọn tab 'Dự đoán giá nhà' trong menu bên trái")
    st.markdown("2. Điền thông tin bất động sản")
    st.markdown("3. Nhấn nút 'Dự đoán giá nhà' để xem kết quả")


def render_author_tab():
    st.title("Thông tin tác giả")
    st.subheader("Dự án dự đoán giá nhà")
    st.write("Bạn có thể cập nhật thông tin tác giả bên dưới theo nhu cầu.")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Họ và tên", value="Nguyen Van A", disabled=True)
        st.text_input("Email", value="nguyenvana@example.com", disabled=True)
    with c2:
        st.text_input("Đơn vị", value="Seminar TTTH", disabled=True)
        st.text_input("Niên khóa", value="2025 - 2026", disabled=True)

    st.info("Nếu bạn muốn, mình có thể sửa thông tin tác giả đúng theo thông tin thật của bạn.")


def render_anomaly_tab():
    st.markdown(
        """
        <div class="app-hero">
            <h2>Phát hiện căn nhà bất thường</h2>
            <p>Đánh dấu bản ghi nghi ngờ nhập sai và hiển thị nguyên nhân chi tiết cho từng căn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_source, data_path = load_anomaly_input_data()
    if df_source is None:
        return
    if df_source.empty:
        st.warning("Dữ liệu rỗng. Vui lòng cung cấp dữ liệu để tiếp tục.")
        return

    st.caption(f"Nguồn dữ liệu: {data_path} | Tổng số dòng: {len(df_source):,}")

    anomalies = detect_anomalies_with_reasons(df_source)

    total_rows = len(df_source)
    anomaly_rows = len(anomalies)
    ratio = (anomaly_rows / total_rows * 100) if total_rows else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số căn", f"{total_rows:,}")
    m2.metric("Số căn bất thường", f"{anomaly_rows:,}")
    m3.metric("Tỷ lệ bất thường", f"{ratio:.2f}%")

    if anomaly_rows == 0:
        st.success("Không phát hiện căn nhà bất thường về thông số kỹ thuật.")
    else:
        st.markdown("### Danh sách bất thường thông số kỹ thuật")

        default_columns = [
            "tieu_de",
            "dia_chi",
            "dien_tich_num",
            "chieu_ngang_num",
            "chieu_dai_num",
            "so_phong_ngu",
            "so_phong_ve_sinh",
            "tong_so_tang",
            "so_luong_nguyen_nhan",
            "ly_do_bat_thuong",
        ]
        view_columns = [c for c in default_columns if c in anomalies.columns]

        top_k = st.slider(
            "Số dòng hiển thị (thông số kỹ thuật)",
            min_value=10,
            max_value=300,
            value=50,
            step=10,
        )
        st.dataframe(
            localize_display_columns(anomalies[view_columns].head(top_k)),
            use_container_width=True,
            height=480,
        )

        csv_data = (
            localize_display_columns(anomalies[view_columns])
            .to_csv(index=False)
            .encode("utf-8-sig")
        )
        st.download_button(
            label="Tải CSV bất thường thông số kỹ thuật",
            data=csv_data,
            file_name="nha_bat_thuong_thong_so.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    price_anomalies = detect_price_anomalies_ensemble(df_source)
    price_total = len(df_source)
    price_anom_count = len(price_anomalies)
    price_ratio = (price_anom_count / price_total * 100) if price_total else 0

    p1, p2, p3 = st.columns(3)
    p1.metric("Tổng căn (dùng cho giá)", f"{price_total:,}")
    p2.metric("Số căn bất thường giá", f"{price_anom_count:,}")
    p3.metric("Tỷ lệ bất thường giá", f"{price_ratio:.2f}%")

    if price_anom_count == 0:
        st.success("Không phát hiện căn nhà bất thường về giá.")
        return

    with st.spinner("Đang tính giá đề xuất từ mô hình đã lưu..."):
        price_anomalies = add_recommended_price_columns(price_anomalies)

    st.markdown("### Danh sách bất thường giá")
    price_view_cols = [
        c
        for c in [
            "tieu_de",
            "dia_chi",
            "gia_ban_num",
            "gia_de_xuat_ty",
            "chenh_lech_ty",
            "chenh_lech_pct",
            "nhan_dinh_gia",
            "anomaly_votes",
            "ly_do_gia_bat_thuong",
        ]
        if c in price_anomalies.columns
    ]

    price_top_k = st.slider(
        "Số dòng hiển thị (bất thường giá)",
        min_value=10,
        max_value=300,
        value=50,
        step=10,
    )
    st.dataframe(
        localize_display_columns(price_anomalies[price_view_cols].head(price_top_k)),
        use_container_width=True,
        height=480,
    )

    price_csv_data = (
        localize_display_columns(price_anomalies[price_view_cols])
        .to_csv(index=False)
        .encode("utf-8-sig")
    )
    st.download_button(
        label="Tải CSV bất thường giá",
        data=price_csv_data,
        file_name="nha_bat_thuong_gia_co_gia_de_xuat.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_prediction_tab():
    st.markdown(
        """
        <div class="app-hero">
            <h2>Ứng dụng dự đoán giá nhà</h2>
            <p>Dự đoán giá bán (tỷ đồng) bằng mô hình PhoBERT kết hợp dữ liệu số.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("---")
        with st.expander("Trạng thái mô hình", expanded=False):
            st.write(f"GPU khả dụng: {torch.cuda.is_available()}")
            st.write("Nơi tìm PhoBERT local:")
            for model_dir in PHOBERT_DIR_CANDIDATES:
                st.write(f"- {model_dir}")
            st.write("Nơi tìm artifact:")
            for artifact_dir in ARTIFACT_DIR_CANDIDATES:
                st.write(f"- {artifact_dir}")

    try:
        with st.spinner("Đang nạp mô hình đã lưu..."):
            model, tokenizer, meta, device, resolved_weights, resolved_meta = load_saved_price_model()
    except Exception as exc:
        st.error("Không thể nạp mô hình đã lưu.")
        st.exception(exc)
        st.info(
            "Bạn hãy mở notebook deep_model, chạy cell huấn luyện để tạo "
            "best_model_state.pt va inference_meta.pkl."
        )
        return

    c_info1, c_info2 = st.columns(2)
    c_info1.success(f"Đã nạp mô hình thành công trên device: {device}")
    c_info2.info("Sẵn sàng dự đoán. Hãy nhập thông tin bất động sản bên dưới.")

    with st.expander("Chi tiết tệp mô hình", expanded=False):
        st.caption(f"Weights đang dùng: {resolved_weights}")
        st.caption(f"Metadata đang dùng: {resolved_meta}")

    text_title_col = meta["text_title_col"]
    text_desc_col = meta["text_desc_col"]
    text_addr_col = meta["text_addr_col"]
    numeric_cols = meta["numeric_cols"]

    st.markdown('<div class="section-title">Nhập thông tin cần dự đoán</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1.0])

    with c1:
        with st.container(border=True):
            st.markdown("#### Thông tin mô tả")
            title = st.text_input("Tiêu đề", value="Bán nhà phố đẹp khu trung tâm")
            description = st.text_area(
                "Mô tả",
                value="Nhà mới, hẻm xe hơi, gần trường học, chợ và bệnh viện.",
                height=160,
            )
            address = st.text_input("Địa chỉ", value="Quận Bình Thạnh, TP.HCM")

    with c2:
        with st.container(border=True):
            st.markdown("#### Đặc trưng số")
            numeric_defaults = build_default_numeric_values(numeric_cols)
            numeric_values = {}
            for col in numeric_cols:
                display_label = get_feature_display_label(col)
                numeric_values[col] = st.number_input(
                    display_label,
                    value=float(numeric_defaults[col]),
                    step=1.0,
                    format="%.4f",
                    key=f"num_input_{col}",
                )

    predict_btn = st.button("Dự đoán giá nhà", type="primary", use_container_width=True)

    if predict_btn:
        sample = {
            text_title_col: title,
            text_desc_col: description,
            text_addr_col: address,
            **numeric_values,
        }

        _, pred_ty = predict_house_price(sample, model, tokenizer, meta, device)
        pred_vnd = pred_ty * 1_000_000_000

        st.markdown("### Kết quả dự đoán")
        r1, r2 = st.columns(2)
        r1.metric("Giá dự đoán (tỷ đồng)", f"{pred_ty:,.3f}")
        r2.metric("Giá dự đoán (VND)", f"{pred_vnd:,.0f}")


def main():
    st.set_page_config(page_title="Dự đoán giá nhà", page_icon="🏠", layout="wide")
    inject_global_styles()

    with st.sidebar:
        st.header("Menu")
        selected_tab = st.radio(
            "Chọn tab",
            ["Giới thiệu ứng dụng", "Thông tin tác giả", "Dự đoán giá nhà", "Phát hiện bất thường"],
            index=2,
        )

    if selected_tab == "Giới thiệu ứng dụng":
        render_intro_tab()
    elif selected_tab == "Thông tin tác giả":
        render_author_tab()
    elif selected_tab == "Phát hiện bất thường":
        render_anomaly_tab()
    else:
        render_prediction_tab()


if __name__ == "__main__":
    main()
