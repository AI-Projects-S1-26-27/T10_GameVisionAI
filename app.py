from __future__ import annotations

import csv
import hashlib
import html
import json
import tempfile
from pathlib import Path
from typing import Any

import av
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


# -----------------------------------------------------------------------------
# Paths and model metadata
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "squash_gru_model.pth"
METRICS_PATH = ROOT / "metrics.json"
CONFUSION_PATH = ROOT / "confusion_matrix.png"
CURVES_PATH = ROOT / "training_curves.png"
HISTORY_PATH = ROOT / "training_history.csv"
SPLIT_PATH = ROOT / "data_split.csv"

st.set_page_config(
    page_title="RALLY / Squash review",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Visual identity: paper, ink, and court-lime rather than the default Streamlit UI
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');
      :root { --paper:#f3f1e9; --ink:#17231e; --muted:#74796f; --line:#d9d8ce;
              --lime:#d5f36a; --rust:#c36b4d; --card:#fbfaf5; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp { background:var(--paper); color:var(--ink); }
      #MainMenu, footer, header { visibility:hidden; }
      .block-container { max-width:1240px; padding:1.35rem 2.5rem 4rem; }
      [data-testid="stToolbar"] { display:none; }
      .masthead { display:flex; align-items:center; justify-content:space-between;
                  border-bottom:1px solid var(--line); padding:0 0 16px; margin-bottom:26px; }
      .brand { display:flex; align-items:center; gap:11px; font-weight:800; letter-spacing:.13em; font-size:13px; }
      .brand-dot { width:12px; height:12px; border-radius:50%; background:var(--rust); display:inline-block; }
      .mast-right { color:var(--muted); font:11px 'DM Mono',monospace; letter-spacing:.08em; }
      .eyebrow { color:#73786e; font:11px 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }
      .hero-title { font:700 clamp(42px,6vw,72px)/.98 'Playfair Display',Georgia,serif;
                    letter-spacing:-.045em; color:var(--ink); margin:12px 0 14px; }
      .hero-copy { color:#656c63; font-size:15px; line-height:1.65; max-width:640px; }
      .section-title { font:700 34px/1.05 'Playfair Display',Georgia,serif; letter-spacing:-.025em; margin:8px 0 10px; }
      .section-copy { color:#697067; font-size:14px; line-height:1.6; }
      .rule { height:1px; background:var(--line); margin:22px 0; }
      .paper-card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px; }
      .dark-card { background:var(--ink); color:#f6f5ed; border-radius:16px; padding:22px; }
      .small-label { color:#888d82; font:10px 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }
      .result-name { font:700 clamp(30px,4vw,48px)/1 'Playfair Display',Georgia,serif; margin:10px 0 8px; }
      .result-confidence { color:#c7d19c; font:12px 'DM Mono',monospace; }
      .stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }
      .stat-card { background:#fbfaf5; border:1px solid #ddddd3; border-radius:13px; padding:17px 18px; min-height:104px; }
      .stat-value { font:700 29px/1.1 'DM Sans',sans-serif; letter-spacing:-.04em; margin-top:11px; }
      .stat-label { color:#777d72; font:10px 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
      .arch-track { display:grid; grid-template-columns:1fr 28px 1fr 28px 1fr 28px 1fr; gap:6px; align-items:stretch; margin:24px 0; }
      .arch-node { background:#fbfaf5; border:1px solid #d9d8ce; border-radius:13px; min-height:130px; padding:17px 15px; }
      .arch-node.feature { background:#e8edda; border-color:#cdd8b2; }
      .arch-node.output { background:#17231e; color:#f5f4ec; border-color:#17231e; }
      .arch-num { font:10px 'DM Mono',monospace; color:#94998d; }
      .arch-name { font-weight:800; font-size:15px; margin:13px 0 6px; }
      .arch-detail { color:#777d72; font-size:11px; line-height:1.5; }
      .arch-node.output .arch-num, .arch-node.output .arch-detail { color:#c6cebb; }
      .arch-arrow { display:flex; align-items:center; justify-content:center; color:#9ba08f; font-size:20px; }
      .note { border-left:3px solid var(--rust); padding:10px 14px; background:#eeeae0; color:#5c655b; font-size:12px; line-height:1.6; border-radius:0 8px 8px 0; }
      .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:#fbfaf5; }
      table.rally-table { border-collapse:collapse; width:100%; font-size:13px; }
      table.rally-table th { text-align:left; color:#73796f; background:#eeede5; font:10px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
      table.rally-table td, table.rally-table th { padding:12px 14px; border-bottom:1px solid #e5e4db; white-space:nowrap; }
      table.rally-table tr:last-child td { border-bottom:0; }
      .footnote { color:#8a8e83; font:10px 'DM Mono',monospace; letter-spacing:.04em; margin-top:18px; }
      div[role="radiogroup"] { gap:8px; }
      div[role="radiogroup"] label { background:#e9e8df; border:1px solid #d8d8ce; padding:8px 15px; border-radius:99px; }
      div[role="radiogroup"] label:has(input:checked) { background:#17231e; color:#f7f6ed; border-color:#17231e; }
      div[data-testid="stFileUploader"] { background:#fbfaf5; border:1px dashed #aeb29f; border-radius:13px; padding:16px; }
      div[data-testid="stFileUploader"] section { border:0; }
      .stButton button { background:#d5f36a; color:#17231e; border:0; border-radius:99px;
                         padding:12px 23px; font-weight:800; min-height:46px; }
      .stButton button:hover { background:#c7e65b; color:#17231e; border:0; }
      .stProgress > div > div > div > div { background:#b3d64b; }
      [data-testid="stMetric"] { background:#fbfaf5; border:1px solid #ddddd3; padding:13px 15px; border-radius:12px; }
      @media (max-width:850px) {
        .block-container { padding:1rem 1rem 3rem; }
        .stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .arch-track { grid-template-columns:1fr; }
        .arch-arrow { transform:rotate(90deg); min-height:18px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="masthead">
      <div class="brand"><span class="brand-dot"></span> RALLY <span style="font-weight:500;color:#858a7e">/ VIDEO REVIEW</span></div>
      <div class="mast-right">SQUASH · VIDEO CLASSIFIER&nbsp;&nbsp; / &nbsp;&nbsp;MODEL 01</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    ["01  /  Make a call", "02  /  Model & results"],
    horizontal=True,
    label_visibility="collapsed",
)


# -----------------------------------------------------------------------------
# Model: ResNet50 frame embeddings followed by the saved GRU classification head
# -----------------------------------------------------------------------------
class GRUHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, dropout: float, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(sequence)
        final_step = outputs[:, -1, :]
        return self.classifier(self.dropout(final_step))


@st.cache_resource(show_spinner=False)
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH.name}")

    # The supplied checkpoint stores the GRU and classifier, not the ResNet50 weights.
    checkpoint: dict[str, Any] = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("backbone") != "resnet50":
        raise ValueError("This app expects the ResNet50 backbone recorded in the checkpoint.")

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    backbone = resnet50(weights=weights)
    backbone.fc = nn.Identity()

    class_names = checkpoint["class_names"]
    head = GRUHead(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_size=int(checkpoint["hidden_size"]),
        dropout=float(checkpoint["dropout"]),
        n_classes=len(class_names),
    )
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()
    head.to(device).eval()
    return backbone, head, transform, class_names, int(checkpoint["num_frames"]), device


def _video_stream(container: av.container.InputContainer):
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        raise ValueError("The uploaded file does not contain a video stream.")
    stream.thread_type = "AUTO"
    return stream


def sample_video_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    """Sample frames uniformly by decoded frame index, matching the saved 24-frame setup."""
    with av.open(str(video_path)) as container:
        stream = _video_stream(container)
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            total_frames = sum(1 for _ in container.decode(stream))

    if total_frames <= 0:
        raise ValueError("No readable video frames were found.")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
    wanted = set(int(i) for i in frame_indices)
    selected: dict[int, Image.Image] = {}

    with av.open(str(video_path)) as container:
        stream = _video_stream(container)
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                selected[index] = frame.to_image().convert("RGB")
            if len(selected) == len(wanted):
                break

    # If the container's frame-count metadata was inaccurate, recount and retry once.
    if len(selected) != len(wanted):
        with av.open(str(video_path)) as container:
            stream = _video_stream(container)
            total_frames = sum(1 for _ in container.decode(stream))
        if total_frames <= 0:
            raise ValueError("The video could not be decoded into frames.")
        frame_indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
        wanted = set(int(i) for i in frame_indices)
        selected = {}
        with av.open(str(video_path)) as container:
            stream = _video_stream(container)
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    selected[index] = frame.to_image().convert("RGB")
                if len(selected) == len(wanted):
                    break

    if not selected:
        raise ValueError("No frames could be sampled from this video.")
    return [selected[int(index)] for index in frame_indices]


def predict_video(video_path: Path):
    backbone, head, transform, class_names, num_frames, device = load_model()
    frames = sample_video_frames(video_path, num_frames)
    batch = torch.stack([transform(frame) for frame in frames]).to(device)

    with torch.inference_mode():
        frame_features = backbone(batch)             # [24, 2048]
        sequence = frame_features.unsqueeze(0)       # [1, 24, 2048]
        logits = head(sequence)                       # [1, 3]
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    return {
        "label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(class_names)},
        "device": str(device),
        "frames": num_frames,
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric_card(label: str, value: str) -> str:
    return (
        '<div class="stat-card">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        '</div>'
    )


# -----------------------------------------------------------------------------
# Page 1: live inference. Uploaded video is decoded and classified by the model.
# No stored test predictions or labels are used for this result.
# -----------------------------------------------------------------------------
if page.startswith("01"):
    st.markdown('<div class="eyebrow">01 / VIDEO REVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Review a rally.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Drop in a squash clip. RALLY samples the action, extracts visual features frame by frame, '
        'and sends the sequence through the trained ResNet50–GRU classifier.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="eyebrow">UPLOAD / 01</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a video file",
            type=None,
            accept_multiple_files=False,
            label_visibility="collapsed",
            help="Upload a video clip.",
        )
        uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
        upload_hash = hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes is not None else None
        if st.session_state.get("active_upload_hash") != upload_hash:
            st.session_state.pop("last_prediction", None)
            st.session_state["active_upload_hash"] = upload_hash

        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".video"
            st.markdown(
                f'<div class="note"><b>{html.escape(uploaded.name)}</b><br>'
                f'{uploaded.size / (1024 * 1024):.1f} MB · ready to analyse</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        run_prediction = st.button("Analyse clip  ↗", type="primary", disabled=uploaded is None, use_container_width=True)

    with right:
        st.markdown(
            '<div class="dark-card" style="min-height:225px">'
            '<div class="small-label" style="color:#bdc6b0">HOW IT WORKS</div>'
            '<div style="font:700 25px/1.15 Playfair Display,Georgia,serif;margin:18px 0 10px">The rally, frame by frame.</div>'
            '<div style="font-size:13px;line-height:1.65;color:#c4ccbb">The model analyses 24 frames from the clip and uses their visual features and movement to make a call.</div>'
            '<div style="margin-top:20px;font:10px DM Mono,monospace;letter-spacing:.12em;color:#d5f36a">24 FRAMES&nbsp; / &nbsp;3 CLASSES</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if run_prediction and uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".video"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_bytes)
                temp_path = Path(temp_file.name)
            try:
                with st.spinner("Analysing clip…"):
                    result = predict_video(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

            st.session_state["last_prediction"] = result
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            st.info(
                "Make sure the file contains a readable video. On first run, the app may also need "
                "internet access (or a local torchvision cache) to load the pretrained ResNet50 weights."
            )

    result = st.session_state.get("last_prediction") if uploaded is not None else None
    if result:
        names = {"no_let": "No let", "stroke": "Stroke", "yes_let": "Yes let"}
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([0.95, 1.05], gap="large")
        with result_left:
            st.markdown(
                '<div class="dark-card">'
                '<div class="small-label" style="color:#bdc6b0">MODEL VERDICT</div>'
                f'<div class="result-name">{html.escape(names.get(result["label"], result["label"]))}</div>'
                f'<div class="result-confidence">{percent(result["confidence"])} confidence</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with result_right:
            st.markdown('<div class="eyebrow">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
            for class_name in ["no_let", "stroke", "yes_let"]:
                prob = result["probabilities"][class_name]
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin:12px 0 4px;font-size:13px">'
                    f'<span>{html.escape(names[class_name])}</span><span style="font-family:DM Mono,monospace">{percent(prob)}</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(prob)
        st.markdown(
            f'<div class="footnote">INFERENCE DEVICE: {html.escape(result["device"].upper())} '
            f'&nbsp;·&nbsp; FRAMES PASSED TO MODEL: {result["frames"]} '
            '&nbsp;·&nbsp; RESULT FROM THIS CLIP</div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Page 2: architecture and measured evaluation results from the supplied files
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="eyebrow">02 / MODEL & RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model & results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A frame-sequence classifier trained to distinguish three squash decisions. '
        'The figures below are read from the supplied evaluation artefacts, not generated by the live uploader.</div>',
        unsafe_allow_html=True,
    )

    if not METRICS_PATH.exists():
        st.error("metrics.json is missing. Keep it beside app.py.")
        st.stop()
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    st.markdown('<div class="eyebrow" style="margin-top:28px">MODEL FLOW</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="arch-track">
          <div class="arch-node">
            <div class="arch-num">01 / INPUT</div>
            <div class="arch-name">Video frames</div>
            <div class="arch-detail">24 frames sampled uniformly across the clip<br>RGB · default ResNet50 transform</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-node feature">
            <div class="arch-num">02 / FEATURES</div>
            <div class="arch-name">ResNet50</div>
            <div class="arch-detail">Pretrained visual backbone<br>2,048-dimensional frame features</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-node">
            <div class="arch-num">03 / TEMPORAL</div>
            <div class="arch-name">GRU</div>
            <div class="arch-detail">Sequence modelling<br>Hidden size 128 · dropout 0.35</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-node output">
            <div class="arch-num">04 / CLASSIFIER</div>
            <div class="arch-name">3-way output</div>
            <div class="arch-detail">Linear layer + softmax<br>no_let · stroke · yes_let</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">TEST SET / 55 CLIPS</div>', unsafe_allow_html=True)
    main_cards = [
        ("Accuracy", percent(metrics["accuracy"])),
        ("Balanced accuracy", percent(metrics["balanced_accuracy"])),
        ("Macro F1", percent(metrics["macro_f1"])),
        ("Macro ROC-AUC", percent(metrics["roc_auc_macro_ovr"])),
    ]
    st.markdown('<div class="stat-grid">' + ''.join(metric_card(k, v) for k, v in main_cards) + '</div>', unsafe_allow_html=True)

    extra_cards = [
        ("Macro precision", percent(metrics["macro_precision"])),
        ("Macro recall", percent(metrics["macro_recall"])),
        ("Weighted F1", percent(metrics["weighted_f1"])),
        ("Matthews corr. coeff.", f'{metrics["matthews_correlation_coefficient"]:.4f}'),
    ]
    st.markdown('<div class="stat-grid">' + ''.join(metric_card(k, v) for k, v in extra_cards) + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:28px">CLASS-WISE BREAKDOWN</div>', unsafe_allow_html=True)
    class_rows = []
    for name in metrics["class_order"]:
        item = metrics["per_class"][name]
        class_rows.append(
            "<tr>"
            f"<td><b>{html.escape(name)}</b></td>"
            f"<td>{percent(item['precision'])}</td>"
            f"<td>{percent(item['recall_sensitivity'])}</td>"
            f"<td>{percent(item['specificity'])}</td>"
            f"<td>{percent(item['f1_score'])}</td>"
            f"<td>{int(item['support'])}</td>"
            "</tr>"
        )
    class_table = (
        '<div class="table-wrap"><table class="rally-table"><thead><tr>'
        '<th>Class</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1 score</th><th>Support</th>'
        '</tr></thead><tbody>' + ''.join(class_rows) + '</tbody></table></div>'
    )
    aggregate_rows = []
    report = metrics.get("classification_report", {})
    for key, label in [("macro avg", "Macro avg"), ("weighted avg", "Weighted avg")]:
        if key in report:
            item = report[key]
            aggregate_rows.append(
                "<tr>"
                f"<td><b>{label}</b></td>"
                f"<td>{percent(float(item['precision']))}</td>"
                f"<td>{percent(float(item['recall']))}</td>"
                "<td>—</td>"
                f"<td>{percent(float(item['f1-score']))}</td>"
                f"<td>{int(item['support'])}</td>"
                "</tr>"
            )
    class_table = class_table.replace('</tbody></table>', ''.join(aggregate_rows) + '</tbody></table>')
    st.markdown(class_table, unsafe_allow_html=True)

    # Keep the raw training history alongside the plot so the final epoch is visible.
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
        if history:
            last_epoch = history[-1]
            train_cards = [
                ("Training epochs", str(len(history))),
                ("Final train loss", f"{float(last_epoch['train_loss']):.4f}"),
                ("Final train accuracy", percent(float(last_epoch['train_accuracy']))),
                ("Test videos", str(metrics["test_video_count"])),
            ]
            st.markdown('<div class="eyebrow" style="margin-top:26px">TRAINING RUN / FINAL EPOCH</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-grid">' + ''.join(metric_card(k, v) for k, v in train_cards) + '</div>', unsafe_allow_html=True)

    image_left, image_right = st.columns([1, 1], gap="large")
    with image_left:
        st.markdown('<div class="eyebrow" style="margin-top:28px">CONFUSION MATRIX / TEST VIDEOS</div>', unsafe_allow_html=True)
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), use_container_width=True)
        else:
            st.warning("confusion_matrix.png is missing.")
    with image_right:
        st.markdown('<div class="eyebrow" style="margin-top:28px">TRAINING / 50 EPOCHS</div>', unsafe_allow_html=True)
        if CURVES_PATH.exists():
            st.image(str(CURVES_PATH), use_container_width=True)
        else:
            st.warning("training_curves.png is missing.")

    st.markdown('<div class="footnote">EVALUATION ARTEFACTS: metrics.json · confusion_matrix.png · training_curves.png</div>', unsafe_allow_html=True)
