# -*- coding: utf-8 -*-
"""
부유세균 콜로니 자동 카운팅 (1단계 프로토타입)
- TISCH TE-10-890 Single Stage N6 Andersen Impactor 기준
- 사진 업로드(드래그앤드롭) -> 회전/자르기 편집 -> 디시 자동검출 -> 콜로니 카운트 -> CFU/m3
- 현행 SOP(콜로니 수 x 7.07) + Feller 양성공 보정(N=400) 토글
- 검출 방식: 고전 영상처리(OpenCV). AI 아님. (2단계에서 AI로 교체 예정)
실행:  python -m streamlit run app.py
"""

import os
import io
import base64
import tempfile
import subprocess
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------------------------------------------------------
# 계산 엔진 (결정론적 - 항상 정확)
# ----------------------------------------------------------------------------

def feller_correction(r, N=400):
    """양성공 보정(Feller). r=양성 구멍(콜로니) 수, N=총 구멍 수."""
    r = int(round(r))
    if r <= 0:
        return 0.0
    if r >= N:
        r = N - 1
    return sum(N / (N - i) for i in range(r))


def cfu_per_m3(count, air_volume_L):
    if air_volume_L <= 0:
        return 0.0
    return count * (1000.0 / air_volume_L)


# ----------------------------------------------------------------------------
# 이미지 유틸
# ----------------------------------------------------------------------------

def pil_to_bgr(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def auto_detect_dish(img_bgr):
    """접시(원형 한천)를 자동으로 찾아 (cx, cy, r) 반환. 못 찾으면 가운데 추정."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
        param1=100, param2=40,
        minRadius=int(min(h, w) * 0.25), maxRadius=int(min(h, w) * 0.72))
    if circles is not None:
        c = np.round(circles[0, 0]).astype(int)
        # 테두리(반사 림)를 피하려고 반지름을 살짝 안쪽으로
        return int(c[0]), int(c[1]), int(c[2] * 0.93)
    return w // 2, h // 2, int(min(h, w) * 0.44)


def make_dish_mask(shape, cx, cy, r):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
    return mask


def detect_colonies(img_bgr, mask, sensitivity, min_area, max_area,
                    circularity_min, split_touching):
    """색상 기반 검출: 한천(배경) 색과 다른 영역(노란 콜로니/검은 점)을 잡는다."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]  # 밝기 (검은색 판별용)
    # 접시 테두리 오검출 방지: 마스크를 안쪽으로 깎음
    inner = cv2.erode(mask, np.ones((25, 25), np.uint8), iterations=1)

    pts = lab[inner > 0]
    if len(pts) == 0:
        return []
    # 한천(배경) 대표색 = 디시 내부 중앙값
    med = np.median(pts, axis=0)
    # 각 픽셀이 배경색과 얼마나 다른가 (색 거리)
    diff = np.sqrt(((lab - med) ** 2).sum(axis=2))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff8 = np.clip(diff, 0, 255).astype(np.uint8)
    # top-hat: 조명/얼룩 같은 '넓은 배경 변화'는 지우고, 콜로니 크기 덩어리만 남김
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    tophat = cv2.morphologyEx(diff8, cv2.MORPH_TOPHAT, k)

    thresh_val = max(3.0, 12.0 - sensitivity * 0.2)
    binary = np.where(tophat > thresh_val, 255, 0).astype(np.uint8)
    # 검은색(마커 점/손글씨) 제외 — 콜로니는 검은색으로 자라지 않음
    binary[L < 90] = 0
    binary = cv2.bitwise_and(binary, binary, mask=inner)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # close 크게: 점을 지운 콜로니 가운데 구멍을 메움
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    if split_touching:
        binary = _watershed_split(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < circularity_min:
            continue
        good.append(c)
    return good


def _watershed_split(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return binary
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)
    out = np.zeros_like(binary)
    out[markers > 1] = 255
    return out


def draw_results(img_bgr, contours, cx, cy, r):
    vis = img_bgr.copy()
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        cv2.circle(vis, (int(x), int(y)), max(int(radius), 4), (0, 0, 255), 2)
    return vis


def contour_centers(contours):
    """윤곽선들의 중심점 목록 [[x, y], ...] 반환."""
    pts = []
    for c in contours:
        (x, y), _ = cv2.minEnclosingCircle(c)
        pts.append([int(x), int(y)])
    return pts


def contour_diameters_px(contours):
    """각 콜로니의 지름(px) 목록 반환."""
    ds = []
    for c in contours:
        (_, _), radius = cv2.minEnclosingCircle(c)
        ds.append(2.0 * radius)
    return ds


@st.cache_resource
def load_ai_model(path):
    """학습된 YOLO 모델을 불러온다(한 번만)."""
    from ultralytics import YOLO
    return YOLO(path)


@st.cache_data(show_spinner=False)
def load_resized(file_bytes, maxd=1400):
    """업로드 사진을 한 번만 디코딩·축소(캐시) → 클릭할 때마다 다시 안 함."""
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes)))
    if max(pil.size) > maxd:
        s = maxd / max(pil.size)
        pil = pil.resize((int(pil.size[0] * s), int(pil.size[1] * s)))
    return pil


def build_report_html(key_rows, detail_rows, before_jpg, after_jpg, title):
    """결과 표 + 검출 전/후 사진이 담긴 리포트 HTML 생성."""
    b64b = base64.b64encode(before_jpg).decode()
    b64a = base64.b64encode(after_jpg).decode()

    def _rows_html(rr):
        return "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rr)

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>
@page {{ size:A4; margin:14mm 16mm; }}
* {{ box-sizing:border-box; }}
body {{ font-family:'Malgun Gothic',sans-serif; color:#1f2937; }}
h1 {{ font-size:17pt; text-align:center; color:#0f172a; border-bottom:2px solid #1e40af; padding-bottom:7px; margin:0 0 12px; }}
.imgs {{ display:flex; gap:12px; justify-content:center; margin:6px 0 14px; }}
.imgs figure {{ margin:0; width:49%; text-align:center; }}
.imgs img {{ width:100%; border:1px solid #cbd5e1; border-radius:5px; }}
figcaption {{ font-size:9pt; color:#475569; margin-top:4px; font-weight:600; }}
h2 {{ font-size:11pt; color:#1e40af; border-left:3.5px solid #1e40af; padding-left:8px; margin:10px 0 5px; }}
table {{ border-collapse:collapse; width:78%; margin:4px auto; font-size:9.5pt; }}
td {{ border:1px solid #d1d5db; padding:5px 10px; }}
td.k {{ background:#eff6ff; font-weight:700; width:42%; color:#334155; }}
</style></head><body>
<h1>🦠 {title}</h1>
<div class="imgs">
  <figure><img src="data:image/jpeg;base64,{b64b}"><figcaption>검출 전 (원본)</figcaption></figure>
  <figure><img src="data:image/jpeg;base64,{b64a}"><figcaption>검출 후 (콜로니 표시)</figcaption></figure>
</div>
<h2>핵심 결과</h2>
<table>{_rows_html(key_rows)}</table>
<h2>상세 정보</h2>
<table>{_rows_html(detail_rows)}</table>
</body></html>"""


def html_to_pdf(html):
    """Edge 헤드리스로 HTML → PDF 변환, PDF 바이트 반환."""
    edges = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
    edge = next((e for e in edges if os.path.exists(e)), None)
    if edge is None:
        raise RuntimeError("Microsoft Edge를 찾을 수 없어요")
    tmp = tempfile.mkdtemp()
    hp, pp = os.path.join(tmp, "r.html"), os.path.join(tmp, "r.pdf")
    with open(hp, "w", encoding="utf-8") as f:
        f.write(html)
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--print-to-pdf-no-header", f"--print-to-pdf={pp}", hp],
                   timeout=60, capture_output=True)
    with open(pp, "rb") as f:
        return f.read()


def detect_colonies_ai(model, img_bgr, mask, conf, skip_dark=True):
    """AI(YOLO)로 콜로니 검출 → [(cx, cy, 지름px), ...] 반환.
    접시 안쪽만, 검은 글씨/마커(중심이 검은 것) 제외."""
    res = model.predict(img_bgr, conf=conf, verbose=False)[0]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dets = []
    h, w = mask.shape[:2]
    for b in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ix, iy = int(min(max(cx, 0), w - 1)), int(min(max(cy, 0), h - 1))
        if mask[iy, ix] == 0:          # 접시 밖 제외
            continue
        if skip_dark and gray[iy, ix] < 80:   # 검은 글씨/마커 위면 제외 (콜로니는 검지 않음)
            continue
        dets.append((cx, cy, float(max(x2 - x1, y2 - y1))))
    return dets


def measure_radius_at(img_bgr, agar_med, px, py, win=100):
    """클릭한 위치의 콜로니 크기를 측정해 반지름(px) 반환. 못 찾으면 None."""
    h, w = img_bgr.shape[:2]
    px, py = int(px), int(py)
    x0, y0 = max(0, px - win), max(0, py - win)
    x1, y1 = min(w, px + win), min(h, py + win)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(((lab - agar_med) ** 2).sum(2))
    binr = np.where(diff > 12, 255, 0).astype(np.uint8)
    binr = cv2.morphologyEx(binr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binr = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lx, ly = px - x0, py - y0
    if not (0 <= ly < binr.shape[0] and 0 <= lx < binr.shape[1]) or binr[ly, lx] == 0:
        return None
    num, comp = cv2.connectedComponents(binr)
    lbl = comp[ly, lx]
    area = int((comp == lbl).sum())
    if area < 6:
        return None
    return max(2, int((area / np.pi) ** 0.5))   # 면적 기반 반지름


def draw_markers(img_bgr, cx, cy, r, markers):
    """접시(초록 원) + 각 마커(콜로니 크기에 맞춘 링)를 그려서 RGB로 반환.
    각 마커는 [x, y, 반지름] 형태. 링이 콜로니 테두리를 따라감."""
    vis = img_bgr.copy()
    hh, ww = vis.shape[:2]
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 200, 0), 2)
    for m in markers:
        mx, my = int(m[0]), int(m[1])
        mr = int(m[2]) if len(m) > 2 else max(3, int(min(hh, ww) * 0.006))
        cv2.circle(vis, (mx, my), max(mr, 3), (57, 255, 20), 2)   # 콜로니 크기 링
        cv2.circle(vis, (mx, my), 1, (57, 255, 20), -1)          # 중심 점
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

# Edge(PDF 변환용)가 있는 환경(로컬 PC)에서만 PDF 기능 표시. 클라우드(리눅스)에선 자동 숨김.
EDGE_OK = any(os.path.exists(_e) for _e in (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"))

st.set_page_config(page_title="부유세균 콜로니 카운터", page_icon="🦠", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1150px; }
h1 { font-weight: 800 !important; letter-spacing: -1px; color: #0f172a; }
h3 {
  color: #0f766e !important; font-weight: 700 !important; font-size: 1.15rem !important;
  padding: 9px 0 9px 21px !important; margin-top: 0.6rem !important;
  border-left: 4px solid #0d9488;
  background: linear-gradient(90deg,#f0fdfa,rgba(240,253,250,0));
  border-radius: 0 8px 8px 0;
}
/* 소제목 안쪽 텍스트/앵커도 간격 확보 */
h3 a, h3 span, [data-testid="stHeadingWithActionElements"] { padding-left: 0 !important; }
/* 본문·위젯 라벨 글씨 키우기 (소제목과 균형) */
.stMarkdown p, label, [data-testid="stWidgetLabel"] p { font-size: 0.96rem; }
/* 사이드바 글씨 키우기 */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 1.0rem !important; }
[data-testid="stSidebar"] h2 { font-size: 1.2rem !important; color: #0f766e; }
/* 라디오 선택지(AI/고전) 글씨 키우기 */
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-size: 1.03rem !important; }
/* 슬라이더 라벨(민감도 등) 글씨 줄이기 */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] p { font-size: 0.86rem !important; }
/* 사이드바 위젯 위아래 간격 좁히기 */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
/* 사이드바 내용 위로 올리기 */
[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
/* 사이드바 안내(caption) 글씨는 작게 */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 0.78rem !important; color: #64748b; }
.stButton > button, .stDownloadButton > button {
  border-radius: 9px; font-weight: 600; border: 1px solid #cbd5e1;
  transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #0d9488; color: #0d9488; transform: translateY(-1px);
  box-shadow: 0 3px 10px rgba(13,148,136,.18);
}
[data-testid="stMetric"] {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
  padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
[data-testid="stMetricValue"] { color: #0f766e; font-weight: 800; }
table { border-radius: 8px; overflow: hidden; border-collapse: collapse; }
thead tr th { background: #0d9488 !important; color: #fff !important; font-weight: 600; }
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stFileUploader"] section {
  border: 1px dashed #cbd5e1; border-radius: 10px; background: #fbfdfd;
  padding: 0.4rem 0.8rem;
}
hr { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🦠 부유세균 콜로니 자동 카운팅")

with st.sidebar:
    st.header("측정 조건")
    media = st.selectbox("배지 종류", ["TSA", "SDA"],
                         help="TSA=세균용 배지, SDA=진균(곰팡이)용 배지")
    flow = st.number_input("유량 (L/min)", value=28.3, step=0.1)
    minutes = st.number_input("채취 시간 (분)", value=5.0, step=0.5)
    air_volume = flow * minutes
    st.info(f"채취 공기량 = **{air_volume:.1f} L**")
    dish_mm = st.number_input("페트리디시 지름 (mm)", value=90.0, step=1.0,
                              help="크기(mm) 계산 기준자. 보통 90mm 또는 100mm")
    use_feller = st.checkbox("Feller 양성공 보정 (N=400)", value=False,
                             help="끄면 현행 SOP(x7.07)와 동일")

    st.divider()
    st.header("검출 방식")
    ai_available = os.path.exists("model.pt")
    if ai_available:
        _method = st.radio("방식 선택", ["AI 검출 (추천)", "고전 영상처리"],
                           index=0, label_visibility="collapsed")
        use_ai = _method.startswith("AI")
    else:
        use_ai = False
        st.caption("model.pt 없음 → 고전 영상처리로 검출")

    if use_ai:
        ai_conf = st.slider("AI 민감도 (낮출수록 더 많이 잡음)", 0.05, 0.90, 0.25, 0.05)
        sensitivity, min_area, max_area, circularity_min, split_touching = 0, 25, 5000, 0.30, True
    else:
        ai_conf = 0.25
        st.caption("색·모양 규칙으로 검출해요. 아래 값으로 민감도를 조절하세요.")
        sensitivity = st.slider("민감도 (높일수록 흐린 것도 잡음)", -40, 60, 0, 5)
        min_area = st.slider("최소 콜로니 크기(px)", 2, 200, 25, 1)
        max_area = st.slider("최대 콜로니 크기(px)", 200, 12000, 5000, 100)
        circularity_min = st.slider("최소 원형도 (1=완전한 원)", 0.0, 1.0, 0.30, 0.05)
        split_touching = st.checkbox("붙은 콜로니 분리", value=True)

# 1) 업로드 ---------------------------------------------------------------
st.subheader("사진 올리기")
st.markdown("<div style='height:9px'></div>", unsafe_allow_html=True)
st.caption("아래 칸에 사진을 **끌어다 놓거나** 업로드 버튼으로 선택하세요. **여러 장** 가능.")
uploaded_files = st.file_uploader("페트리디시 사진",
                                  type=["jpg", "jpeg", "png", "bmp", "webp"],
                                  accept_multiple_files=True,
                                  label_visibility="collapsed")

if not uploaded_files:
    st.stop()

# 여러 장이면 이전/다음으로 넘겨보기
n_files = len(uploaded_files)
if "photo_idx" not in st.session_state:
    st.session_state.photo_idx = 0
st.session_state.photo_idx = min(st.session_state.photo_idx, n_files - 1)
if n_files > 1:
    nv1, nv2, nv3 = st.columns([1, 2, 1])
    if nv1.button("⬅️ 이전", use_container_width=True,
                  disabled=st.session_state.photo_idx == 0):
        st.session_state.photo_idx -= 1
        st.rerun()
    nv2.markdown(
        f"<div style='text-align:center;font-weight:700'>디시 {st.session_state.photo_idx+1} / {n_files}</div>"
        f"<div style='text-align:center;font-size:0.8em;color:#666'>{uploaded_files[st.session_state.photo_idx].name}</div>",
        unsafe_allow_html=True)
    if nv3.button("다음 ➡️", use_container_width=True,
                  disabled=st.session_state.photo_idx == n_files - 1):
        st.session_state.photo_idx += 1
        st.rerun()

uploaded = uploaded_files[st.session_state.photo_idx]
pil_img = load_resized(uploaded.getvalue())   # 캐시됨(한 번만 디코딩)

# 2) 검출 ----------------------------------------------------------------
st.subheader("콜로니 검출")
if "angle" not in st.session_state:
    st.session_state.angle = 0

view_col, edit_col = st.columns([1.9, 1])   # 왼쪽=검출이미지(크게), 오른쪽=편집(좁게)

with edit_col:
    st.markdown("**편집**")
    b1, b2, b3 = st.columns(3, gap="small")
    if b1.button("↺ 왼쪽", use_container_width=True):
        st.session_state.angle = (st.session_state.angle + 90) % 360
    if b2.button("↻ 오른쪽", use_container_width=True):
        st.session_state.angle = (st.session_state.angle - 90) % 360
    if b3.button("↩ 초기화", use_container_width=True):
        st.session_state.angle = 0
    work = pil_img.rotate(st.session_state.angle, expand=True)

    use_crop = st.checkbox("자르기", value=False, help="손글씨 뺄 때만")
    if use_crop:
        st.caption("초록 박스를 드래그해 남길 부분 선택")
        cropped = st_cropper(work, realtime_update=True, box_color="#00FF00", aspect_ratio=None)
        img = pil_to_bgr(cropped)
    else:
        img = pil_to_bgr(work)
    h, w = img.shape[:2]

    dish_sig = f"{st.session_state.angle}_{use_crop}_{w}x{h}"
    if use_crop or st.session_state.get("dish_sig") != dish_sig:
        acx, acy, ar = auto_detect_dish(img)          # 접시 검출도 캐시 → 클릭 땐 생략
        st.session_state.dish_cache = (acx, acy, ar)
        st.session_state.dish_sig = dish_sig
    else:
        acx, acy, ar = st.session_state.dish_cache
    with st.expander("접시 영역 미세조정"):
        st.caption(f"숫자는 **사진 속 픽셀(px) 위치**예요 (사진 크기 {w}×{h}px). 접시 원이 안 맞을 때만 조절.")
        use_manual = st.checkbox("수동 조정", value=False)
        if use_manual:
            cx = st.slider("중심 X (px)", 0, w, acx)
            cy = st.slider("중심 Y (px)", 0, h, acy)
            r = st.slider("반지름 (px)", 20, int(max(h, w)), ar)
        else:
            cx, cy, r = acx, acy, ar
            st.caption(f"자동: 중심({cx},{cy}) 반지름 {r}px")

mask = make_dish_mask(img.shape, cx, cy, r)

# 4) 검출 + 클릭 편집 -----------------------------------------------------
def _classical():
    cnts = detect_colonies(img, mask, sensitivity, min_area, max_area,
                           circularity_min, split_touching)
    cen, dia = [], []
    for c in cnts:
        (x, y), rad = cv2.minEnclosingCircle(c)
        cen.append([int(x), int(y), max(2, int(rad))])
        dia.append(2.0 * rad)
    return cen, dia

# 검출 조건이 바뀔 때만 다시 검출하고, 그 결과를 저장(캐시)해둔다.
# → 콜로니를 클릭(추가/제거)할 땐 재검출을 건너뛰어 아주 빠르게 반응한다.
det_sig = (f"{uploaded.name}_{uploaded.size}_{use_crop}_{st.session_state.angle}_{w}x{h}"
           f"_{use_ai}_{ai_conf}_{sensitivity}_{min_area}_{max_area}_{circularity_min}_{cx}_{cy}_{r}")
if st.session_state.get("det_sig") != det_sig:
    if use_ai:
        try:
            _model = load_ai_model("model.pt")
            _dets = detect_colonies_ai(_model, img, mask, ai_conf)
            detected = [[int(x), int(y), max(2, int(d / 2))] for (x, y, d) in _dets]
            diams_px = [d for (x, y, d) in _dets]
            det_method = "AI"
        except Exception as e:
            st.warning(f"AI 검출 실패({e}) → 고전 방식")
            detected, diams_px = _classical()
            det_method = "고전"
    else:
        detected, diams_px = _classical()
        det_method = "고전"
    st.session_state.det_cache = (detected, diams_px, det_method)
    st.session_state.det_sig = det_sig
    # 한천(배경) 대표색 캐시 → 클릭으로 콜로니 추가할 때 크기 측정에 사용
    _labf = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    _innr = cv2.erode(mask, np.ones((15, 15), np.uint8), 1)
    _bpts = _labf[_innr > 0]
    st.session_state.agar_med = (np.median(_bpts, axis=0) if len(_bpts)
                                 else np.array([150, 128, 128], np.float32))
detected, diams_px, det_method = st.session_state.det_cache

# 사진별 마커 저장 (여러 장 넘겨봐도 각 사진 편집이 유지됨)
photo_key = f"{uploaded.name}_{uploaded.size}"
mstore = st.session_state.setdefault("markers_store", {})
rstore = st.session_state.setdefault("reset_sig_store", {})
# 사진이 바뀌면 되돌리기 기록 초기화
if st.session_state.get("cur_photo_key") != photo_key:
    st.session_state.cur_photo_key = photo_key
    st.session_state.undo_stack = []
    st.session_state.redo_stack = []
# 검출민감도가 바뀌면 자동검출 반영, '접시(dish)만' 바꾸면 수동편집 유지
reset_sig = (f"{photo_key}_{use_crop}_{st.session_state.angle}_{w}x{h}"
             f"_{use_ai}_{ai_conf}_{sensitivity}_{min_area}_{max_area}_{circularity_min}")
if rstore.get(photo_key) != reset_sig:
    mstore[photo_key] = [list(p) for p in detected]
    rstore[photo_key] = reset_sig
markers = mstore[photo_key]

click = None
with view_col:
    st.caption("**수동 보정** — 빈 곳 클릭 → 콜로니 추가,  콜로니 클릭 → 제거")
    marked_rgb = draw_markers(img, cx, cy, r, markers)
    # 원본 크기 유지 + JPEG 고품질(92) → 화질 좋게, 그러면서 PNG보다 가벼워 빠름
    click = streamlit_image_coordinates(marked_rgb, use_column_width="always",
                                        image_format="JPEG", jpeg_quality=92,
                                        key="marker_canvas", cursor="crosshair")
    auto_count = len(detected)
    final_count = len(markers)
    st.info(f"자동 {auto_count}개 → 수정 후 **{final_count}개**")
    u1, u2 = st.columns(2)
    if u1.button("↶ 되돌리기", use_container_width=True,
                 disabled=not st.session_state.get("undo_stack")):
        st.session_state.setdefault("redo_stack", []).append([list(m) for m in markers])
        mstore[photo_key] = st.session_state.undo_stack.pop()
        st.rerun()
    if u2.button("↷ 다시하기", use_container_width=True,
                 disabled=not st.session_state.get("redo_stack")):
        st.session_state.setdefault("undo_stack", []).append([list(m) for m in markers])
        mstore[photo_key] = st.session_state.redo_stack.pop()
        st.rerun()
    bc1, bc2 = st.columns(2)
    if bc1.button("자동검출로 초기화", use_container_width=True):
        st.session_state.setdefault("undo_stack", []).append([list(m) for m in markers])
        st.session_state.redo_stack = []
        mstore[photo_key] = [list(p) for p in detected]
        st.rerun()
    if bc2.button("전체 지우기", use_container_width=True):
        st.session_state.setdefault("undo_stack", []).append([list(m) for m in markers])
        st.session_state.redo_stack = []
        mstore[photo_key] = []
        st.rerun()

# --- 클릭 처리 (새 클릭일 때만) ---
if click is not None and click.get("unix_time") != st.session_state.get("last_click_time"):
    st.session_state.last_click_time = click.get("unix_time")
    # 렌더 좌표 -> 실제 이미지 좌표로 변환
    rw = click.get("width") or w
    rh = click.get("height") or h
    ax = click["x"] * (w / rw)
    ay = click["y"] * (h / rh)
    # 콜로니 원 안(또는 근처)을 클릭하면 제거 → 제거를 쉽게
    base_tol = max(14, int(0.03 * min(h, w)))
    nearest_i, nearest_d = -1, 1e9
    for i, m in enumerate(markers):
        d = ((m[0] - ax) ** 2 + (m[1] - ay) ** 2) ** 0.5
        if d < nearest_d:
            nearest_i, nearest_d = i, d
    tol = base_tol
    if nearest_i >= 0 and len(markers[nearest_i]) > 2:
        tol = max(base_tol, markers[nearest_i][2] + 8)   # 콜로니 크기만큼 넉넉히
    # 되돌리기용 스냅샷 (수정 직전 상태 저장)
    st.session_state.setdefault("undo_stack", []).append([list(m) for m in markers])
    if len(st.session_state.undo_stack) > 60:
        st.session_state.undo_stack.pop(0)
    st.session_state.redo_stack = []
    if nearest_i >= 0 and nearest_d <= tol:
        markers.pop(nearest_i)          # 콜로니 클릭 -> 제거
    else:
        # 클릭한 자리의 실제 콜로니 크기를 측정해서 반영
        r_meas = measure_radius_at(img, st.session_state.get("agar_med"), ax, ay) \
            if st.session_state.get("agar_med") is not None else None
        if r_meas is None:   # 측정 실패 시 기존 검출들의 평균 크기
            r_meas = int(np.median([m[2] for m in markers if len(m) > 2])) if any(len(m) > 2 for m in markers) else max(4, int(0.01 * min(h, w)))
        markers.append([int(ax), int(ay), r_meas])  # 빈 곳 클릭 -> 추가
    st.rerun()

# 3) 최종 결과 (표) -------------------------------------------------------
st.subheader("측정 결과")
ic1, ic2, ic3 = st.columns(3)
user_name = ic1.text_input("측정자 (User)", value=st.session_state.get("user_name", ""))
sample_id = ic2.text_input("시료/디시 번호", value="")
measure_dt = ic3.text_input("측정일시",
                            value=st.session_state.get("measure_dt", ""),
                            placeholder="예: 2026-07-08 14:30",
                            help="디시를 실제로 채취한 때")
st.session_state.user_name = user_name
st.session_state.measure_dt = measure_dt
comment = st.text_input("코멘트 (Comment)", value="")

if use_feller:
    corrected = feller_correction(final_count, N=400)
    method = "Feller 보정 (N=400)"
else:
    corrected = final_count
    method = "보정 없음 (현행 SOP)"
cfu = cfu_per_m3(corrected, air_volume)

mm_per_px = dish_mm / (2.0 * r) if r > 0 else 0.0
# 화면에 찍힌 마커(=최종 콜로니) 기준으로 크기 계산
marker_diams_px = [2 * m[2] for m in markers if len(m) > 2]
diams_mm = [d * mm_per_px for d in marker_diams_px]
if diams_mm:
    _a = np.array(diams_mm)
    size_str = f"평균 {_a.mean():.1f} mm  (최소 {_a.min():.1f} · 최대 {_a.max():.1f})"
else:
    size_str = "콜로니 없음"
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

# 핵심 지표(왼쪽) / 상세 정보(오른쪽)
key_rows = [
    ("배지", media),
    ("채취 공기량", f"{air_volume:.1f} L"),
    ("자동 검출 수", f"{auto_count} 개"),
    ("최종 콜로니 수", f"{final_count} 개"),
    ("총부유세균 농도", f"{cfu:,.0f} CFU/m³"),
]
if use_feller:
    key_rows.insert(4, ("보정 후 균 수", f"{corrected:.0f}"))
detail_rows = [
    ("측정자(User)", user_name or "-"),
    ("시료/디시 번호", sample_id or "-"),
    ("검출 방식", det_method),
    ("콜로니 크기", size_str),
    ("측정일시(채취)", measure_dt or "-"),
    ("분석일시(자동)", now_str),
    ("코멘트", comment or "-"),
]

rt1, rt2 = st.columns([1, 1.25])
with rt1:
    st.markdown("**핵심 결과**")
    st.table({"항목": [k for k, v in key_rows], "값": [v for k, v in key_rows]})
with rt2:
    st.markdown("**상세 정보**")
    st.table({"항목": [k for k, v in detail_rows], "값": [v for k, v in detail_rows]})

rows = key_rows + detail_rows   # 저장 리포트용

# 여러 장 올렸을 때 전체 디시 요약
if n_files > 1:
    _factor = 1000.0 / air_volume if air_volume > 0 else 0.0
    with st.expander(f"전체 디시 요약 ({n_files}장)", expanded=True):
        _srows = []
        for _i, _f in enumerate(uploaded_files):
            _k = f"{_f.name}_{_f.size}"
            _mark = "◀ 지금" if _i == st.session_state.photo_idx else ""
            if _k in mstore:
                _cnt = len(mstore[_k])
                _srows.append((f"{_i+1}", _f.name, f"{_cnt}개", f"{_cnt*_factor:,.0f}", _mark))
            else:
                _srows.append((f"{_i+1}", _f.name, "미확인", "-", _mark))
        st.table({"#": [s[0] for s in _srows], "파일": [s[1] for s in _srows],
                  "콜로니 수": [s[2] for s in _srows], "CFU/m³": [s[3] for s in _srows],
                  "": [s[4] for s in _srows]})
        st.caption("'이전/다음'으로 각 디시를 넘겨보며 확인·수정하면 여기 채워져요. (CFU는 개수×환산, 보정 미적용)")

# 4) 결과 저장 ------------------------------------------------------------
st.subheader("저장 · 다운로드")
# 저장용 데이터 미리 준비
_marked = draw_markers(img, cx, cy, r, markers)
_marked_png = cv2.imencode(".png", cv2.cvtColor(_marked, cv2.COLOR_RGB2BGR))[1].tobytes()
_after_jpg = cv2.imencode(".jpg", cv2.cvtColor(_marked, cv2.COLOR_RGB2BGR))[1].tobytes()
_clean_jpg = cv2.imencode(".jpg", img)[1].tobytes()
_label_lines = []
for m in markers:
    mx, my = m[0], m[1]
    mr = m[2] if len(m) > 2 else max(4, int(0.01 * min(h, w)))
    _label_lines.append(f"0 {mx/w:.6f} {my/h:.6f} {2*mr/w:.6f} {2*mr/h:.6f}")
_label_txt = "\n".join(_label_lines)
_report = "\n".join([f"{k}: {v}" for k, v in rows])
_base = (sample_id or os.path.splitext(uploaded.name)[0]).strip().replace(" ", "_") or "result"

col_L, col_R = st.columns(2)
with col_L:
    st.markdown("**콜로니 크기 분포**")
    if diams_mm:
        _df = pd.DataFrame({"크기(mm)": [round(float(x), 2) for x in diams_mm]})
        _lo = max(0.0, float(min(diams_mm)) - 0.3)
        _hi = float(max(diams_mm)) + 0.3
        _chart = (alt.Chart(_df)
                  .mark_circle(size=80, opacity=0.5, color="#3b82f6")
                  .encode(
                      x=alt.X("크기(mm):Q",
                              scale=alt.Scale(domain=[_lo, _hi], nice=True),
                              axis=alt.Axis(labelAngle=0, tickMinStep=0.5, format=".1f",
                                            title="콜로니 크기 (mm)")),
                      y=alt.Y("jitter:Q", axis=None),
                      tooltip=[alt.Tooltip("크기(mm):Q", title="크기(mm)")])
                  .transform_calculate(jitter="random()")
                  .properties(height=200))
        st.altair_chart(_chart, use_container_width=True)
        st.caption("점 하나 = 콜로니 하나 · 가로 = 크기(mm)")
    else:
        st.caption("검출된 콜로니 없음")
with col_R:
    st.markdown("**결과 다운로드**")
    if EDGE_OK:   # PDF 리포트는 Edge 필요 → 클라우드에선 자동 숨김
        if st.button("결과 PDF 리포트 만들기", use_container_width=True):
            with st.spinner("PDF 만드는 중..."):
                try:
                    _pdf = html_to_pdf(build_report_html(key_rows, detail_rows, _clean_jpg, _after_jpg,
                                                         "부유세균 콜로니 카운팅 결과"))
                    st.session_state.report_pdf = _pdf
                    st.session_state.report_pdf_name = f"{_base}_리포트.pdf"
                except Exception as e:
                    st.error(f"PDF 생성 실패: {e}")
        if st.session_state.get("report_pdf"):
            st.download_button("PDF 리포트 내려받기", st.session_state.report_pdf,
                               file_name=st.session_state.get("report_pdf_name", "리포트.pdf"),
                               mime="application/pdf", use_container_width=True)
    st.download_button("검출 후 사진", _marked_png,
                       file_name=f"{_base}_marked.png", mime="image/png", use_container_width=True)
    st.download_button("표 내용(txt)", _report,
                       file_name=f"{_base}_report.txt", use_container_width=True)

st.divider()
st.markdown("**AI 재학습용으로 저장**")
st.caption("콜로니 정답(위치)을 labeled_data 폴더에 저장해요. 쌓이면 나중에 AI를 다시 학습시켜요.")
sv1, sv2 = st.columns(2)
if sv1.button("이 디시 저장", use_container_width=True):
    os.makedirs("labeled_data/images", exist_ok=True)
    os.makedirs("labeled_data/labels", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{_base}_{stamp}"
    with open(f"labeled_data/images/{fn}.jpg", "wb") as f:
        f.write(_clean_jpg)
    with open(f"labeled_data/labels/{fn}.txt", "w") as f:
        f.write(_label_txt)
    st.success("이 디시 저장됨!")

if n_files > 1:
    _n_reviewed = sum(1 for _f in uploaded_files if f"{_f.name}_{_f.size}" in mstore)
    if sv2.button(f"검토한 디시 모두 저장 ({_n_reviewed}장)", use_container_width=True):
        os.makedirs("labeled_data/images", exist_ok=True)
        os.makedirs("labeled_data/labels", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        for _f in uploaded_files:
            _k = f"{_f.name}_{_f.size}"
            if _k not in mstore:
                continue
            _imgf = pil_to_bgr(load_resized(_f.getvalue()).rotate(st.session_state.angle, expand=True))
            _hf, _wf = _imgf.shape[:2]
            _lines = []
            for m in mstore[_k]:
                mx, my = m[0], m[1]
                mr = m[2] if len(m) > 2 else max(4, int(0.01 * min(_hf, _wf)))
                _lines.append(f"0 {mx/_wf:.6f} {my/_hf:.6f} {2*mr/_wf:.6f} {2*mr/_hf:.6f}")
            _bn = os.path.splitext(_f.name)[0].strip().replace(" ", "_") or "dish"
            _fn = f"{_bn}_{stamp}"
            with open(f"labeled_data/images/{_fn}.jpg", "wb") as ff:
                ff.write(cv2.imencode(".jpg", _imgf)[1].tobytes())
            with open(f"labeled_data/labels/{_fn}.txt", "w") as ff:
                ff.write("\n".join(_lines))
            saved += 1
        st.success(f"검토한 {saved}장 모두 저장됨! (자르기 쓴 사진은 개별 저장 권장)")

with st.expander("ℹ️ 검출 방식 안내"):
    st.markdown(
        """
- **AI 방식**: 학습된 YOLO 모델(model.pt)로 검출. 장치사진엔 강하지만
  휴대폰 사진(밝은 배경)엔 약할 수 있어요. → 클릭으로 수정 후 **저장**하면 재학습 데이터가 쌓여요.
- **고전 방식**: "배지색과 다른 노란 동그라미 찾기" 규칙. 왼쪽 슬라이더로 조절.
- 어느 방식이든 **클릭으로 최종 확정**하는 게 핵심이에요.
        """
    )
