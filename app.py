"""Ballet calc: A0, B0, R からスカート型紙（正方形の一辺 L）を求め、
Blender のクロスシミュレーションでスカートの3次元形状を生成する Streamlit アプリ."""

import base64
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.patches import Arc, Circle
from scipy.optimize import fsolve


# ---------------------------------------------------------------- 計算
def equations(x, A, B):
    """sample.m の myEquations と同じ連立方程式 (x = [alpha, beta, theta, L])."""
    alpha, beta, theta, L = x
    return [
        L**2 - (A**2 + B**2 - 2 * A * B * math.cos(theta)),
        L**2 - 2 * B**2 * (1 - math.cos(beta)),
        L**2 - 2 * A**2 * (1 - math.cos(alpha)),
        alpha + beta + 2 * theta - 2 * math.pi,
    ]


def solve(A0, B0, R):
    """解析解で L と中心高さ h を求め、sample.m の式を fsolve で解いて検証する.

    中心 P は正方形の縦中心線上 (x = L/2) にあり、
      (L/2)^2 + h^2       = B^2
      (L/2)^2 + (L - h)^2 = A^2
    を満たす。u = L^2, D = A^2 - B^2 とおくと
      2u^2 - (2D + 4B^2)u + D^2 = 0
    となり、P が正方形内部 (0 <= h < L) となる根を採用する。
    """
    A, B = A0 + R, B0 + R
    D = A**2 - B**2
    k = 2 * D + 4 * B**2
    disc = k**2 - 8 * D**2
    if disc < 0:
        return None

    for u in sorted({(k + math.sqrt(disc)) / 4, (k - math.sqrt(disc)) / 4}, reverse=True):
        if u <= 0:
            continue
        L = math.sqrt(u)
        h = (u - D) / (2 * L)
        if 0 <= h < L:
            break
    else:
        return None

    # 余弦定理による角度 (初期値) → sample.m と同じ式を fsolve で解く
    alpha0 = math.acos(max(-1.0, min(1.0, 1 - L**2 / (2 * A**2))))
    beta0 = math.acos(max(-1.0, min(1.0, 1 - L**2 / (2 * B**2))))
    theta0 = math.acos(max(-1.0, min(1.0, (A**2 + B**2 - L**2) / (2 * A * B))))
    x, info, ier, _ = fsolve(equations, [alpha0, beta0, theta0, L], args=(A, B), full_output=True)
    return dict(A=A, B=B, L=x[3], h=h, alpha=x[0], beta=x[1], theta=x[2],
                residual=np.abs(info["fvec"]).max(), converged=ier == 1)


# ---------------------------------------------------------------- 描画
DIM = "#2e7d32"   # 寸法線 (緑)
GEO = "#37474f"   # 形状線
INP = "#1565c0"   # 入力値 (青)
OUT = "#c62828"   # 出力値 (赤)


def dim_line(ax, p1, p2, offset, text, color=DIM, fontsize=11, weight="normal"):
    """p1-p2 間に、法線方向へ offset ずらした寸法線を描く."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    q1, q2 = p1 + n * offset, p2 + n * offset
    for p, q in ((p1, q1), (p2, q2)):  # 補助線
        ax.plot(*zip(p + n * np.sign(offset) * 0.5, q + n * np.sign(offset) * 1.0),
                color=color, lw=0.6, alpha=0.6)
    ax.annotate("", xy=q2, xytext=q1,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.0, shrinkA=0, shrinkB=0))
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    mid = (q1 + q2) / 2
    ax.text(*mid, text, color=color, fontsize=fontsize, fontweight=weight, rotation=ang,
            rotation_mode="anchor", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8))


def angle_arc(ax, center, v1, v2, radius, label, color):
    """center で v1→v2 (反時計回り) の角度に円弧とラベルを描く."""
    a1 = math.degrees(math.atan2(v1[1], v1[0]))
    a2 = math.degrees(math.atan2(v2[1], v2[0]))
    if a2 < a1:
        a2 += 360
    ax.add_patch(Arc(center, 2 * radius, 2 * radius, theta1=a1, theta2=a2, color=color, lw=1.0))
    mid = math.radians((a1 + a2) / 2)
    ax.text(center[0] + 1.45 * radius * math.cos(mid), center[1] + 1.45 * radius * math.sin(mid),
            label, color=color, fontsize=9, ha="center", va="center")


def draw(A0, B0, R, sol, show_angles):
    L, h, A, B = sol["L"], sol["h"], sol["A"], sol["B"]
    P = np.array([L / 2, h])
    BL, BR, TL, TR = map(np.array, ([0, 0], [L, 0], [0, L], [L, L]))
    s = L / 70  # 図のスケールに合わせたオフセット単位

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(plt.Polygon([BL, BR, TR, TL], closed=True, fc="#e3f2fd", ec=GEO, lw=1.5))
    ax.add_patch(Circle(P, R, fill=False, ec=GEO, lw=1.2, ls="--"))

    for corner in (BL, BR, TL, TR):
        ax.plot(*zip(corner, P), color=GEO, lw=1.3)
        u = (P - corner) / np.linalg.norm(P - corner)
        ax.plot(*(corner + u * (np.linalg.norm(P - corner) - R)), "o", color=GEO, ms=4)
    ax.plot(*P, "o", color=OUT, ms=6)

    # 出力 L (上辺・左辺)
    dim_line(ax, TL, TR, 5 * s, f"L = {L:.2f}", color=OUT, fontsize=13, weight="bold")
    dim_line(ax, BL, TL, 5 * s, f"L = {L:.2f}", color=OUT, fontsize=11)

    # 入力 A0 (左上の角→円周), A = A0 + R
    uA = (P - TL) / A
    dim_line(ax, TL, TL + uA * A0, -3 * s, f"A0 = {A0:g}", color=INP, fontsize=11, weight="bold")
    dim_line(ax, TL, P, 7 * s, f"A0+R = {A:g}", color=DIM, fontsize=9)

    # 入力 B0 (左下の角→円周), B = B0 + R
    uB = (P - BL) / B
    dim_line(ax, BL, BL + uB * B0, 3 * s, f"B0 = {B0:g}", color=INP, fontsize=11, weight="bold")
    dim_line(ax, BL, P, -4 * s, f"B0+R = {B:g}", color=DIM, fontsize=9)

    # 入力 R (右上方向への半径)
    rdir = np.array([math.cos(math.radians(60)), math.sin(math.radians(60))])
    ax.annotate("", xy=P + rdir * R, xytext=P,
                arrowprops=dict(arrowstyle="-|>", color=INP, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(*(P + rdir * R + np.array([1.5, 1.0]) * s), f"R = {R:g}", color=INP,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INP, lw=0.8))

    # 中心の高さ h (右辺側)
    dim_line(ax, np.array([L, 0]), np.array([L, h]), -4 * s, f"h = {h:.3f}", color=DIM, fontsize=9)

    if show_angles:
        r = 6 * s
        angle_arc(ax, P, TR - P, TL - P, r, f"α={math.degrees(sol['alpha']):.1f}°", "#6a1b9a")
        angle_arc(ax, P, BL - P, BR - P, r, f"β={math.degrees(sol['beta']):.1f}°", "#6a1b9a")
        angle_arc(ax, P, TL - P, BL - P, r, f"θ={math.degrees(sol['theta']):.1f}°", "#6a1b9a")

    m = 12 * s
    ax.set_xlim(-m, L + m)
    ax.set_ylim(min(-m, h - R - 2 * s), L + m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- スカート生成 (Blender)
ROOT = Path(__file__).resolve().parent
SIM_SCRIPT = ROOT / "skirt" / "simulate_patterns.py"
RUNS_DIR = ROOT / "runs"

# 既定の3つの型紙（外側→内側）と布の色（Blender のリニア RGB）
DEFAULT_SETS = [
    dict(A0=25.0, B0=20.0, R=2.5, color=(0.80, 0.72, 0.02)),
    dict(A0=30.0, B0=20.0, R=2.5, color=(0.03, 0.22, 0.10)),
    dict(A0=40.0, B0=30.0, R=2.5, color=(0.10, 0.07, 0.65)),
]
SET_LABELS = ["型紙1（外側）", "型紙2（中間）", "型紙3（内側）"]


def linear_to_hex(rgb):
    def enc(c):
        c = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, c)) * 255)
    return "#" + "".join(f"{enc(c):02x}" for c in rgb)


def hex_to_linear(hx):
    def dec(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return [round(dec(int(hx[i:i + 2], 16) / 255), 4) for i in (1, 3, 5)]


def blender_command():
    """Blender 実行コマンドを探す。見つからなければ pip の bpy、どちらもなければ None."""
    candidates = [os.environ.get("BLENDER_PATH"), shutil.which("blender"), shutil.which("Blender"),
                  "/Applications/Blender.app/Contents/MacOS/Blender",
                  r"C:\Program Files\Blender Foundation\Blender\blender.exe"]
    for c in candidates:
        if c and Path(c).exists():
            return [c, "-b", "--factory-startup", "-P", str(SIM_SCRIPT), "--"]
    if importlib.util.find_spec("bpy") is not None:
        return [sys.executable, str(SIM_SCRIPT), "--"]
    return None


def fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}分{sec % 60:02d}秒" if sec >= 60 else f"{sec}秒"


def run_simulation(patterns, frames, fast, render):
    """simulate_patterns.py を Blender で実行し、進捗をプログレスバーに表示する."""
    cmd = blender_command()
    if cmd is None:
        st.error("Blender が見つかりません。Blender をインストールするか、環境変数 BLENDER_PATH を設定してください。")
        return None

    run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "patterns.json").write_text(json.dumps(patterns, ensure_ascii=False, indent=1), encoding="utf-8")
    cmd = cmd + ["--patterns", str(run_dir / "patterns.json"), "--out-dir", str(run_dir),
                 "--out", "skirt", "--frames", str(frames)]
    if fast:
        cmd.append("--fast")
    if not render:
        cmd.append("--no-render")

    # 全体に占める各フェーズの割合
    w_setup, w_sim, w_render = 0.03, (0.72 if render else 0.92), (0.2 if render else 0.0)
    bar = st.progress(0.0, text="Blender を起動しています…")
    status = st.empty()
    log_box = st.expander("Blender のログ", expanded=False)
    log_area = log_box.empty()
    log_lines = []

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, cwd=run_dir)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            el = time.time() - t0
            if line.startswith("@@PROGRESS "):
                p = json.loads(line[len("@@PROGRESS "):])
                ph = p["phase"]
                if ph == "sim":
                    frac = w_setup + w_sim * p["done"] / p["total"]
                    bar.progress(frac, text=f"布のシミュレーション中: 段 {p['tier']}/{p['tiers']}  "
                                            f"フレーム {p['done']}/{p['total']}  ({frac * 100:.0f}%)")
                    status.caption(f"経過 {fmt_time(el)} ／ 残り約 {fmt_time(p['eta'])}"
                                   f"{'（＋レンダリング）' if render else ''} ／ {p['sec_per_frame']:.2f} 秒/フレーム")
                elif ph == "render":
                    k, n = p.get("sample", 0), p.get("samples", 0)
                    frac = w_setup + w_sim + (w_render * k / n if n else 0)
                    bar.progress(frac, text=f"レンダリング中（Cycles）: サンプル {k}/{n}  ({frac * 100:.0f}%)"
                                 if n else "レンダリング中（Cycles）…")
                    status.caption(f"経過 {fmt_time(el)}")
                elif ph == "export":
                    bar.progress(0.97, text="GLB を書き出し中…")
                elif ph == "done":
                    bar.progress(1.0, text="完了")
                continue
            log_lines.append(line)
            log_area.code("\n".join(log_lines[-40:]), language=None)
        proc.wait()
    finally:
        if proc.poll() is None:  # 画面操作などで中断されたら Blender も止める
            proc.kill()

    glb = run_dir / "skirt.glb"
    if proc.returncode != 0 or not glb.exists():
        st.error(f"シミュレーションに失敗しました（終了コード {proc.returncode}）。ログを確認してください。")
        log_box.code("\n".join(log_lines[-200:]), language=None)
        return None
    status.caption(f"完了しました（所要時間 {fmt_time(time.time() - t0)}）")
    png = run_dir / "skirt.png"
    return dict(dir=str(run_dir), glb=str(glb), png=str(png) if png.exists() else None)


def show_glb(path, height=640):
    """<model-viewer> で GLB をマウス操作できる3D表示にする."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    components.html(f"""
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
<model-viewer src="data:model/gltf-binary;base64,{data}" camera-controls auto-rotate
  shadow-intensity="1" exposure="1.0" environment-image="neutral" camera-orbit="30deg 80deg auto"
  style="width:100%;height:{height - 10}px;background:#f3f4f6;border-radius:8px"></model-viewer>
""", height=height)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Ballet calc", layout="wide")
st.title("スカート型紙の計算と3次元形状の生成")
st.caption("A0: 上の角〜円周、B0: 下の角〜円周、R: 穴（ウエスト）の半径。"
           "A = A0 + R, B = B0 + R として sample.m と同じ連立方程式を解き、正方形の一辺 L を求めます。")

with st.sidebar:
    st.header("型紙のパラメータ")
    sets = []
    for i, (label, d) in enumerate(zip(SET_LABELS, DEFAULT_SETS)):
        st.subheader(label)
        c1, c2, c3 = st.columns(3)
        A0 = c1.number_input("A0", min_value=0.0, value=d["A0"], step=1.0, format="%.2f", key=f"A0_{i}")
        B0 = c2.number_input("B0", min_value=0.0, value=d["B0"], step=1.0, format="%.2f", key=f"B0_{i}")
        R = c3.number_input("R", min_value=0.0, value=d["R"], step=0.5, format="%.2f", key=f"R_{i}")
        color = st.color_picker("布の色", linear_to_hex(d["color"]), key=f"color_{i}")
        sets.append(dict(A0=A0, B0=B0, R=R, color=color))
    st.divider()
    show_angles = st.checkbox("角度 α, β, θ を表示", value=True)

sols = [solve(p["A0"], p["B0"], p["R"]) for p in sets]

tab_pat, tab_skirt = st.tabs(["型紙", "スカート（3D）"])

with tab_pat:
    cols = st.columns(3)
    for col, label, p, sol in zip(cols, SET_LABELS, sets, sols):
        with col:
            st.subheader(f"{label}  A0={p['A0']:g}, B0={p['B0']:g}")
            if sol is None:
                st.error("この A0, B0, R では中心が正方形の内側に入る解がありません。")
                continue
            m1, m2 = st.columns(2)
            m1.metric("L", f"{sol['L']:.2f}")
            m2.metric("中心の高さ h", f"{sol['h']:.3f}")
            fig = draw(p["A0"], p["B0"], p["R"], sol, show_angles)
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("詳細"):
                st.markdown(
                    f"""
| 変数 | 値 |
|---|---|
| A = A0 + R | {sol['A']:g} |
| B = B0 + R | {sol['B']:g} |
| α | {math.degrees(sol['alpha']):.4f}° |
| β | {math.degrees(sol['beta']):.4f}° |
| θ | {math.degrees(sol['theta']):.4f}° |
| 残差 max\\|F\\| | {sol['residual']:.2e} |
""")
                if not sol["converged"]:
                    st.warning("fsolve が収束しませんでした。")

with tab_skirt:
    st.markdown("3枚の型紙を穴の位置でウエストに重ねて留め、内側の段から順に Blender のクロスシミュレーションで垂らします。"
                "計算には数分かかります。")
    c1, c2, c3 = st.columns(3)
    frames = c1.slider("フレーム数（段ごと）", 30, 200, 110, step=10,
                       help="多いほど布が落ち着きますが時間がかかります（元のスクリプトは 110）")
    fast = c2.checkbox("軽量モード", value=False, help="布同士の当たり判定を省き、メッシュを粗くして高速化します")
    render = c3.checkbox("Cycles でレンダリング画像も作る", value=False, help="さらに数分かかります")

    ok = all(s is not None for s in sols)
    if blender_command() is None:
        st.warning("この環境では Blender が見つからないため、スカートを生成できません。")
    if not ok:
        st.warning("解のない型紙があります。パラメータを見直してください。")
    if st.button("スカートを生成", type="primary", disabled=not ok or blender_command() is None):
        patterns = [dict(name=f"tier{i + 1}", L=round(sol["L"], 4), h=round(sol["h"], 4), R=p["R"],
                         color=hex_to_linear(p["color"]))
                    for i, (p, sol) in enumerate(zip(sets, sols))]
        result = run_simulation(patterns, frames, fast, render)
        if result:
            result["params"] = [dict(A0=p["A0"], B0=p["B0"], R=p["R"]) for p in sets]
            st.session_state["skirt"] = result

    res = st.session_state.get("skirt")
    if res:
        params = ", ".join(f"({p['A0']:g}, {p['B0']:g}, R={p['R']:g})" for p in res["params"])
        st.subheader("生成結果")
        st.caption(f"型紙（外側→内側, A0, B0）: {params}")
        if res["params"] != [dict(A0=p["A0"], B0=p["B0"], R=p["R"]) for p in sets]:
            st.info("現在のパラメータとは異なる条件で生成した結果です。")
        if res["png"]:
            v1, v2 = st.columns([3, 2])
            with v1:
                show_glb(res["glb"])
            v2.image(res["png"], caption="Cycles レンダリング")
        else:
            show_glb(res["glb"])
        d1, d2 = st.columns(2)
        d1.download_button("GLB をダウンロード", Path(res["glb"]).read_bytes(), "skirt.glb", "model/gltf-binary")
        if res["png"]:
            d2.download_button("画像をダウンロード", Path(res["png"]).read_bytes(), "skirt.png", "image/png")
