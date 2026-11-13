"""Ballet calc: A0, B0, R から正方形の一辺 L を求める Streamlit アプリ (sample.m の移植)."""

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
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


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Ballet calc", layout="wide")
st.title("正方形の一辺 L の計算")
st.caption("A0: 上の角〜円周、B0: 下の角〜円周、R: 円の半径。A = A0 + R, B = B0 + R として sample.m と同じ連立方程式を解きます。")

with st.sidebar:
    st.header("入力")
    A0 = st.number_input("A0", min_value=0.0, value=40.0, step=1.0, format="%.3f")
    B0 = st.number_input("B0", min_value=0.0, value=30.0, step=1.0, format="%.3f")
    R = st.number_input("R", min_value=0.0, value=2.5, step=1.0, format="%.3f")
    show_angles = st.checkbox("角度 α, β, θ を表示", value=True)

sol = solve(A0, B0, R)
if sol is None:
    st.error("この A0, B0, R では中心が正方形の内側に入る解がありません。値を見直してください。")
    st.stop()

col_fig, col_res = st.columns([3, 1])
with col_res:
    st.metric("L", f"{sol['L']:.4f}")
    st.metric("中心の高さ h", f"{sol['h']:.4f}")
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
"""
    )
    if not sol["converged"]:
        st.warning("fsolve が収束しませんでした。")
with col_fig:
    st.pyplot(draw(A0, B0, R, sol, show_angles))
