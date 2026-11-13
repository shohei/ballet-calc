"""
型紙（正方形＋中心をずらした半径2.5cmの穴）からスカートを作り、クロスシミュレーションで垂らす。

型紙の寸法（画像 25_20.png / 30_20.png / 40_30.png より, 単位 cm）:
  L  : 正方形の一辺
  h  : 穴の中心の、下辺からの高さ（左右方向は中央）
  → 上の2角までの距離 = A0+R, 下の2角までの距離 = B0+R, R = 2.5

実行例:
  Blender -b -P simulate_patterns.py -- --tiers 25_20,30_20,40_30 --out skirt_sim
  Blender -b -P simulate_patterns.py -- --tiers 40_30 --out sim_40_30
  # 任意の型紙（Streamlit アプリから呼ぶ形式）
  Blender -b -P simulate_patterns.py -- --patterns patterns.json --out-dir runs/xxx

patterns.json は外側→内側の順のリスト:
  [{"name": "t1", "L": 35.17, "h": 14.034, "R": 2.5, "color": [0.8, 0.72, 0.02]}, ...]

進捗は標準出力に "@@PROGRESS {json}" 形式でも出力する（アプリ側で解析）。
"""
import bpy
import json
import bmesh
import math
import os
import sys
import argparse
import time

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# 実寸（ウエスト周り15.7cm）だとBlenderのクロスが硬くなりすぎるため、
# S倍（≒大人サイズ）で計算し、GLB書き出し時に実寸へ戻す
S = 4.0
CM = 0.01 * S

# ------------------------------------------------------------------ 型紙データ
R_HOLE = 2.5
PATTERNS = {
    #  name     L       h       色
    "25_20": (35.17, 14.034, (0.80, 0.72, 0.02)),
    "30_20": (38.19, 11.897, (0.03, 0.22, 0.10)),
    "40_30": (52.54, 19.134, (0.10, 0.07, 0.65)),
}
# 型紙の上辺側（A0 = 長い側）を体の後ろ(+Y)に向ける
# 各段の回転（度）。0 なら全段の角がそろう。
TIER_ROTATION = {"25_20": 0.0, "30_20": 0.0, "40_30": 0.0}

WAIST_Z = 0.45 * S      # ウエスト（穴）の高さ（実寸45cm）
N_AROUND = 200          # 周方向分割
N_RADIAL = 44           # 穴→外周の分割
FRAMES = 110


def progress(**kw):
    """アプリ向けの機械可読な進捗行"""
    print("@@PROGRESS " + json.dumps(kw, ensure_ascii=False), flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", default="25_20,30_20,40_30",
                   help="外側→内側の順にカンマ区切り")
    p.add_argument("--out", default="skirt_sim")
    p.add_argument("--out-dir", default=OUT_DIR, help="出力先フォルダ")
    p.add_argument("--patterns", default=None,
                   help="型紙定義 JSON ファイル（指定時は --tiers より優先）")
    p.add_argument("--rot", default=None,
                   help="各段の回転(度) カンマ区切り。例 0,30,-30")
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="軽量設定（布自身の当たり判定なし・計算精度とメッシュを粗く）")
    return p.parse_args(argv)


# ------------------------------------------------------------------ 型紙 → メッシュ
def ray_to_square(phi, xmin, xmax, ymin, ymax):
    """原点から角度phiの半直線が長方形外周に当たる距離"""
    c, s = math.cos(phi), math.sin(phi)
    ts = []
    if c > 1e-12:
        ts.append(xmax / c)
    if c < -1e-12:
        ts.append(xmin / c)
    if s > 1e-12:
        ts.append(ymax / s)
    if s < -1e-12:
        ts.append(ymin / s)
    return min(ts)


def pattern_mesh(name, L, h, z, rot_deg, ring_extra=0.0, r_hole=R_HOLE):
    """穴の中心を原点にした型紙（cm）を、極座標状の四角メッシュで作る。
    外周の4つの角が必ず頂点になるように角度サンプルを調整する。"""
    xmin, xmax = -L / 2, L / 2
    ymin, ymax = -h, L - h
    corners = [math.atan2(y, x) % (2 * math.pi)
               for x, y in ((xmax, ymax), (xmin, ymax), (xmin, ymin), (xmax, ymin))]
    phis = [2 * math.pi * i / N_AROUND for i in range(N_AROUND)]
    for c in corners:  # 最も近いサンプルを角にスナップ
        k = min(range(N_AROUND), key=lambda i: abs((phis[i] - c + math.pi) % (2 * math.pi) - math.pi))
        phis[k] = c
    phis.sort()

    rot = math.radians(rot_deg)
    bm = bmesh.new()
    grid = []
    for j in range(N_RADIAL + 1):
        t = j / N_RADIAL
        row = []
        for phi in phis:
            D = ray_to_square(phi, xmin, xmax, ymin, ymax)
            # ring_extra: 重ねた段のウエスト縫い代ぶり、ピン位置だけ外へ（布の形はそのまま）
            r = r_hole + (D - r_hole) * t
            r_place = r + ring_extra * (1 - t)
            x, y = r_place * math.cos(phi), r_place * math.sin(phi)
            xr = x * math.cos(rot) - y * math.sin(rot)
            yr = x * math.sin(rot) + y * math.cos(rot)
            row.append(bm.verts.new((xr * CM, yr * CM, z)))
        grid.append(row)
    n = len(phis)
    for j in range(N_RADIAL):
        for i in range(n):
            i2 = (i + 1) % n
            # 表面（法線）が上＝垂れた後は外側を向くように頂点順を決める
            bm.faces.new((grid[j][i], grid[j + 1][i], grid[j + 1][i2], grid[j][i2]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    for p in me.polygons:
        p.use_smooth = True
    # ウエスト（穴の縁）をピン留め
    vg = obj.vertex_groups.new(name="Pin")
    vg.add(list(range(n)), 1.0, "REPLACE")
    return obj


# ------------------------------------------------------------------ マテリアル
def make_material(name, color, chiffon=True):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = 0.55
    if chiffon:
        b.inputs["Sheen Weight"].default_value = 0.15
        b.inputs["Sheen Tint"].default_value = (*color, 1)
        b.inputs["Transmission Weight"].default_value = 0.08
    return mat


# ------------------------------------------------------------------ ボディ（ドレスフォーム）
def build_dress_form():
    """1/4スケール程度のトルソー（ウエスト周り ≒ 穴の周長 15.7cm）。
    プロファイル: (ウエストからの高さcm, 半径cm)"""
    prof = [(-13.5, 0.3), (-13.0, 1.8), (-11.0, 3.2), (-8.0, 3.7), (-5.0, 3.6),
            (-2.5, 3.0), (0.0, 2.35), (3.0, 2.6), (7.0, 3.5), (10.0, 3.9),
            (13.0, 3.6), (16.0, 3.4), (18.0, 2.0), (19.0, 1.0), (20.0, 0.9), (23.0, 0.9)]
    seg = 64
    bm = bmesh.new()
    rings = []
    for dz, r in prof:
        ring = []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            ring.append(bm.verts.new((r * CM * math.cos(a), 0.82 * r * CM * math.sin(a),
                                      WAIST_Z + dz * CM)))
        rings.append(ring)
    for j in range(len(rings) - 1):
        for i in range(seg):
            i2 = (i + 1) % seg
            bm.faces.new((rings[j][i], rings[j][i2], rings[j + 1][i2], rings[j + 1][i]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    me = bpy.data.meshes.new("DressForm")
    bm.to_mesh(me)
    bm.free()
    form = bpy.data.objects.new("DressForm", me)
    bpy.context.collection.objects.link(form)
    for p in me.polygons:
        p.use_smooth = True
    form.data.materials.append(make_material("Mat_Form", (0.02, 0.02, 0.025), chiffon=False))
    col = form.modifiers.new("Collision", "COLLISION")
    form.collision.thickness_outer = 0.001 * S
    form.collision.cloth_friction = 8

    mat_stand = make_material("Mat_Stand", (0.6, 0.6, 0.62), chiffon=False)
    mat_stand.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 1
    mat_stand.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.25
    top = WAIST_Z - 13.5 * CM
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.004 * S, depth=top, location=(0, 0, top / 2))
    pole = bpy.context.active_object
    pole.name = "Stand_Pole"
    pole.data.materials.append(mat_stand)
    pole.modifiers.new("Collision", "COLLISION")
    pole.collision.thickness_outer = 0.002 * S
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.10 * S, depth=0.01 * S, location=(0, 0, 0.005 * S))
    base = bpy.context.active_object
    base.name = "Stand_Base"
    base.data.materials.append(mat_stand)
    base.modifiers.new("Collision", "COLLISION")


# ------------------------------------------------------------------ クロス設定（シフォン相当）
def add_cloth(obj, fast=False):
    m = obj.modifiers.new("Cloth", "CLOTH")
    s = m.settings
    s.quality = 8 if fast else 10
    s.mass = 0.15
    s.air_damping = 1.0
    s.tension_stiffness = 15
    s.compression_stiffness = 15
    s.shear_stiffness = 5
    s.bending_stiffness = 0.05
    s.tension_damping = 3
    s.compression_damping = 3
    s.shear_damping = 3
    s.bending_damping = 0.2
    s.vertex_group_mass = "Pin"
    s.pin_stiffness = 1.0
    c = m.collision_settings
    c.collision_quality = 3 if fast else 4
    c.distance_min = (0.0008 if fast else 0.0012) * S
    c.use_self_collision = not fast
    c.self_distance_min = 0.0010 * S
    c.self_friction = 5
    m.point_cache.frame_start = 1
    m.point_cache.frame_end = bpy.context.scene.frame_end


def freeze_tier(obj):
    """計算し終わった段を、その形のまま固定メッシュにして衝突物にする"""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)
    obj.modifiers.new("Collision", "COLLISION")
    obj.collision.thickness_outer = 0.0010 * S
    obj.collision.cloth_friction = 5
    obj.collision.use_culling = False   # 両面で衝突（片面だと法線側へ押し出される）
    sol = obj.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0004 * S
    sol.offset = 0
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 1


# ------------------------------------------------------------------ シーン
def setup_scene(frames):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = 24

    bpy.ops.mesh.primitive_plane_add(size=4 * S, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(make_material("Mat_Floor", (0.85, 0.85, 0.87), chiffon=False))

    for loc, rot, e, size in [((0.7, -0.9, 1.1), (50, 0, 38), 60, 0.8),
                              ((-0.9, -0.5, 0.8), (65, 0, -60), 30, 1.0),
                              ((0, 0.9, 1.0), (-50, 0, 0), 55, 0.6)]:
        bpy.ops.object.light_add(type="AREA", location=[v * S for v in loc])
        l = bpy.context.active_object
        l.data.energy = e * S * S
        l.data.size = size * S
        l.rotation_euler = [math.radians(a) for a in rot]

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.95, 0.95, 0.97, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35

    bpy.ops.object.camera_add(location=(0.50 * S, -1.00 * S, 0.34 * S))
    cam = bpy.context.active_object
    cam.data.lens = 50
    tgt = bpy.data.objects.new("CamTarget", None)
    bpy.context.collection.objects.link(tgt)
    tgt.location = (0, 0, 0.27 * S)
    tr = cam.constraints.new("TRACK_TO")
    tr.target = tgt
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"
    scene.camera = cam

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 1200
    scene.view_settings.view_transform = "Standard"


def main():
    args = parse_args()
    if args.patterns:
        with open(args.patterns, encoding="utf-8") as f:
            defs = json.load(f)
        tiers = []
        for d in defs:
            PATTERNS[d["name"]] = (float(d["L"]), float(d["h"]), tuple(d["color"]), float(d.get("R", R_HOLE)))
            tiers.append(d["name"])
    else:
        tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    rots = ([float(r) for r in args.rot.split(",")] if args.rot
            else [TIER_ROTATION.get(t, 0.0) for t in tiers])

    global N_AROUND, N_RADIAL
    if args.fast:
        N_AROUND, N_RADIAL = 144, 32
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_scene(args.frames)
    build_dress_form()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    blend = os.path.join(out_dir, args.out + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)

    scene = bpy.context.scene
    log_path = os.path.join(out_dir, args.out + "_progress.log")
    log = open(log_path, "w")

    def report(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # 内側の段から1枚ずつ計算 → 固定 → その上に次の段を落とす。
    # 布どうしが同時に動く衝突を避けるので、全部一度に計算するより大幅に速い。
    n = len(tiers)
    order = list(reversed(list(enumerate(zip(tiers, rots)))))
    frames = scene.frame_end - scene.frame_start + 1
    total = frames * n
    report(f"start: tiers(外→内)={tiers} fast={args.fast} frames/段={frames}")
    t0 = time.time()
    done = 0
    frozen = []
    from collections import deque
    recent = deque([t0], maxlen=11)
    for step, (k, (tname, rot)) in enumerate(order, 1):
        L, h, color, *rest = PATTERNS[tname]
        r_hole = rest[0] if rest else R_HOLE
        z = WAIST_Z + (n - 1 - k) * 0.004 * S   # 外側の段ほど上から落とす
        # 重ねた段はウエストの縫い代ぶん（0.1cm/段）だけピン位置を外へ
        ring_extra = (n - 1 - k) * 0.1
        obj = pattern_mesh("Tier_" + tname, L, h, z, rot, ring_extra=ring_extra, r_hole=r_hole)
        obj.data.materials.append(make_material("Mat_" + tname, color))
        add_cloth(obj, fast=args.fast)
        t1 = time.time()
        for f in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(f)
            done += 1
            now = time.time()
            el = now - t0
            recent.append(now)
            rate = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)  # 直近の1フレーム秒数
            eta = rate * (total - done)
            bar = "#" * (done * 30 // total)
            progress(phase="sim", tier=step, tiers=n, name=tname, done=done, total=total,
                     sec_per_frame=round(rate, 3), elapsed=round(el, 1), eta=round(eta, 1))
            report(f"段{step}/{n} {tname} [{bar:<30}] {done}/{total}  "
                   f"{rate:4.1f}秒/フレーム  経過 {el/60:4.1f}分  残り約 {eta/60:4.1f}分")
        freeze_tier(obj)
        frozen.append(obj)
        scene.frame_set(scene.frame_start)
        report(f"段{step}/{n} {tname} 完了 ({(time.time()-t1):.0f}秒)")
    bpy.ops.wm.save_mainfile()

    report("simulation done -> rendering")
    if not args.no_render:
        progress(phase="render")

        def on_stats(stats):
            # 例: "Fra:110 | ... | Sample 12/96" → アプリ向けに進捗を出す
            import re
            m = re.search(r"Sample (\d+)/(\d+)", stats)
            if m:
                progress(phase="render", sample=int(m.group(1)), samples=int(m.group(2)))
        bpy.app.handlers.render_stats.append(on_stats)
        scene.render.filepath = os.path.join(out_dir, args.out + ".png")
        bpy.ops.render.render(write_still=True)

    # 最終フレームの形状を実寸（1/S）で GLB に
    root = bpy.data.objects.new("Skirt_RealScale", None)
    bpy.context.collection.objects.link(root)
    root.scale = (1 / S,) * 3
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name != "Floor":
            o.parent = root
            o.select_set(True)
    progress(phase="export")
    bpy.ops.export_scene.gltf(filepath=os.path.join(out_dir, args.out + ".glb"),
                              export_format="GLB", use_selection=True, export_apply=True)
    report("all done: " + args.out + ".png / .blend / .glb")
    progress(phase="done", elapsed=round(time.time() - t0, 1))
    log.close()


main()
