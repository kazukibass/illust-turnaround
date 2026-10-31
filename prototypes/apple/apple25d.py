"""2.5D rotation prototype on an apple.

Stand-in data
  A "true" 3D apple (squashed sphere body with longitudinal stripes and one brown mark, a stem, a leaf)
  is drawn as an illustration - flat two-tone cel shading from a light fixed to the viewer, an outline,
  a highlight - one RGBA image per layer (body / stem / leaf). The same drawer gives
    * the multi-view data the rig is built from (front, right, back, left: yaw 0/90/180/270), and
    * the ground truth at any in-between angle, for scoring.

The 2.5D rig (the idea under test)
  * one 2D mesh per (layer, view), cut from that view's drawing;
  * every vertex is lifted to 3D on a proxy (body: the squashed sphere; stem: its axis, as a billboard;
    leaf: its plane);
  * at an angle the vertices are rotated in 3D and projected; body vertices that turn past the rim are
    folded onto the silhouette ("畳み込み"), so nothing needs culling;
  * the two views nearest the angle are cross-faded (weights linear in angle), both lying on the same
    proxy so their outlines coincide;
  * layers are ordered by their mean depth.
  Everything the renderer needs per angle is vertex positions, opacity and draw order - exactly what a
  moc3 / Inochi2D keyform holds - so the rig is also baked at 15-degree keys and checked for bake error.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

W = H = 512
CX, CY = 256.0, 300.0
A, B, C = 150.0, 140.0, 150.0            # body semi-axes (x, y, z)
LIGHT = np.array([-0.5, -0.65, 0.58]); LIGHT /= np.linalg.norm(LIGHT)
MARK = (np.radians(40), np.radians(10))  # longitude, latitude of the brown mark
VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]
BAND = 10       # degrees of longitude per body patch
LAT_BAND = 20   # degrees of latitude per body patch (each patch = one mesh with its own opacity keys)
_lon = np.radians((np.arange(360 // BAND) + 0.5) * BAND)
_lat = np.radians(-90 + (np.arange(180 // LAT_BAND) + 0.5) * LAT_BAND)
_LO, _LA = np.meshgrid(_lon, _lat, indexing="ij")
PATCH_NORMALS = np.column_stack([(np.cos(_LA) * np.sin(_LO)).ravel(), (-np.sin(_LA)).ravel(), (np.cos(_LA) * np.cos(_LO)).ravel()])


def patch_of(points):
    n = points / np.array([A, B, C])
    lon = np.degrees(np.arctan2(n[:, 0], n[:, 2])) % 360
    lat = np.degrees(np.arcsin(np.clip(-n[:, 1] / np.linalg.norm(n, axis=1), -1, 1)))
    li = np.clip(np.floor(lon / BAND), 0, 360 // BAND - 1).astype(int)
    ai = np.clip(np.floor((lat + 90) / LAT_BAND), 0, 180 // LAT_BAND - 1).astype(int)
    return li * (180 // LAT_BAND) + ai
SHARP = 6.0     # how strongly a squarely-seen view wins over a grazing one
CAM_BIAS = 0.2  # floor of the "close to the current camera" factor
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"


# ---------------------------------------------------------------- 3D model (world frame: x right, y down, z to viewer at yaw 0)
def rot(yaw, pitch=0.0):
    """World -> view rotation for a camera turned by yaw (about y) and pitch (about x)."""
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry


def to_px(p):
    return np.column_stack([p[:, 0] + CX, p[:, 1] + CY])


STEM = np.array([[0.0, -B * 0.97, 0.0], [10.0, -B * 0.97 - 55.0, -4.0]])
LEAF_BASE = STEM[0] + 0.55 * (STEM[1] - STEM[0])
LEAF_DIR = np.array([1.0, -0.2, 0.55]); LEAF_DIR /= np.linalg.norm(LEAF_DIR)
LEAF_SIDE = np.cross(LEAF_DIR, np.array([0.0, -1.0, 0.0])); LEAF_SIDE /= np.linalg.norm(LEAF_SIDE)
LEAF_SIDE = 0.75 * LEAF_SIDE + 0.25 * np.array([0.0, -1.0, 0.0]); LEAF_SIDE /= np.linalg.norm(LEAF_SIDE)
LEAF_N = np.cross(LEAF_DIR, LEAF_SIDE); LEAF_N /= np.linalg.norm(LEAF_N)


def leaf_outline(n=40):
    t = np.linspace(0, 1, n)
    half = 26 * np.sin(np.pi * t) ** 0.8
    top = LEAF_BASE + np.outer(t * 85, LEAF_DIR) + np.outer(half, LEAF_SIDE)
    bot = LEAF_BASE + np.outer(t[::-1] * 85, LEAF_DIR) - np.outer(half[::-1], LEAF_SIDE)
    return np.vstack([top, bot])


# ---------------------------------------------------------------- illustration drawer (ground truth + view data)
def draw_body(yaw, pitch=0.0, part="all"):
    R = rot(yaw, pitch)
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    # ray along view z through pixel; body in view frame is the rotated ellipsoid: solve for front hit
    Rinv = R.T
    d = Rinv @ np.array([0, 0, 1.0])                       # view z in world
    o = np.stack([xs - CX, ys - CY, np.zeros_like(xs)], -1) @ Rinv.T   # pixel origin in world
    S = np.array([1 / A, 1 / B, 1 / C])
    od, dd = o * S, d * S
    qa = (dd ** 2).sum()
    qb = 2 * (od * dd).sum(-1)
    qc = (od ** 2).sum(-1) - 1
    disc = qb ** 2 - 4 * qa * qc
    hit = disc > 0
    t = (-qb + np.sqrt(np.clip(disc, 0, None))) / (2 * qa)    # far root along +z = nearest to viewer
    p = o + t[..., None] * d
    nw = p * S * S; nw /= np.linalg.norm(nw, axis=-1, keepdims=True) + 1e-9
    nv = nw @ R.T                                          # normal in view frame
    lon = np.arctan2(p[..., 0], p[..., 2])
    lat = np.arcsin(np.clip(-p[..., 1] / B, -1, 1))
    base = np.array([205, 38, 48.0])
    stripe = (0.5 + 0.5 * np.sin(9 * lon)) * np.clip(1 - np.abs(lat) / 1.2, 0, 1)
    col = base * (1 - 0.35 * stripe[..., None]) + np.array([240, 170, 60.0]) * 0.35 * stripe[..., None]
    dm = np.hypot((lon - MARK[0] + np.pi) % (2 * np.pi) - np.pi, lat - MARK[1])
    col[dm < 0.16] = [120, 70, 35]
    lit = (nv @ LIGHT) > 0.25
    hl = (nv @ (LIGHT + np.array([0, 0, 1.0])) / np.linalg.norm(LIGHT + np.array([0, 0, 1.0]))) > 0.985
    rim = nv[..., 2] < 0.22
    img = np.zeros((H, W, 4), np.uint8)
    if part == "pattern":                     # base colour, stripes, mark: attached to the surface
        img[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
        img[..., 3] = np.where(hit, 255, 0)
        return img
    # lighting: cel shadow (a translucent dark layer, like a multiply layer), highlight and outline
    shade = np.zeros((H, W, 4), float)
    shade[~lit] = [70, 10, 20, 0.45 * 255]
    shade[hl] = [255, 245, 240, 255]
    shade[rim] = [90, 20, 25, 255]
    shade[..., 3] *= hit
    if part == "light":
        return shade.astype(np.uint8)
    a = shade[..., 3:4] / 255
    col = col * (1 - a) + shade[..., :3] * a
    img[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
    img[..., 3] = np.where(hit, 255, 0)
    return img


def draw_stem(yaw, pitch=0.0):
    R = rot(yaw, pitch)
    p = to_px(STEM @ R.T)
    im = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(im)
    d.line([tuple(p[0]), tuple(p[1])], fill=(60, 35, 20, 255), width=14)
    d.line([tuple(p[0]), tuple(p[1])], fill=(120, 80, 45, 255), width=8)
    return np.array(im)


def draw_leaf(yaw, pitch=0.0):
    R = rot(yaw, pitch)
    p = to_px(leaf_outline() @ R.T)
    facing = (R @ LEAF_N)[2] > 0
    im = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(im)
    d.polygon([tuple(q) for q in p], fill=(70, 160, 60, 255) if facing else (45, 115, 45, 255), outline=(25, 60, 25, 255))
    mid = to_px(np.vstack([LEAF_BASE, LEAF_BASE + 85 * LEAF_DIR]) @ R.T)
    d.line([tuple(mid[0]), tuple(mid[1])], fill=(35, 90, 35, 255), width=2)
    return np.array(im)


LAYERS = {"body": draw_body, "stem": draw_stem, "leaf": draw_leaf}
# what the rig is built from: the body split as an illustrator would (paint / shading layers)
RIG_LAYERS = {"body": lambda y: draw_body(y, part="pattern"), "light": lambda y: draw_body(y, part="light"),
              "stem": draw_stem, "leaf": draw_leaf}


def layer_depth(name, yaw, pitch=0.0):
    R = rot(yaw, pitch)
    if name == "body":
        return 0.0
    if name == "light":
        return 1e-3                       # right above the body pattern
    pts = STEM if name == "stem" else leaf_outline()
    # stem and leaf sit on top: compare with the body's depth right under them (its top point)
    top = np.array([[0, -B, 0.0]]) @ R.T
    return float((pts @ R.T)[:, 2].mean() - top[0, 2])


def composite(layers):
    """layers: list of (depth, rgba float 0..1 premultiplied). Far first."""
    out = np.ones((H, W, 3))
    for _, im in sorted(layers, key=lambda t: t[0]):
        a = im[..., 3:4]
        out = im[..., :3] + out * (1 - a)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def premul(img):
    f = img.astype(float) / 255
    f[..., :3] *= f[..., 3:4]
    return f


def truth(yaw, pitch=0.0):
    return composite([(layer_depth(n, yaw, pitch), premul(fn(yaw, pitch))) for n, fn in LAYERS.items()])


# ---------------------------------------------------------------- the 2.5D rig
def mesh_from_alpha(alpha, step=10, threshold=127):
    m = (alpha > threshold).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in cs:
        c = c[:, 0, :].astype(float)
        pts += list(c[:: max(1, len(c) // max(8, int(cv2.arcLength(c.astype(np.float32), True) / step)))])
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    inside = m[ys, xs] > 0
    pts += list(np.column_stack([xs[inside], ys[inside]]).astype(float))
    pts = np.unique(np.round(np.array(pts), 1), axis=0)
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(1).round().astype(int)
    keep = cv2.dilate(m, np.ones((3, 3), np.uint8))[np.clip(cen[:, 1], 0, H - 1), np.clip(cen[:, 0], 0, W - 1)] > 0
    return pts, tri[keep]


def body_hit(R, x, y):
    """Front intersection of the view ray through (x, y) with the body ellipsoid, in world coordinates.
    Rays that miss (outline pixels just outside) are clamped onto the silhouette."""
    d = R.T @ np.array([0, 0, 1.0])
    o = np.column_stack([x, y, np.zeros_like(x)]) @ R
    S = np.array([1 / A, 1 / B, 1 / C])
    od, dd = o * S, d * S
    qa = (dd ** 2).sum()
    qb = 2 * (od * dd).sum(-1)
    qc = (od ** 2).sum(-1) - 1
    t = (-qb + np.sqrt(np.clip(qb ** 2 - 4 * qa * qc, 0, None))) / (2 * qa)
    return o + t[:, None] * d


def lift(name, view, px):
    """Pixel positions in a view image -> 3D world points on the layer's proxy."""
    R = rot(*view)
    x, y = px[:, 0] - CX, px[:, 1] - CY
    if name == "body":
        return body_hit(R, x, y)
    elif name == "light":
        pv = np.column_stack([x, y, np.zeros_like(x)])
        return pv                                          # kept in screen space
    elif name == "stem":
        s = (STEM @ R.T)
        d = s[1] - s[0]
        t = np.clip(((np.column_stack([x, y]) - s[0, :2]) @ d[:2]) / (d[:2] @ d[:2]), 0, 1)
        pv = np.column_stack([x, y, s[0, 2] + t * d[2]])     # billboard at the axis depth
    else:
        n = R @ LEAF_N
        p0 = R @ LEAF_BASE
        nz = n[2] if abs(n[2]) > 1e-3 else 1e-3
        z = p0[2] - (n[0] * (x - p0[0]) + n[1] * (y - p0[1])) / nz
        pv = np.column_stack([x, y, z])
    return pv @ R                                          # view -> world (R orthonormal)


def project(name, world, yaw, pitch=0.0):
    if name == "light":
        return to_px(world)
    pv = world @ rot(yaw, pitch).T
    if name == "body":
        # fold: points turned past the rim go radially onto the silhouette ellipse
        back = pv[:, 2] < 0
        p = np.radians(pitch)
        ay = np.hypot(B * np.cos(p), C * np.sin(p))
        r = np.sqrt((pv[back, 0] / A) ** 2 + (pv[back, 1] / ay) ** 2)
        pv[back, :2] /= np.maximum(r, 1e-6)[:, None]
    return to_px(pv)


def warp(tex, src, dst, tris):
    out = np.zeros((H, W, 4), np.float32)
    for t in tris:
        s, d = src[t].astype(np.float32), dst[t].astype(np.float32)
        if abs((d[1, 0] - d[0, 0]) * (d[2, 1] - d[0, 1]) - (d[1, 1] - d[0, 1]) * (d[2, 0] - d[0, 0])) < 0.5:
            continue
        x0, y0 = np.floor(d.min(0)).astype(int) - 1
        x1, y1 = np.ceil(d.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        m = cv2.getAffineTransform(s, d - np.float32([x0, y0]))
        patch = cv2.warpAffine(tex, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round((d - [x0, y0]) * 4).astype(np.int32), 255, shift=2)
        sel = mask > 0
        out[y0:y1, x0:x1][sel] = patch[sel]
    return out


def view_weights(yaw, views):
    """Cross-fade weights over the horizontal views (pitch 0), linear in yaw. Keys are view tuples."""
    yaw %= 360
    w = {}
    vs = sorted(v[0] for v in views if v[1] == 0)
    for i, v in enumerate(vs):
        nxt = vs[(i + 1) % len(vs)] + (360 if i + 1 == len(vs) else 0)
        span = nxt - v
        d = (yaw - v) % 360
        if d < span:
            w[(v, 0)] = w.get((v, 0), 0) + 1 - d / span
            w[(nxt % 360, 0)] = w.get((nxt % 360, 0), 0) + d / span
    return {k: x for k, x in w.items() if x > 1e-6}


class Rig:
    def __init__(self, views):
        self.views = views
        self.parts = {}
        for name, fn in RIG_LAYERS.items():
            for v in views:
                if name != "body" and v[1] != 0:
                    continue                      # top/bottom views only feed the body surface
                img = draw_body(*v, part="pattern") if name == "body" else fn(v[0])
                if (img[..., 3] > 0).sum() < 30:
                    continue
                # the lighting layer is translucent and has small shapes (highlight, thin outline)
                px, tris = mesh_from_alpha(img[..., 3], step=4, threshold=0) if name == "light" \
                    else mesh_from_alpha(img[..., 3])
                world = lift(name, v, px)
                part = dict(tex=premul(img).astype(np.float32), px=px, tris=tris, world=world)
                if name == "body":
                    # split into surface patches (longitude x latitude); every patch is its own mesh
                    # with its own opacity keys
                    part["band"] = patch_of(world[tris].mean(1))
                self.parts[(name, v)] = part

    def pose(self, yaw, pitch=0.0):
        """Per-part vertex positions, opacity (per body band), draw depth: the keyform content."""
        w = view_weights(yaw, self.views)
        # body: every view that saw a surface band may draw it, weighted by how squarely it saw it
        # (normal . view direction, sharpened so that grazing, outline-heavy parts barely count) and a
        # little by how close that view is to the current camera (keeps view-dependent shading);
        # normalised per band
        nrm = PATCH_NORMALS
        dcam = rot(yaw, pitch).T @ np.array([0, 0, 1.0])
        raw = {}
        for v in self.views:
            dv = rot(*v).T @ np.array([0, 0, 1.0])
            raw[v] = np.clip(nrm @ dv, 0, None) ** SHARP * (CAM_BIAS + max(0.0, float(dv @ dcam))) ** 2
        facing = (nrm @ dcam) > -0.05            # bands turned away are folded onto the rim: draw nothing
        raw = {v: x * facing for v, x in raw.items()}
        tot = sum(raw.values())
        band_w = {v: np.where(tot > 1e-9, raw[v] / np.maximum(tot, 1e-9), 0.0) for v in self.views}
        out = {}
        for (name, v), p in self.parts.items():
            op = band_w[v] if name == "body" else w.get(v, 0.0)
            out[(name, v)] = dict(pos=project(name, p["world"], yaw, pitch), opacity=op,
                                  depth=layer_depth(name, yaw, pitch))
        return out

    def render(self, pose):
        layers = {}
        for (name, v), k in pose.items():
            p = self.parts[(name, v)]
            op = np.asarray(k["opacity"], float)
            if op.max() <= 0:
                continue
            if op.ndim == 0:
                im = warp(p["tex"], p["px"], k["pos"], p["tris"]) * float(op)
            else:
                im = np.zeros((H, W, 4), np.float32)
                for b in np.unique(p["band"]):
                    if op[b] > 0:
                        im += warp(p["tex"], p["px"], k["pos"], p["tris"][p["band"] == b]) * float(op[b])
            if name in layers:
                layers[name] = (layers[name][0], layers[name][1] + im)
            else:
                layers[name] = (k["depth"], im)
        # views of one layer are blended by weight; where their coverage overlaps unevenly (rasterised
        # triangle edges, the fold at the rim) renormalise so that the layer never exceeds full opacity
        fixed = []
        for name, (depth, im) in layers.items():
            a = im[..., 3:4]
            scale = np.where(a > 1, 1 / np.maximum(a, 1e-6), 1.0)
            im = im * scale
            if name == "body":
                # pinholes between rasterised triangles of different views: fill from the neighbours
                blur = cv2.blur(im, (3, 3))
                hole = (im[..., 3] < 0.98) & (blur[..., 3] > 0.5)
                im[hole] = blur[hole] / blur[hole][:, 3:4]
            fixed.append((depth, im))
        return composite(fixed)


def bake(rig, step=15):
    keys = list(range(0, 360, step))
    return keys, {k: rig.pose(k) for k in keys}


def baked_pose(keys, table, yaw, step=15):
    k0 = int(yaw // step) * step % 360
    k1 = (k0 + step) % 360
    f = (yaw % step) / step
    a, b = table[k0], table[k1]
    return {part: dict(pos=a[part]["pos"] * (1 - f) + b[part]["pos"] * f,
                       opacity=np.asarray(a[part]["opacity"]) * (1 - f) + np.asarray(b[part]["opacity"]) * f,
                       depth=a[part]["depth"] * (1 - f) + b[part]["depth"] * f) for part in a}


# ---------------------------------------------------------------- scoring
def mark_pos(img):
    m = (np.abs(img[..., 0].astype(int) - 120) < 12) & (np.abs(img[..., 1].astype(int) - 70) < 12) & (np.abs(img[..., 2].astype(int) - 35) < 12)
    # the mark under the cel shadow (45 % dark layer)
    m |= (np.abs(img[..., 0].astype(int) - 98) < 14) & (np.abs(img[..., 1].astype(int) - 43) < 14) & (np.abs(img[..., 2].astype(int) - 28) < 14)
    m |= (np.abs(img[..., 0].astype(int) - 74) < 12) & (np.abs(img[..., 1].astype(int) - 43) < 12) & (np.abs(img[..., 2].astype(int) - 22) < 12)
    yy, xx = np.mgrid[0:H, 0:W]
    m &= ((xx - CX) / A) ** 2 + ((yy - CY) / B) ** 2 < 0.97 ** 2     # on the body only (not the stem)
    ys, xs = np.where(m)
    return (xs.mean(), ys.mean(), len(xs)) if len(xs) > 20 else None


def score(ref, img):
    fg_r = (ref < 250).any(-1)
    fg_i = (img < 250).any(-1)
    iou = (fg_r & fg_i).sum() / max((fg_r | fg_i).sum(), 1)
    both = fg_r & fg_i
    err = np.abs(ref.astype(int) - img.astype(int))[both].mean() if both.any() else 0
    mr, mi = mark_pos(ref), mark_pos(img)
    mark = None
    if mr and mi:
        mark = float(np.hypot(mr[0] - mi[0], mr[1] - mi[1]))
    return dict(iou=round(float(iou), 4), color_err=round(float(err), 2), mark_err=None if mark is None else round(mark, 1))


def strip(frames, labels, cols):
    th = 170
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * th, rows * (th + 16)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (f, l) in enumerate(zip(frames, labels)):
        x, y = (i % cols) * th, (i // cols) * (th + 16)
        sheet.paste(Image.fromarray(f).resize((th, th), Image.LANCZOS), (x, y + 16))
        d.text((x + 4, y + 2), l, fill=(0, 0, 0))
    return sheet


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    report = {}
    angles = list(range(0, 360, 30))
    test_angles = [a + 15 for a in angles] + [a + 7.5 for a in angles]
    rigs = {"4views": Rig(VIEWS), "2views": Rig([(0, 0), (180, 0)]),
            "6views": Rig(VIEWS + [(0, 90), (0, -90)])}
    rig = rigs["4views"]
    keys, table = bake(rig)
    for tag, r in rigs.items():
        rows = {}
        for a in angles + test_angles:
            rows[a] = score(truth(a), r.render(r.pose(a)))
        report[tag] = rows
    report["4views_baked"] = {a: score(truth(a), rig.render(baked_pose(keys, table, a))) for a in test_angles}
    pitched = [(y, p) for y in (0, 45, 90, 180) for p in (-45, -25, 25, 45)]
    report["4views_pitch"] = {f"{y}/{p}": score(truth(y, p), rig.render(rig.pose(y, p))) for y, p in pitched}
    six = rigs["6views"]
    report["6views_pitch"] = {f"{y}/{p}": score(truth(y, p), six.render(six.pose(y, p))) for y, p in pitched}

    # pictures: truth vs rig around the turn, the view data, and a turntable GIF
    frames, labels = [], []
    for a in range(0, 360, 30):
        b = a + 15
        frames += [truth(b), rig.render(rig.pose(b))]
        labels += [f"truth {b}", f"2.5D {b}"]
    strip(frames, labels, 8).save(os.path.join(OUT, "compare_inbetween.png"))
    six_views = VIEWS + [(0, 90), (0, -90)]
    strip([truth(*v) for v in six_views], [f"view yaw {v[0]} pitch {v[1]}" for v in six_views], 6).save(os.path.join(OUT, "views.png"))
    fr, lb = [], []
    for y, p in [(45, -45), (45, 45), (180, -45)]:
        fr += [truth(y, p), rig.render(rig.pose(y, p)), six.render(six.pose(y, p))]
        lb += [f"truth {y}/{p}", f"4 views {y}/{p}", f"6 views {y}/{p}"]
    strip(fr, lb, 3).save(os.path.join(OUT, "pitch.png"))
    gif = [Image.fromarray(rig.render(baked_pose(keys, table, a))).resize((256, 256)) for a in np.arange(0, 360, 5)]
    gif[0].save(os.path.join(OUT, "turntable.gif"), save_all=True, append_images=gif[1:], duration=60, loop=0)
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)

    def summary(rows, keys_=None):
        v = [rows[k] for k in (keys_ or rows)]
        m = [x["mark_err"] for x in v if x["mark_err"] is not None]
        return (np.mean([x["iou"] for x in v]), np.mean([x["color_err"] for x in v]), np.mean(m) if m else float("nan"), max(m) if m else float("nan"))
    print("                       IoU    color_err  mark_err(mean/max px)")
    for tag, rows, ks in (("4 views @ drawn views", report["4views"], angles[::3]),
                          ("4 views @ in-between", report["4views"], test_angles),
                          ("4 views baked 15deg", report["4views_baked"], None),
                          ("2 views @ in-between", report["2views"], test_angles),
                          ("4 views with pitch", report["4views_pitch"], None),
                          ("6 views with pitch", report["6views_pitch"], None)):
        s = summary(rows, ks)
        print(f"{tag:22s} {s[0]:.3f}  {s[1]:8.2f}   {s[2]:.1f} / {s[3]:.1f}")
