"""
make_figure.py — schematic "how to annotate" illustration for the real MACS clips (como_anotar.png).
Six panels: parts of the clip, what to mark, open jaws, tip hidden behind the aneurysm, blurred edge,
common mistakes; plus a key strip. Drawn at 2x and downsampled for anti-aliasing.

Usage: python make_figure.py   ->  real_eval/kit/como_anotar.png
"""

import os
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SS = 2
W, H = 1800 * SS, 1180 * SS
FONT_DIR = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size * SS)
    return ImageFont.load_default()


INK = (33, 33, 38)
MUTED = (95, 95, 105)
TISSUE = (238, 205, 198)
DOME = (214, 120, 128)
HEAD = (38, 104, 214)
HEAD_HI = (150, 205, 255)
JAW = (52, 56, 66)
APPLIER = (128, 130, 136)
APPLIER_DK = (92, 94, 100)
RED = (226, 30, 40)
BLUE = (20, 140, 255)
MAG = (205, 0, 205)
YELLOW = (240, 190, 0)
OK = (20, 150, 70)
BAD = (215, 30, 30)


class Clip:
    """Clip geometry in local coordinates (x along the axis, towards the jaw tip), placed by (cx, cy, angle, scale)."""

    def __init__(self, cx, cy, ang=0.0, s=1.0, open_deg=0.0, held=True, jaw_len=175):
        self.cx, self.cy, self.a, self.s = cx, cy, math.radians(ang), s
        self.open, self.held, self.L = math.radians(open_deg), held, jaw_len

    def P(self, x, y):
        c, s = math.cos(self.a), math.sin(self.a)
        return (self.cx + SS * self.s * (x * c - y * s), self.cy + SS * self.s * (x * s + y * c))

    def jaw_poly(self, sign):
        th = sign * self.open
        hx, hy = 0.0, sign * 7.0

        def r(x, y):
            dx, dy = x - hx, y - hy
            return (hx + dx * math.cos(th) - dy * math.sin(th), hy + dx * math.sin(th) + dy * math.cos(th))
        pts = [(0, sign * 2), (0, sign * 17), (self.L, sign * 3.5), (self.L, sign * 1.0)]
        return [self.P(*r(x, y)) for x, y in pts], self.P(*r(self.L, sign * 2.0))

    def tip_points(self):
        _, t1 = self.jaw_poly(-1)
        _, t2 = self.jaw_poly(1)
        return t1, t2, ((t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2)

    def head_centre(self):
        return self.P(-32, 0)

    def back_end(self):
        return self.P(-66, 0)

    def draw(self, d, jaws=True, head=True, applier=None):
        applier = self.held if applier is None else applier
        if applier:
            d.polygon([self.P(-330, -20), self.P(-58, -20), self.P(-58, 20), self.P(-330, 20)], fill=APPLIER)
            d.polygon([self.P(-80, -31), self.P(-44, -31), self.P(-44, -21), self.P(-80, -21)], fill=APPLIER_DK)
            d.polygon([self.P(-80, 21), self.P(-44, 21), self.P(-44, 31), self.P(-80, 31)], fill=APPLIER_DK)
        if jaws:
            for sg in (-1, 1):
                poly, _ = self.jaw_poly(sg)
                d.polygon(poly, fill=JAW)
        if head:
            d.polygon([self.P(-66, -26), self.P(0, -26), self.P(6, -14), self.P(6, 14), self.P(0, 26), self.P(-66, 26)], fill=HEAD)
            for (x, y, rx, ry) in ((-50, -12, 7, 5), (-26, -13, 8, 5), (-36, 11, 9, 5), (-10, 8, 5, 4)):
                cx, cy = self.P(x, y)
                d.ellipse([cx - rx * SS * self.s, cy - ry * SS * self.s, cx + rx * SS * self.s, cy + ry * SS * self.s], fill=HEAD_HI)
            if self.held:   # applier prongs drawn over the back of the head
                d.polygon([self.P(-80, -31), self.P(-48, -31), self.P(-48, -21), self.P(-80, -21)], fill=APPLIER_DK)
                d.polygon([self.P(-80, 21), self.P(-48, 21), self.P(-48, 31), self.P(-80, 31)], fill=APPLIER_DK)

    def bbox(self, include_applier=False):
        pts = [self.P(-70, -30), self.P(-70, 30), self.P(8, -26), self.P(8, 26)]
        for sg in (-1, 1):
            pts += self.jaw_poly(sg)[0]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        m = 8 * SS
        return [min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m]


def dot(d, p, col, r=9, hollow=False):
    r *= SS
    if hollow:
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=4 * SS)
    else:
        d.ellipse([p[0] - r - 3 * SS, p[1] - r - 3 * SS, p[0] + r + 3 * SS, p[1] + r + 3 * SS], fill=(255, 255, 255))
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)


def label(d, p, text, at, col=INK, size=19, bold=False, line=True, align="left"):
    """Text whose left (or right) edge is at `at`; optional leader line from point p."""
    f = font(size, bold)
    if line and p is not None:
        d.line([p, at], fill=MUTED, width=2 * SS)
    tw = d.textlength(text, font=f)
    x = at[0] + 6 * SS if align == "left" else at[0] - tw - 6 * SS
    d.text((x, at[1] - size * SS * 0.65), text, font=f, fill=col)


def dashed_rect(d, b, col, dash=14, width=4):
    x0, y0, x1, y1 = b
    for (a, c) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(math.hypot(c[0] - a[0], c[1] - a[1]) / (dash * SS)))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1.0, (i + 1) / n)
            d.line([(a[0] + (c[0] - a[0]) * t0, a[1] + (c[1] - a[1]) * t0), (a[0] + (c[0] - a[0]) * t1, a[1] + (c[1] - a[1]) * t1)],
                   fill=col, width=width * SS)


def cross(d, p, r=13):
    r *= SS
    d.line([(p[0] - r, p[1] - r), (p[0] + r, p[1] + r)], fill=BAD, width=5 * SS)
    d.line([(p[0] - r, p[1] + r), (p[0] + r, p[1] - r)], fill=BAD, width=5 * SS)


def panel(img, d, k, title, notes):
    col, row = k % 3, k // 3
    x0, y0 = (24 + col * 588) * SS, (78 + row * 500) * SS
    x1, y1 = x0 + 570 * SS, y0 + 482 * SS
    d.rounded_rectangle([x0, y0, x1, y1], radius=14 * SS, fill=(255, 255, 255), outline=(215, 215, 222), width=2 * SS)
    d.text((x0 + 18 * SS, y0 + 12 * SS), title, font=font(23, True), fill=INK)
    yy = y0 + 364 * SS
    for t in notes:
        x = x0 + 18 * SS
        if isinstance(t, tuple):
            pre, c, t = t
            d.text((x, yy), pre, font=font(17, True), fill=c)
            x += d.textlength(pre, font=font(17, True)) + 6 * SS
        d.text((x, yy), t, font=font(17), fill=INK)
        yy += 27 * SS
    return x0 + 14 * SS, y0 + 52 * SS, x1 - 14 * SS, y0 + 352 * SS


class Art:
    """Drawing surface clipped to a panel's art area: draw in panel-local px, paste with paste()."""

    def __init__(self, area):
        self.area = [int(v) for v in area]
        self.img = Image.new("RGB", (W, H), TISSUE)
        self.d = ImageDraw.Draw(self.img)

    def A(self, x, y):
        return (self.area[0] + x * SS, self.area[1] + y * SS)

    def paste(self, img, blur=None):
        x0, y0, x1, y1 = self.area
        tile = self.img.crop((x0, y0, x1, y1))
        if blur:
            tile = tile.filter(ImageFilter.GaussianBlur(blur * SS))
        m = Image.new("L", tile.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], radius=10 * SS, fill=255)
        img.paste(tile, (x0, y0), m)


def main():
    img = Image.new("RGB", (W, H), (246, 246, 249))
    d = ImageDraw.Draw(img)
    d.text((24 * SS, 18 * SS), "Cómo anotar el clip en los frames reales", font=font(34, True), fill=INK)
    d.text((780 * SS, 30 * SS), "Esquema, no a escala. Orden de clics: caja (si hay que corregirla) → 1 punta → 2 extremo trasero → 3 centro del resorte",
           font=font(17), fill=MUTED)

    # 1. parts
    ar = Art(panel(img, d, 0, "1. Partes del clip",
                   ["Cabeza azul: el cuerpo del clip (titanio anodizado),", "  con el resorte dentro. Suele tenerla el colocador.",
                    "Mandíbulas: las dos hojas oscuras que cierran", "  el cuello del aneurisma. La punta es su extremo libre."]))
    a, A = ar.d, ar.A
    a.ellipse([*A(405, 85), *A(530, 245)], fill=DOME)
    c = Clip(*A(230, 165), ang=8)
    c.draw(a)
    _, _, tip = c.tip_points()
    label(a, c.P(-170, -20), "Colocador", A(20, 40))
    label(a, c.P(-30, -24), "Cabeza azul", A(215, 40))
    label(a, tip, "Punta", A(345, 70))
    label(a, c.P(95, 7), "Mandíbulas", A(250, 270))
    label(a, None, "Aneurisma", A(415, 272), line=False)
    ar.paste(img)

    # 2. what to mark
    ar = Art(panel(img, d, 1, "2. Qué marcar",
                   ["Caja amarilla: cabeza + mandíbulas, sin el colocador.", "  Si la precargada no los encierra: c y dos esquinas.",
                    "1 Punta  ·  2 Extremo final de la cabeza", "3 Centro del resorte: solo si se ve; si no, tecla x."]))
    a, A = ar.d, ar.A
    c = Clip(*A(225, 170), ang=-14)
    c.draw(a)
    dashed_rect(a, c.bbox(), YELLOW)
    _, _, tip = c.tip_points()
    hc, be = c.head_centre(), c.back_end()
    a.line([be, tip], fill=(255, 255, 255), width=3 * SS)
    dot(a, tip, RED)
    dot(a, be, BLUE)
    dot(a, hc, MAG, hollow=True)
    label(a, tip, "1 punta", A(420, 70), col=RED, bold=True)
    label(a, be, "2 extremo trasero", A(15, 268), col=BLUE, bold=True)
    label(a, hc, "3 resorte (x si no se ve)", A(255, 268), col=MAG, bold=True)
    ar.paste(img)

    # 3. open jaws
    ar = Art(panel(img, d, 2, "3. Mandíbulas abiertas  (tecla o)",
                   ["Punta = punto medio entre las dos puntas,", "  aunque caiga en el aire.",
                    "Es lo normal cuando el colocador aprieta el clip."]))
    a, A = ar.d, ar.A
    c = Clip(*A(200, 160), ang=4, open_deg=11)
    c.draw(a)
    t1, t2, tip = c.tip_points()
    a.line([t1, t2], fill=MUTED, width=2 * SS)
    dot(a, t1, MUTED, r=5)
    dot(a, t2, MUTED, r=5)
    dot(a, tip, RED)
    label(a, tip, "1 punta (punto medio)", A(300, 40), col=RED, bold=True)
    ar.paste(img)

    # 4. tip hidden behind the aneurysm
    ar = Art(panel(img, d, 3, "4. Punta oculta  (tecla p)",
                   ["Si las mandíbulas pasan detrás del aneurisma o del", "  tejido: sigue su dirección y estima dónde acaban.",
                    "Si no hay forma de estimarla: excluye con e."]))
    a, A = ar.d, ar.A
    c = Clip(*A(170, 170), ang=-6)
    c.draw(a)
    _, _, tip = c.tip_points()
    hid = c.P(c.L * 0.42, 0)
    a.ellipse([hid[0], A(0, 25)[1], hid[0] + 210 * SS, A(0, 285)[1]], fill=DOME)
    for sg in (-1, 1):
        poly, _ = c.jaw_poly(sg)
        p0, p1 = poly[1], poly[2]
        for i in range(9, 20, 2):
            q0 = (p0[0] + (p1[0] - p0[0]) * i / 20, p0[1] + (p1[1] - p0[1]) * i / 20)
            q1 = (p0[0] + (p1[0] - p0[0]) * (i + 1) / 20, p0[1] + (p1[1] - p0[1]) * (i + 1) / 20)
            a.line([q0, q1], fill=(255, 255, 255), width=3 * SS)
    dot(a, tip, RED)
    label(a, tip, "1 punta estimada", A(330, 275), col=RED, bold=True)
    ar.paste(img)

    # 5. blurred edge (blur only the clip: draw it on a transparent layer)
    area = panel(img, d, 4, "5. Borde borroso  (tecla b si todo está borroso)",
                 ["Marca a mitad de la transición clip → fondo,", "  no en el último halo ni sobre un reflejo.",
                  "Usa z para alternar zoom y frame completo."])
    ar = Art(area)
    c = Clip(*ar.A(190, 160), ang=12)
    c.draw(ar.d)
    ar.paste(img, blur=6)
    a, A = d, ar.A
    _, _, tip = c.tip_points()
    halo = c.P(c.L + 20, 0)
    dot(a, tip, RED)
    cross(a, halo, r=9)
    label(a, tip, "1 aquí", A(390, 110), col=OK, bold=True)
    label(a, halo, "no en el halo", A(400, 275), col=BAD, bold=True)

    # 6. common mistakes
    ar = Art(panel(img, d, 5, "6. Errores frecuentes",
                   [("No", BAD, "dejar la caja precargada sin revisar."), ("No", BAD, "punta sobre el aneurisma si las mandíbulas no van ahí."),
                    ("No", BAD, "extremo trasero en el fondo o fuera del clip."), ("Sí", OK, "ante la duda: estima, pulsa p y anota el n.º de frame.")]))
    a, A = ar.d, ar.A
    a.ellipse([*A(330, 25), *A(520, 185)], fill=DOME)
    c = Clip(*A(190, 195), ang=12, s=0.95)
    c.draw(a)
    dashed_rect(a, [*A(300, 40), *A(505, 120)], YELLOW)
    cross(a, A(512, 32))
    wt = A(425, 140)
    dot(a, wt, RED)
    cross(a, (wt[0] + 26 * SS, wt[1] - 22 * SS), r=10)
    wb = A(45, 70)
    dot(a, wb, BLUE)
    cross(a, (wb[0] + 26 * SS, wb[1] - 22 * SS), r=10)
    label(a, None, "caja sobre otra cosa", A(300, 20), col=BAD, size=16, line=False)
    label(a, None, "punta en el aneurisma", A(330, 170), col=BAD, size=16, line=False)
    label(a, None, "trasero en el fondo", A(30, 30), col=BAD, size=16, line=False)
    ar.paste(img)
    d = ImageDraw.Draw(img)

    # key strip
    y = 1096 * SS
    d.rounded_rectangle([24 * SS, y, 1776 * SS, y + 66 * SS], radius=12 * SS, fill=(30, 32, 40))
    keys = [("clic", "1 punta · 2 trasero · 3 resorte"), ("x", "resorte no visible"), ("c", "caja nueva"), ("o", "abierto"), ("h", "colocador"),
            ("p", "ocluido"), ("b", "borroso"), ("m", "varios clips"), ("e", "sin clip"), ("n", "guardar y seguir")]
    x = 40 * SS
    for k, t in keys:
        f1, f2 = font(18, True), font(17)
        kw = d.textlength(k, font=f1) + 18 * SS
        d.rounded_rectangle([x, y + 17 * SS, x + kw, y + 49 * SS], radius=6 * SS, fill=(250, 250, 250))
        d.text((x + 9 * SS, y + 19 * SS), k, font=f1, fill=INK)
        x += kw + 8 * SS
        d.text((x, y + 20 * SS), t, font=f2, fill=(235, 235, 240))
        x += d.textlength(t, font=f2) + 22 * SS

    out = img.resize((W // SS, H // SS), Image.LANCZOS)
    p = os.path.join(HERE, "como_anotar.png")
    out.save(p, optimize=True)
    print(p, out.size)


if __name__ == "__main__":
    main()
