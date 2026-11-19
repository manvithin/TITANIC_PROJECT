"""Generate the custom Titanic ship Lottie animation used on the website.

Run from the project root:

    python scripts/generate_lottie.py

The animation is intentionally code-drawn (no third-party artwork): a starry
sky, a moon, three ocean layers, a bobbing Titanic with four funnels, drifting
smoke and a small shooting star.  Output: static/lottie/titanic.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "lottie" / "titanic.json"

W, H = 1000, 560
FPS = 30
DURATION = 180  # 6 seconds, seamless loop


def color(hex_value: str, alpha: float = 1.0) -> list[float]:
    value = hex_value.lstrip("#")
    return [
        int(value[0:2], 16) / 255,
        int(value[2:4], 16) / 255,
        int(value[4:6], 16) / 255,
        alpha,
    ]


def ease(t: int, s: list, ease_in=(0.4, 1.0), ease_out=(0.6, 0.0)) -> dict:
    return {
        "i": {"x": ease_in[0], "y": ease_in[1]},
        "o": {"x": ease_out[0], "y": ease_out[1]},
        "t": t,
        "s": s,
    }


def linear(t: int, s: list) -> dict:
    return {
        "i": {"x": 1, "y": 1},
        "o": {"x": 0, "y": 0},
        "t": t,
        "s": s,
    }


def static_prop(value):
    return {"a": 0, "k": value}


def animated_prop(frames: list[dict]):
    return {"a": 1, "k": frames}


def transform(**kwargs) -> dict:
    return {
        "ty": "tr",
        "p": static_prop(list(kwargs.get("p", (0, 0)))),
        "a": static_prop(list(kwargs.get("a", (0, 0)))),
        "s": static_prop(list(kwargs.get("s", (100, 100)))),
        "r": static_prop(kwargs.get("r", 0)),
        "o": static_prop(kwargs.get("o", 100)),
        "sk": static_prop(0),
        "sa": static_prop(0),
        "nm": "Transform",
    }


def fill(hex_value: str, opacity: float = 100) -> dict:
    return {
        "ty": "fl",
        "c": static_prop(color(hex_value)),
        "o": static_prop(opacity),
        "r": 1,
        "nm": "Fill",
    }


def rect(x: float, y: float, w: float, h: float, fill_color: str, radius=0, opacity=100) -> dict:
    return {
        "ty": "gr",
        "it": [
            {
                "ty": "rc",
                "d": 1,
                "s": static_prop([w, h]),
                "p": static_prop([x, y]),
                "r": static_prop(radius),
                "nm": "Rectangle",
            },
            fill(fill_color, opacity),
            transform(),
        ],
        "nm": "Rectangle group",
    }


def ellipse(x: float, y: float, w: float, h: float, fill_color: str, opacity=100) -> dict:
    return {
        "ty": "gr",
        "it": [
            {
                "ty": "el",
                "d": 1,
                "s": static_prop([w, h]),
                "p": static_prop([x, y]),
                "nm": "Ellipse",
            },
            fill(fill_color, opacity),
            transform(),
        ],
        "nm": "Ellipse group",
    }


def path(points: list[tuple[float, float]], fill_color: str, opacity=100) -> dict:
    vertices = [[round(x, 2), round(y, 2)] for x, y in points]
    return {
        "ty": "gr",
        "it": [
            {
                "ty": "sh",
                "ks": static_prop(
                    {
                        "i": [[0, 0] for _ in vertices],
                        "o": [[0, 0] for _ in vertices],
                        "v": vertices,
                        "c": True,
                    }
                ),
                "nm": "Path",
            },
            fill(fill_color, opacity),
            transform(),
        ],
        "nm": "Path group",
    }


def pivot_group(pole, points, fill_color, rotation_frames, opacity=100) -> dict:
    """A filled shape that rotates around an off-centre pivot (a waving flag)."""
    vertices = [[round(x - pole[0], 2), round(y - pole[1], 2)] for x, y in points]
    return {
        "ty": "gr",
        "it": [
            {
                "ty": "sh",
                "ks": static_prop(
                    {
                        "i": [[0, 0] for _ in vertices],
                        "o": [[0, 0] for _ in vertices],
                        "v": vertices,
                        "c": True,
                    }
                ),
                "nm": "Flag path",
            },
            fill(fill_color, opacity),
            {
                "ty": "tr",
                "p": static_prop([pole[0], pole[1], 0]),
                "a": static_prop([0, 0, 0]),
                "s": static_prop([100, 100, 100]),
                "r": animated_prop(rotation_frames),
                "o": static_prop(100),
                "sk": static_prop(0),
                "sa": static_prop(0),
                "nm": "Transform",
            },
        ],
        "nm": "Flag group",
    }


def layer(
    index: int,
    name: str,
    shapes: list[dict],
    position=(0, 0),
    position_frames: list[dict] | None = None,
    rotation=0,
    rotation_frames: list[dict] | None = None,
    scale=(100, 100),
    scale_frames: list[dict] | None = None,
    opacity=100,
    opacity_frames: list[dict] | None = None,
    anchor=(0, 0),
) -> dict:
    ks = {
        "o": animated_prop(opacity_frames) if opacity_frames else static_prop(opacity),
        "r": animated_prop(rotation_frames) if rotation_frames else static_prop(rotation),
        "p": animated_prop(position_frames)
        if position_frames
        else static_prop([position[0], position[1], 0]),
        "a": static_prop([anchor[0], anchor[1], 0]),
        "s": animated_prop(scale_frames)
        if scale_frames
        else static_prop([scale[0], scale[1], 100]),
    }
    return {
        "ddd": 0,
        "ind": index,
        "ty": 4,
        "nm": name,
        "sr": 1,
        "ks": ks,
        "ao": 0,
        "shapes": shapes,
        "ip": 0,
        "op": DURATION,
        "st": 0,
        "bm": 0,
    }


# ---------------------------------------------------------------------------
# Ship (one shape layer so the whole vessel bobs together)
# ---------------------------------------------------------------------------

FUNNELS = (400, 455, 510, 565)

# Shape order matters: Lottie paints the first shape in this list last, so the
# first entry is the topmost. The flag and railing sit above the funnels, the
# funnels above the decks (identical caps, bands and bodies for all four), and
# the hull is the bottom layer.
ship_shapes = [
    # Fluttering White Star pennant, pivoting on the aft masthead
    pivot_group(
        (678, 244),
        [(678, 235), (706, 244), (678, 253)],
        "#e76f51",
        [
            linear(frame, [round(4.8 * math.sin(2 * math.pi * frame / 40), 2)])
            for frame in range(0, DURATION + 1, 4)
        ],
    ),
    # Deck railing line
    rect(509, 344, 380, 2, "#f1e7d3", radius=1, opacity=55),
    # Portholes along the hull
    *[
        ellipse(x, 385, 7, 7, "#f4c27a", opacity=85)
        for x in range(370, 660, 28)
    ],
    # Lifeboats on the boat deck
    *[
        ellipse(x, 333, 26, 8, "#fbfdfe", opacity=90)
        for x in (392, 442, 548, 598)
    ],
    # Masts and crossbars
    rect(350, 292, 3, 84, "#e6dfcf", radius=1, opacity=85),
    rect(676, 284, 3, 98, "#e6dfcf", radius=1, opacity=85),
    rect(350, 300, 34, 2.5, "#e6dfcf", radius=1, opacity=70),
    rect(676, 292, 28, 2.5, "#e6dfcf", radius=1, opacity=70),
    # Four identical funnel outlets: dark cap, gold band, then body.
    # Caps are drawn above the bodies so the opening reads uniformly.
    *[rect(x, 229, 32, 12, "#22384b", radius=3) for x in FUNNELS],
    *[rect(x, 244, 28, 5, "#f4a261", radius=1) for x in FUNNELS],
    *[rect(x, 263, 28, 74, "#c96a45", radius=2) for x in FUNNELS],
    # Promenade windows (upper deck first)
    *[
        rect(x, 297, 6, 6, "#f4c27a", radius=1, opacity=75)
        for x in range(430, 590, 24)
    ],
    *[
        rect(x, 317, 8, 9, "#f4c27a", radius=1, opacity=90)
        for x in range(392, 628, 26)
    ],
    # Deck and superstructure, top deck first
    rect(509, 281, 72, 12, "#fbfdfe", radius=2),
    rect(509, 294, 190, 18, "#f2f7fa", radius=3),
    rect(509, 315, 258, 34, "#e7eef3", radius=3),
    rect(509, 337, 406, 13, "#f1e7d3", radius=4),
    # Red waterline band
    path(
        [
            (330, 417), (702, 417), (690, 470), (352, 470), (326, 417),
        ],
        "#a74e39",
    ),
    # Hull (extends under the foreground wave)
    path(
        [
            (310, 344), (708, 344), (724, 362), (706, 398),
            (690, 470), (352, 470), (326, 406), (304, 370),
        ],
        "#15304a",
    ),
]

SHIP_X, SHIP_Y = 679, 272
ship_frames = [
    ease(0, [SHIP_X, SHIP_Y, 0]),
    ease(45, [SHIP_X, SHIP_Y - 7, 0]),
    ease(90, [SHIP_X, SHIP_Y, 0]),
    ease(135, [SHIP_X, SHIP_Y + 7, 0]),
    ease(180, [SHIP_X, SHIP_Y, 0]),
]
ship_rotation = [
    ease(0, [0]),
    ease(45, [-0.7]),
    ease(90, [0]),
    ease(135, [0.7]),
    ease(180, [0]),
]

layers: list[dict] = []
index = 1


# ---------------------------------------------------------------------------
# Foreground ocean (drawn above the ship so the hull sits in the water)
# ---------------------------------------------------------------------------

def ocean_path(base_y: float, amplitude: float, wavelength: float, phase: float) -> list[tuple[float, float]]:
    points = []
    x = -1400
    while x <= 2400:
        y = base_y + amplitude * math.sin((2 * math.pi * x / wavelength) + phase)
        points.append((x, y))
        x += 18
    points.extend([(2400, H), (-1400, H)])
    return points


def wave_layer(ind: int, base_y: float, amplitude: float, wavelength: float, phase: float, color_hex: str, opacity=100) -> dict:
    frames = [
        linear(0, [0, 0, 0]),
        linear(DURATION, [-wavelength, 0, 0]),
    ]
    return layer(
        ind,
        f"Ocean {wavelength}",
        [path(ocean_path(base_y, amplitude, wavelength, phase), color_hex, opacity)],
        position_frames=frames,
    )


layers.append(
    wave_layer(index, 387, 13, 420, 1.4, "#08283a")
)
index += 1

# Foam highlights riding the foreground wave
foam = []
for x in range(30, 1000, 95):
    y = 387 + 13 * math.sin((2 * math.pi * x / 420) + 1.4)
    foam.append(ellipse(x, y - 2, 34, 4, "#d8f3f0", opacity=28))
layers.append(
    layer(
        index,
        "Foam",
        foam,
        position_frames=[linear(0, [0, 0, 0]), linear(DURATION, [-420, 0, 0])],
    )
)
index += 1

# ---------------------------------------------------------------------------
# Ship
# ---------------------------------------------------------------------------

layers.append(
    layer(
        index,
        "Titanic",
        ship_shapes,
        position_frames=ship_frames,
        rotation_frames=ship_rotation,
        anchor=(509, 331),
    )
)
index += 1

# ---------------------------------------------------------------------------
# Middle and back ocean layers (behind the ship)
# ---------------------------------------------------------------------------

layers.append(wave_layer(index, 369, 10, 520, 2.6, "#0d4356", opacity=95))
index += 1
layers.append(wave_layer(index, 359, 8, 700, 0.4, "#115066", opacity=90))
index += 1

# ---------------------------------------------------------------------------
# Sky: shooting star, moon, stars, clouds
# ---------------------------------------------------------------------------

layers.append(
    layer(
        index,
        "Shooting star",
        [
            ellipse(0, 0, 74, 2.4, "#fff2cf", opacity=60),
            ellipse(36, 0, 9, 9, "#fff8e7", opacity=100),
        ],
        position_frames=[
            ease(0, [455, 88, 0]),
            {"t": 118, "s": [455, 88, 0]},
            ease(152, [625, 172, 0]),
            ease(DURATION, [625, 172, 0]),
        ],
        rotation=-28,
        opacity_frames=[
            {"t": 0, "s": [0]},
            {"t": 118, "s": [0]},
            {"t": 130, "s": [80]},
            {"t": 150, "s": [30]},
            {"t": 158, "s": [0]},
            {"t": DURATION, "s": [0]},
        ],
    )
)
index += 1

layers.append(
    layer(
        index,
        "Moon",
        [
            ellipse(790, 130, 96, 96, "#fff9e8", opacity=100),
            ellipse(776, 118, 14, 14, "#e7d3a5", opacity=38),
            ellipse(804, 144, 10, 10, "#e7d3a5", opacity=34),
            ellipse(772, 148, 7, 7, "#e7d3a5", opacity=30),
        ],
        opacity_frames=[
            {"t": 0, "s": [88]},
            {"t": 90, "s": [100]},
            {"t": 180, "s": [88]},
        ],
    )
)
index += 1

star_positions = [
    (90, 140, 1.8, 0), (160, 195, 1.4, 35), (240, 128, 2.0, 70),
    (320, 205, 1.3, 15), (390, 160, 1.7, 55), (470, 132, 1.2, 90),
    (545, 128, 1.9, 25), (620, 145, 1.5, 65), (830, 150, 1.3, 10),
    (870, 140, 1.7, 45), (930, 175, 1.4, 75), (935, 148, 1.9, 30),
    (60, 212, 1.2, 50), (280, 218, 1.5, 85), (720, 120, 1.3, 20),
    (118, 126, 1.4, 12), (500, 128, 1.3, 60), (700, 205, 1.2, 5),
]
for x, y, r, phase in star_positions:
    peak = min(phase + 52, DURATION)
    low = min(phase + 96, DURATION)
    frames = [{"t": 0, "s": [75]}]
    if phase > 0:
        frames.append({"t": phase, "s": [75]})
    frames.extend(
        [
            {"t": peak, "s": [100]},
            {"t": low, "s": [62]},
            {"t": DURATION, "s": [75]},
        ]
    )
    layers.append(
        layer(
            index,
            f"Star {index}",
            [
                ellipse(x, y, r * 2.6, r * 2.6, "#ffffff", opacity=100),
            ],
            opacity_frames=frames,
        )
    )
    index += 1

animation = {
    "v": "5.12.2",
    "fr": FPS,
    "ip": 0,
    "op": DURATION,
    "w": W,
    "h": H,
    "nm": "Titanic Voyage",
    "ddd": 0,
    "assets": [],
    "layers": layers,
    "markers": [],
}


def fill_end_values(node) -> None:
    """Normalise keyframes for the Lottie SVG player.

    Older Lottie versions read a keyframe's ``e`` end value, and every keyframe
    must carry easing keys (``i``/``o``) or the player silently hides the layer.
    """
    if isinstance(node, dict):
        if node.get("a") == 1 and isinstance(node.get("k"), list):
            frames = node["k"]
            for i, frame in enumerate(frames):
                if not isinstance(frame, dict) or "s" not in frame:
                    continue
                frame.setdefault("i", {"x": 0.6, "y": 1.0})
                frame.setdefault("o", {"x": 0.4, "y": 0.0})
                if "e" not in frame:
                    nxt = frames[i + 1]["s"] if i + 1 < len(frames) else frame["s"]
                    frame["e"] = nxt
        for value in node.values():
            fill_end_values(value)
    elif isinstance(node, list):
        for item in node:
            fill_end_values(item)


fill_end_values(animation)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(animation, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(layers)} layers)")
