from __future__ import annotations

from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "poster"
DESKTOP = Path.home() / "Desktop"

DREAMER_IMAGE = DESKTOP / "ChatGPT Image Apr 28, 2026 at 11_33_46 AM.png"
PPO_IMAGE = DESKTOP / "ChatGPT Image Apr 28, 2026 at 11_38_24 AM.png"

POSTER_PNG = OUT_DIR / "cs1470_final_project_poster_draft.png"
POSTER_PDF = OUT_DIR / "cs1470_final_project_poster_draft.pdf"

DESKTOP_PNG = DESKTOP / "CS1470_Final_Project_Poster_Draft.png"
DESKTOP_PDF = DESKTOP / "CS1470_Final_Project_Poster_Draft.pdf"

W, H = 4800, 3600
MARGIN = 120

BG = (248, 250, 249)
CARD = (255, 255, 255)
CARD_SOFT = (252, 254, 255)
NAVY = (9, 20, 42)
TEXT = (33, 41, 54)
MUTED = (90, 101, 116)
BORDER = (204, 214, 226)
SOFT_BORDER = (225, 232, 240)
BLUE = (35, 99, 235)
TEAL = (0, 120, 118)
ORANGE = (231, 88, 42)
GREEN = (30, 122, 58)
PURPLE = (100, 64, 190)
GRAY = (110, 118, 130)
PALE_BLUE = (236, 244, 255)
PALE_TEAL = (231, 248, 247)
PALE_ORANGE = (255, 242, 235)
PALE_GREEN = (238, 248, 239)
PALE_PURPLE = (244, 240, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    base = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(base, size=size)


F_TITLE = font(104, True)
F_SUBTITLE = font(56, True)
F_AUTHORS = font(34, False)
F_SECTION = font(42, True)
F_SECTION_SMALL = font(34, True)
F_BODY = font(28, False)
F_BODY_SM = font(24, False)
F_BODY_BOLD = font(28, True)
F_BODY_SM_BOLD = font(24, True)
F_CAPTION = font(22, False)
F_METRIC = font(42, True)
F_METRIC_LABEL = font(22, True)
F_SOURCE = font(20, False)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, fnt: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> int:
    tw, th = text_size(draw, text, fnt)
    draw.text(((W - tw) // 2, y), text, font=fnt, fill=fill)
    return y + th


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.replace("\n", " \n ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if word == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, fnt)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            if text_size(draw, word, fnt)[0] <= max_width:
                current = word
            else:
                chunks = textwrap.wrap(word, width=18)
                lines.extend(chunks[:-1])
                current = chunks[-1]
    if current:
        lines.append(current)
    return lines


def draw_paragraph(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = TEXT,
    line_gap: int = 11,
) -> int:
    for line in wrap_lines(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def draw_bullets(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    x: int,
    y: int,
    width: int,
    fnt: ImageFont.FreeTypeFont,
    bullet_color: tuple[int, int, int] = TEAL,
    fill: tuple[int, int, int] = TEXT,
    line_gap: int = 9,
) -> int:
    for item in items:
        draw.ellipse((x, y + 10, x + 12, y + 22), fill=bullet_color)
        lines = wrap_lines(draw, item, fnt, width - 34)
        for i, line in enumerate(lines):
            draw.text((x + 34, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
        y += 8
    return y


def card(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, accent: tuple[int, int, int], label: str | None = None) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = xy
    shadow = (214, 223, 235)
    draw.rounded_rectangle((x0 + 10, y0 + 12, x1 + 10, y1 + 12), radius=26, fill=shadow)
    draw.rounded_rectangle(xy, radius=26, fill=CARD, outline=BORDER, width=3)
    draw.rounded_rectangle((x0, y0, x0 + 20, y1), radius=22, fill=accent)
    title_x = x0 + 48
    title_y = y0 + 34
    if label:
        pill_w = text_size(draw, label, F_BODY_SM_BOLD)[0] + 42
        draw.rounded_rectangle((title_x, title_y - 4, title_x + pill_w, title_y + 39), radius=20, fill=accent)
        draw.text((title_x + 20, title_y + 4), label, font=F_BODY_SM_BOLD, fill=(255, 255, 255))
        title_x += pill_w + 18
    draw.text((title_x, title_y), title, font=F_SECTION, fill=NAVY)
    return x0 + 48, y0 + 105, x1 - 48, y1 - 42


def small_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, accent: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=22, fill=CARD_SOFT, outline=SOFT_BORDER, width=3)
    draw.rounded_rectangle((x0, y0, x1, y0 + 12), radius=12, fill=accent)
    draw.text((x0 + 28, y0 + 28), title, font=F_SECTION_SMALL, fill=NAVY)
    return x0 + 28, y0 + 82, x1 - 28, y1 - 28


def draw_metric(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], value: str, label: str, fill: tuple[int, int, int], tint: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=22, fill=tint, outline=fill, width=3)
    vw, _ = text_size(draw, value, F_METRIC)
    draw.text((x0 + (x1 - x0 - vw) // 2, y0 + 26), value, font=F_METRIC, fill=fill)
    lines = wrap_lines(draw, label, F_METRIC_LABEL, x1 - x0 - 38)
    ly = y0 + 88
    for line in lines:
        lw, _ = text_size(draw, line, F_METRIC_LABEL)
        draw.text((x0 + (x1 - x0 - lw) // 2, ly), line, font=F_METRIC_LABEL, fill=TEXT)
        ly += 28


def dashed_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], color: tuple[int, int, int], width: int = 4, dash: int = 26, gap: int = 16) -> None:
    x0, y0, x1, y1 = xy
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=color, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=color, width=width)


def placeholder(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=22, fill=(250, 252, 253), outline=SOFT_BORDER, width=2)
    dashed_rect(draw, (x0 + 10, y0 + 10, x1 - 10, y1 - 10), accent, width=4)
    for offset in range(-400, x1 - x0 + 400, 70):
        draw.line((x0 + offset, y1 - 10, x0 + offset + 390, y0 + 10), fill=(235, 240, 246), width=2)
    tw, th = text_size(draw, title, F_BODY_BOLD)
    draw.text((x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0) // 2 - 42), title, font=F_BODY_BOLD, fill=NAVY)
    lines = wrap_lines(draw, subtitle, F_BODY_SM, x1 - x0 - 120)
    sy = y0 + (y1 - y0) // 2 + 10
    for line in lines:
        lw, _ = text_size(draw, line, F_BODY_SM)
        draw.text((x0 + (x1 - x0 - lw) // 2, sy), line, font=F_BODY_SM, fill=MUTED)
        sy += F_BODY_SM.size + 8


def fit_paste(canvas: Image.Image, img_path: Path, box: tuple[int, int, int, int], border: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    img = Image.open(img_path).convert("RGB")
    max_w, max_h = x1 - x0, y1 - y0
    ratio = min(max_w / img.width, max_h / img.height)
    new_w, new_h = int(img.width * ratio), int(img.height * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    px = x0 + (max_w - new_w) // 2
    py = y0 + (max_h - new_h) // 2
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((px - 12, py - 12, px + new_w + 12, py + new_h + 12), radius=20, fill=(255, 255, 255), outline=border, width=3)
    canvas.paste(img, (px, py))


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, W, 32), fill=TEAL)
    draw.rectangle((0, 32, W, 48), fill=ORANGE)
    draw.rounded_rectangle((MARGIN, 84, W - MARGIN, 420), radius=34, fill=(255, 255, 255), outline=SOFT_BORDER, width=2)
    y = 116
    y = draw_centered(draw, "Multi-Agent Reinforcement Learning in Pommerman", y, F_TITLE, NAVY)
    y += 8
    y = draw_centered(draw, "Comparing PPO and Dreamer-Style World Models", y, F_SUBTITLE, TEAL)
    y += 26
    authors = "Kosta Karathanosopoulos, Taari Chandaria, Wilfred Allison, Ryder Swenson"
    draw_centered(draw, authors, y, F_AUTHORS, TEXT)
    chip = "CS 1470 Final Project"
    cw, ch = text_size(draw, chip, F_BODY_SM_BOLD)
    draw.rounded_rectangle((W - MARGIN - cw - 54, 106, W - MARGIN - 24, 158), radius=24, fill=PALE_ORANGE, outline=ORANGE, width=2)
    draw.text((W - MARGIN - cw - 28, 119), chip, font=F_BODY_SM_BOLD, fill=ORANGE)


def build_poster() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw)

    left_x, left_w = 120, 1250
    center_x, center_w = 1440, 1720
    right_x, right_w = 3230, 1450

    # Left column
    ix0, iy, ix1, _ = card(draw, (left_x, 500, left_x + left_w, 1025), "Motivation", BLUE, "01")
    iy = draw_paragraph(
        draw,
        "Real-world decision-making often involves multiple agents whose actions change the environment and each other's learning problem. This makes reinforcement learning harder than standard single-agent settings, where dynamics are more stable.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY,
    )
    iy += 18
    draw_paragraph(
        draw,
        "We test whether Dreamer-style world models can still help when other agents become part of the dynamics.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_BOLD,
        TEAL,
    )

    ix0, iy, ix1, iy1 = card(draw, (left_x, 1070, left_x + left_w, 2025), "Pommerman Environment", TEAL, "02")
    draw_paragraph(
        draw,
        "Pommerman is a four-player grid-world inspired by Bomberman. Agents navigate an 11 x 11 maze, place bombs, collect power-ups, and try to survive while eliminating opponents.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM,
    )
    placeholder(
        draw,
        (ix0, iy1 - 410, ix1, iy1),
        "GAME ENVIRONMENT IMAGE",
        "Add a rollout frame, board screenshot, or qualitative gameplay panel here.",
        TEAL,
    )

    ix0, iy, ix1, _ = card(draw, (left_x, 2070, left_x + left_w, 2985), "Training Challenges", ORANGE, "03")
    iy = draw_paragraph(
        draw,
        "Dreamer-style training was fragile because the agent first has to learn a usable model of bomb dynamics, board changes, rewards, and opponent behavior.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM,
    )
    iy += 26
    draw_bullets(
        draw,
        [
            "Added sequence-aware replay and valid same-episode trajectory checks.",
            "Compared independent, shared, and opponent-aware world models.",
            "Used shaped rewards to discourage passive behavior and reward useful bombs, wood destruction, power-ups, and eliminations.",
        ],
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM,
        ORANGE,
    )

    # Center column
    ix0, iy, ix1, iy1 = card(draw, (center_x, 500, center_x + center_w, 3145), "Model Architectures", PURPLE, "04")
    draw_paragraph(
        draw,
        "We compare a model-free PPO actor-critic baseline against Dreamer-style agents that learn a recurrent latent world model and train policies through imagined rollouts.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM,
        MUTED,
    )
    dreamer_box = (ix0, iy + 95, ix1, iy + 95 + 1030)
    fit_paste(canvas, DREAMER_IMAGE, dreamer_box, TEAL)
    draw.text((ix0, dreamer_box[3] + 26), "Dreamer: learned latent dynamics + imagined policy/value learning", font=F_CAPTION, fill=TEAL)
    ppo_box = (ix0, dreamer_box[3] + 80, ix1, dreamer_box[3] + 80 + 1030)
    fit_paste(canvas, PPO_IMAGE, ppo_box, PURPLE)
    draw.text((ix0, ppo_box[3] + 26), "PPO: model-free actor-critic updates from real rollouts only", font=F_CAPTION, fill=PURPLE)

    # Right column
    ix0, iy, ix1, iy1 = card(draw, (right_x, 500, right_x + right_w, 1590), "Results", GREEN, "05")
    metric_gap = 24
    metric_w = (ix1 - ix0 - 2 * metric_gap) // 3
    draw_metric(draw, (ix0, iy, ix0 + metric_w, iy + 160), "+1.14", "PPO eval reward", GREEN, PALE_GREEN)
    draw_metric(draw, (ix0 + metric_w + metric_gap, iy, ix0 + 2 * metric_w + metric_gap, iy + 160), "50%", "PPO win rate", BLUE, PALE_BLUE)
    draw_metric(draw, (ix0 + 2 * (metric_w + metric_gap), iy, ix1, iy + 160), "0%", "Dreamer final wins", ORANGE, PALE_ORANGE)
    y = iy + 200
    draw_paragraph(
        draw,
        "In the main 2048-step study, PPO was the most stable short-budget baseline. Dreamer variants showed signs of world-model learning, but did not consistently convert imagined predictions into successful control.",
        ix0,
        y,
        ix1 - ix0,
        F_BODY_SM,
    )
    placeholder(
        draw,
        (ix0, iy1 - 455, ix1, iy1),
        "RESULTS GRAPH PLACEHOLDER",
        "Add reward curves, win-rate curves, or a final summary table here.",
        GREEN,
    )

    ix0, iy, ix1, _ = card(draw, (right_x, 1635, right_x + right_w, 2440), "Significance", TEAL, "06")
    iy = draw_paragraph(
        draw,
        "The results suggest that model-free methods can be more stable under limited compute, while multi-agent world modeling remains difficult because the model must predict both game physics and other agents' behavior.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM,
    )
    iy += 28
    draw_paragraph(
        draw,
        "The negative result is still informative: it helps identify where Dreamer-style imagination breaks down first in a non-stationary multiplayer setting.",
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM_BOLD,
        TEAL,
    )

    ix0, iy, ix1, _ = card(draw, (right_x, 2485, right_x + right_w, 3145), "Future Work", BLUE, "07")
    draw_bullets(
        draw,
        [
            "Longer training runs and more random seeds.",
            "Stronger opponent modeling and richer diagnostics for imagined rollouts.",
            "Compare sparse vs. shaped rewards more systematically.",
            "Extend to team settings and smaller ablations for stability.",
        ],
        ix0,
        iy,
        ix1 - ix0,
        F_BODY_SM,
        BLUE,
    )

    # Footer sources
    fy0, fy1 = 3205, 3485
    draw.rounded_rectangle((MARGIN, fy0, W - MARGIN, fy1), radius=24, fill=(255, 255, 255), outline=SOFT_BORDER, width=2)
    draw.text((MARGIN + 34, fy0 + 28), "Sources", font=F_SECTION_SMALL, fill=NAVY)
    source_text = (
        "Resnick, C., Eldridge, W., Ha, D., Britz, D., Foerster, J. N., Togelius, J., Cho, K., & Bruna, J. (2018). "
        "Pommerman: A Multi-Agent Playground. AIIDE Workshops. arXiv:1809.07124.  "
        "Hafner, D., Lillicrap, T. P., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. "
        "ICLR. arXiv:1912.01603."
    )
    draw_paragraph(draw, source_text, MARGIN + 34, fy0 + 92, W - 2 * MARGIN - 68, F_SOURCE, MUTED, line_gap=7)

    # Thin crop/format note for future edits.
    note = "4:3 horizontal poster draft with blank placeholders for gameplay and results figures"
    nw, _ = text_size(draw, note, F_SOURCE)
    draw.text((W - MARGIN - nw, H - 54), note, font=F_SOURCE, fill=GRAY)
    return canvas


def main() -> None:
    for path in (DREAMER_IMAGE, PPO_IMAGE):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    poster = build_poster()
    poster.save(POSTER_PNG, quality=95)
    poster.save(POSTER_PDF, "PDF", resolution=100.0)
    shutil.copy2(POSTER_PNG, DESKTOP_PNG)
    shutil.copy2(POSTER_PDF, DESKTOP_PDF)
    print(f"Wrote {POSTER_PNG}")
    print(f"Wrote {POSTER_PDF}")
    print(f"Copied {DESKTOP_PNG}")
    print(f"Copied {DESKTOP_PDF}")


if __name__ == "__main__":
    main()
