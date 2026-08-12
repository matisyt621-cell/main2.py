"""
OMEGA - wspólny silnik.
Importowany zarówno przez omega_app.py (interfejs Streamlit), jak i
omega_worker.py (niezależny proces renderujący w tle). Dzięki temu cała
logika graficzna/anty-fingerprint istnieje w JEDNYM miejscu i nigdy się
nie rozjeżdża między UI a renderowaniem. Ten plik NIE importuje streamlit -
jest w pełni niezależny, co samo w sobie eliminuje wcześniejszy problem
"proces roboczy próbuje uruchomić UI Streamlit".
"""

import os
import io
import gc
import hashlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from moviepy.editor import ImageClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import moviepy.config as mpy_config

try:
    import imageio_ffmpeg
    mpy_config.change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
except:
    pass


class OmegaCore:
    VERSION = "V14.0 ANTY-TIKTOK (TRYB TŁA)"
    TARGET_RES = (1080, 1920)
    SAFE_MARGIN = 90

    @staticmethod
    def get_magick_path():
        if os.name == 'posix': return "/usr/bin/convert"
        return r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"


def get_font_path(font_selection):
    font_files = {
        "League Gothic Regular": "LeagueGothic-Regular.otf",
        "League Gothic Condensed": "LeagueGothic-CondensedRegular.otf",
        "Impact": "impact.ttf"
    }
    target = font_files.get(font_selection)
    if target and os.path.exists(target): return os.path.abspath(target)
    return "arial.ttf"


# --- CACHE ZDEKODOWANYCH ZDJĘĆ (per proces roboczy) - BEZ ZMIAN ---
_DECODE_CACHE = {}
_DECODE_CACHE_LIMIT = 300


def _load_decoded_image(path):
    cached = _DECODE_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "rb") as f:
        file_bytes = f.read()
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if len(_DECODE_CACHE) >= _DECODE_CACHE_LIMIT:
        _DECODE_CACHE.pop(next(iter(_DECODE_CACHE)))
    _DECODE_CACHE[path] = img
    return img


def process_image_916(path, target_res=OmegaCore.TARGET_RES):
    """BEZ ZMIAN logiki skalowania/kadrowania - input to ścieżka (str)."""
    try:
        img = _load_decoded_image(path)
        t_w, t_h = target_res
        img_w, img_h = img.size
        scale = t_w / img_w
        new_size = (t_w, int(img_h * scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", target_res, (0, 0, 0))
        y_off = (t_h - img_resized.height) // 2
        if y_off < 0:
            img_resized = img_resized.crop((0, abs(y_off), t_w, abs(y_off) + t_h))
            y_off = 0
        canvas.paste(img_resized, (0, y_off))
        return np.array(canvas)
    except:
        return np.zeros((target_res[1], target_res[0], 3), dtype="uint8")


def draw_text_pancerny(text, config, res=OmegaCore.TARGET_RES):
    """Silnik Auto-Scale - BEZ ŻADNYCH ZMIAN względem poprzednich wersji."""
    current_f_size = config['f_size']
    max_w = res[0] - (OmegaCore.SAFE_MARGIN * 2)

    while current_f_size > 15:
        try:
            font = ImageFont.truetype(config['font_path'], current_f_size)
        except:
            font = ImageFont.load_default()

        test_img = Image.new("RGBA", (1, 1))
        test_draw = ImageDraw.Draw(test_img)
        bbox = test_draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_w:
            break
        current_f_size -= 4

    combined = Image.new("RGBA", res, (0, 0, 0, 0))
    shd_layer = Image.new("RGBA", res, (0, 0, 0, 0))
    txt_layer = Image.new("RGBA", res, (0, 0, 0, 0))
    draw_txt = ImageDraw.Draw(txt_layer)
    draw_shd = ImageDraw.Draw(shd_layer)

    final_bbox = draw_txt.textbbox((0, 0), text, font=font)
    tw, th = final_bbox[2] - final_bbox[0], final_bbox[3] - final_bbox[1]
    pos = ((res[0] - tw) // 2, (res[1] - th) // 2)

    c_shd = config['shd_color'].lstrip('#')
    rgb_shd = tuple(int(c_shd[i:i+2], 16) for i in (0, 2, 4))
    shd_pos = (pos[0] + config['shd_x'], pos[1] + config['shd_y'])
    draw_shd.text(shd_pos, text, fill=(*rgb_shd, config['shd_alpha']), font=font)

    if config['shd_blur'] > 0:
        shd_layer = shd_layer.filter(ImageFilter.GaussianBlur(config['shd_blur']))

    draw_txt.text(pos, text, fill=config['t_color'], font=font,
                  stroke_width=config['s_width'], stroke_fill=config['s_color'])

    combined = Image.alpha_composite(combined, shd_layer)
    combined = Image.alpha_composite(combined, txt_layer)
    return combined


def apply_image_adjustments(img_array, brightness=1.0, gamma=1.0):
    """BEZ ZMIAN."""
    img = Image.fromarray(img_array)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    if gamma != 1.0:
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)


# ==============================================================================
# NOWE: KOMPRESJA DUŻYCH ZDJĘĆ (>1MB), bez zauważalnej utraty jakości.
# Uzasadnienie: process_image_916 i tak przycina każdy obraz do ~1080px
# szerokości i konwertuje do RGB (bez alfa) w _load_decoded_image powyżej -
# więc trzymanie oryginału w pełnej rozdzielczości/z kanałem alfa to czysta
# strata miejsca i czasu dekodowania, nie zysk jakości w finalnym wideo.
# ==============================================================================
_MAX_STORED_WIDTH = 1600  # bezpieczny margines nad docelową rozdzielczością (1080px)
_COMPRESS_THRESHOLD_BYTES = 1024 * 1024  # 1 MB


def _maybe_compress_image(content: bytes) -> bytes:
    if len(content) <= _COMPRESS_THRESHOLD_BYTES:
        return content
    try:
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        if img.width > _MAX_STORED_WIDTH:
            ratio = _MAX_STORED_WIDTH / img.width
            new_size = (_MAX_STORED_WIDTH, max(1, int(img.height * ratio)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        compressed = buf.getvalue()
        return compressed if len(compressed) < len(content) else content
    except Exception:
        return content  # w razie problemu - zachowaj oryginał, nie przerywaj procesu


def _save_uploads(files, subdir, base_dir="temp/uploads", compress_images=False):
    """
    Zapisuje wgrane pliki na dysk RAZ (dedup po hashu treści).
    compress_images=True: pliki obrazów >1MB są automatycznie kompresowane.
    """
    dir_path = os.path.join(base_dir, subdir)
    os.makedirs(dir_path, exist_ok=True)
    paths = []
    for f in files:
        content = f.getvalue()
        h = hashlib.md5(content).hexdigest()
        ext = os.path.splitext(f.name)[1] or ".bin"

        stored_content = content
        if compress_images:
            stored_content = _maybe_compress_image(content)
            if stored_content is not content:
                ext = ".jpg"  # skompresowane zawsze zapisywane jako JPEG

        path = os.path.join(dir_path, f"{h}{ext}")
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(stored_content)
        paths.append(path)
    return paths


def _render_single_video(job):
    """BEZ ZMIAN logiki renderowania pojedynczego filmu."""
    try:
        cfg = job['cfg']
        res_mod = tuple(job['res_mod'])

        cov_arr = process_image_916(job['cover_path'], res_mod)
        cov_arr = apply_image_adjustments(cov_arr, job['bright_mod'], job['gamma_mod'])
        clips = [ImageClip(cov_arr).set_duration(job['cov_dur'])]

        for p in job['photo_paths']:
            img_arr = process_image_916(p, res_mod)
            img_arr = apply_image_adjustments(img_arr, job['bright_mod'], job['gamma_mod'])
            clips.append(ImageClip(img_arr).set_duration(job['photo_dur']))

        base = concatenate_videoclips(clips, method="chain")

        t_arr = np.array(draw_text_pancerny(job['text'], cfg, res=res_mod))
        txt_clip = ImageClip(t_arr).set_duration(base.duration)

        final = CompositeVideoClip([base, txt_clip], size=res_mod)

        if job['music_path']:
            aud = AudioFileClip(job['music_path'])
            final = final.set_audio(aud.subclip(0, min(aud.duration, final.duration)))

        out_name = job['out_name']
        final.write_videofile(
            out_name,
            fps=job['fps_mod'],
            codec="libx264",
            audio_codec="aac",
            bitrate=None if job['v_bitrate_mod'] is None else f"{job['v_bitrate_mod']}k",
            audio_bitrate=None if job['a_bitrate_mod'] is None else f"{job['a_bitrate_mod']}k",
            threads=job['threads'],
            logger=None,
            preset="ultrafast"
        )

        final.close()
        base.close()
        gc.collect()
        return out_name
    except Exception as e:
        return f"__ERROR__:{job.get('idx')}:{e}"
