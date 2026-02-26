import streamlit as st
import os
import gc
import random
import time
import datetime
import io
import zipfile
import json
import tempfile
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageChops
from moviepy.editor import (
    ImageClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip,
    VideoClip, TextClip, CompositeAudioClip, vfx
)
import moviepy.config as mpy_config

# ==============================================================================
# 1. KONFIGURACJA PODSTAWOWA I LOGOWANIE
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmegaCore:
    VERSION = "V13.3 PRO (ADVANCED ANTI-FINGERPRINT)"
    TARGET_RES = (1080, 1920)
    SAFE_MARGIN = 90
    SUPPORTED_IMG_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    SUPPORTED_AUDIO_FORMATS = ('.mp3', '.wav', '.m4a', '.aac', '.ogg')
    
    @staticmethod
    def setup_session():
        defaults = {
            'v_covers': [],
            'v_photos': [],
            'v_music': [],
            'v_results': [],
            'zip_files': [],
            'config_profiles': {},
            'current_profile': 'default',
            'processed_files': [],
            'temp_dir': None,
            'render_log': []
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get_magick_path() -> str:
        if os.name == 'posix':
            common_paths = ["/usr/bin/convert", "/usr/local/bin/convert"]
            for path in common_paths:
                if os.path.exists(path):
                    return path
            return "/usr/bin/convert"
        else:
            win_paths = [
                r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
                r"C:\Program Files\ImageMagick-7.0.11-Q16\magick.exe"
            ]
            for path in win_paths:
                if os.path.exists(path):
                    return path
            return r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

    @staticmethod
    def cleanup_temp():
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            st.session_state.temp_dir = None
            logger.info("Katalog tymczasowy usunięty")

# ==============================================================================
# 2. MODELE DANYCH
# ==============================================================================

class TextPosition(Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    CUSTOM = "custom"

class TransitionType(Enum):
    NONE = "none"
    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"

class NoiseType(Enum):
    NONE = "none"
    GAUSSIAN = "gaussian"
    SALT_PEPPER = "salt_pepper"
    POISSON = "poisson"

class ColorShiftMode(Enum):
    NONE = "none"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    RGB_SHIFT = "rgb_shift"
    GAMMA = "gamma"           # Nowy: korekcja gamma
    WHITE_BALANCE = "white_balance"  # Nowy: balans bieli (temperatura)

class PhotoDurationMode(Enum):
    FIXED = "fixed"
    RANGE = "range"
    LIST = "list"

class CodecProfile(Enum):
    AUTO = "auto"
    BASELINE = "baseline"
    MAIN = "main"
    HIGH = "high"

@dataclass
class TextStyleConfig:
    font_name: str = "League Gothic Regular"
    font_size: int = 83
    font_color: str = "#FFFFFF"
    stroke_width: int = 3
    stroke_color: str = "#000000"
    shadow_offset_x: int = 15
    shadow_offset_y: int = 15
    shadow_blur: int = 8
    shadow_alpha: int = 200
    shadow_color: str = "#000000"
    position: TextPosition = TextPosition.CENTER
    custom_position: Tuple[int, int] = (540, 960)
    multiline_align: str = "center"
    line_spacing: int = 10
    uppercase: bool = True
    use_outline: bool = True
    use_shadow: bool = True
    animation_in: str = "none"
    animation_out: str = "none"
    animation_duration: float = 0.5
    # Anti-fingerprint tekstu
    random_position_jitter: bool = False
    position_jitter_range: int = 10
    random_shadow_jitter: bool = False
    shadow_jitter_range: int = 5

@dataclass
class VideoConfig:
    resolution: Tuple[int, int] = (1080, 1920)
    fps: int = 24
    codec: str = "libx264"
    audio_codec: str = "aac"
    preset: str = "ultrafast"
    bitrate: str = "4000k"
    target_duration_min: float = 8.0
    target_duration_max: float = 10.0
    # Czy używać okładek?
    use_cover: bool = True
    cover_duration_multiplier: float = 3.0
    # Tryb czasu trwania zdjęcia
    photo_duration_mode: PhotoDurationMode = PhotoDurationMode.FIXED
    photo_duration_fixed: float = 0.15
    photo_duration_min: float = 0.1
    photo_duration_max: float = 0.2
    photo_duration_list: str = "0.1, 0.11, 0.15, 0.2"
    transition: TransitionType = TransitionType.NONE
    transition_duration: float = 0.2
    randomize_photo_order: bool = True
    repeat_photos_if_needed: bool = True
    audio_fade_in: float = 0.5
    audio_fade_out: float = 0.5
    audio_volume: float = 1.0
    add_timestamp: bool = False
    output_dir: str = "output"
    
    # ========== ANTY-FINGERPRINT ==========
    # Obraz
    enable_random_noise: bool = False
    noise_type: NoiseType = NoiseType.GAUSSIAN
    noise_intensity: float = 0.005  # 0.5% (subtelny szum)
    
    enable_random_color_shift: bool = False
    color_shift_mode: ColorShiftMode = ColorShiftMode.BRIGHTNESS
    color_shift_range: float = 0.02  # ±2% (subtelna korekta)
    
    # Gamma (oddzielna, bo to nieliniowa korekcja)
    enable_gamma_correction: bool = False
    gamma_range: Tuple[float, float] = (0.98, 1.02)  # zmiana gammy o ±2%
    
    # Balans bieli (temperatura)
    enable_white_balance: bool = False
    white_balance_range: Tuple[float, float] = (-0.02, 0.02)  # przesunięcie kanałów R/B
    
    enable_random_zoom: bool = False
    zoom_range: Tuple[float, float] = (0.98, 1.02)
    zoom_crop: bool = True
    
    enable_random_rotation: bool = False
    rotation_range: Tuple[float, float] = (-0.5, 0.5)
    
    enable_random_flip: bool = False
    flip_probability: float = 0.1
    
    # Czas i klatkaż
    enable_random_fps: bool = False
    fps_variation: float = 0.1  # odchylenie od wybranego FPS (np. 30 -> 29.97-30.03)
    
    enable_random_speed_change: bool = False
    speed_change_range: Tuple[float, float] = (0.99, 1.01)  # ±1%
    
    # Zaawansowane: usuwanie co N-tej klatki (decimate)
    enable_decimate: bool = False
    decimate_every: int = 100  # co 100 klatek
    
    # Audio
    enable_random_pitch_shift: bool = False
    pitch_shift_range: Tuple[float, float] = (0.98, 1.02)
    
    # Kontener i metadane
    codec_profile: CodecProfile = CodecProfile.AUTO
    vary_bitrate: bool = False
    bitrate_variation: float = 0.05  # ±5% odchylenia bitrate
    strip_metadata: bool = False  # czyścić metadane po renderze

@dataclass
class RenderJob:
    cover_file: Any
    photo_files: List[Any]
    audio_file: Optional[Any]
    text_lines: List[str]
    text_style: TextStyleConfig
    video_config: VideoConfig
    job_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

# ==============================================================================
# 3. SILNIK GRAFICZNY – rozszerzony o gamma i balans bieli
# ==============================================================================

class ImageProcessor:
    @staticmethod
    def process_image(
        file_obj,
        target_res: Tuple[int, int] = OmegaCore.TARGET_RES,
        mode: str = "cover",
        background_color: Tuple[int, int, int] = (0, 0, 0),
        apply_blur: bool = False,
        blur_radius: float = 0,
        enhance_contrast: float = 1.0,
        enhance_sharpness: float = 1.0,
        grayscale: bool = False,
        noise_params: Optional[Dict] = None,
        color_shift_params: Optional[Dict] = None,
        gamma_params: Optional[Dict] = None,
        white_balance_params: Optional[Dict] = None,
        zoom_params: Optional[Dict] = None,
        rotation_params: Optional[Dict] = None,
        flip_params: Optional[Dict] = None
    ) -> np.ndarray:
        try:
            file_bytes = file_obj.getvalue()
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img).convert("RGBA")
            
            if grayscale:
                img = img.convert("L").convert("RGBA")
            
            if enhance_contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(enhance_contrast)
            
            if enhance_sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(enhance_sharpness)
            
            # Skalowanie
            if mode == "cover":
                img = ImageProcessor._scale_cover(img, target_res)
            elif mode == "contain":
                img = ImageProcessor._scale_contain(img, target_res, background_color)
            elif mode == "crop":
                img = ImageProcessor._scale_crop(img, target_res)
            
            if apply_blur and blur_radius > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Anti-fingerprint modyfikacje
            if noise_params and noise_params.get('enabled'):
                img = ImageProcessor._apply_noise(img, noise_params)
            if color_shift_params and color_shift_params.get('enabled'):
                img = ImageProcessor._apply_color_shift(img, color_shift_params)
            if gamma_params and gamma_params.get('enabled'):
                img = ImageProcessor._apply_gamma(img, gamma_params)
            if white_balance_params and white_balance_params.get('enabled'):
                img = ImageProcessor._apply_white_balance(img, white_balance_params)
            if zoom_params and zoom_params.get('enabled'):
                img = ImageProcessor._apply_zoom(img, zoom_params, target_res)
            if rotation_params and rotation_params.get('enabled'):
                img = ImageProcessor._apply_rotation(img, rotation_params, target_res, background_color)
            if flip_params and flip_params.get('enabled') and random.random() < flip_params.get('probability', 0):
                img = ImageProcessor._apply_flip(img, flip_params)
            
            return np.array(img.convert("RGB"))
        except Exception as e:
            logger.error(f"Błąd przetwarzania obrazu: {e}")
            return np.zeros((target_res[1], target_res[0], 3), dtype=np.uint8)
    
    @staticmethod
    def _apply_noise(img: Image.Image, params: Dict) -> Image.Image:
        noise_type = params.get('type', NoiseType.GAUSSIAN)
        intensity = params.get('intensity', 0.005)
        np_img = np.array(img).astype(np.float32)
        if noise_type == NoiseType.GAUSSIAN:
            noise = np.random.normal(0, intensity * 255, np_img.shape).astype(np.float32)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        elif noise_type == NoiseType.SALT_PEPPER:
            salt_vs_pepper = 0.5
            num_salt = int(intensity * np_img.size * salt_vs_pepper)
            num_pepper = int(intensity * np_img.size * (1 - salt_vs_pepper))
            coords = [np.random.randint(0, i-1, num_salt) for i in np_img.shape]
            np_img[coords[0], coords[1], :] = 255
            coords = [np.random.randint(0, i-1, num_pepper) for i in np_img.shape]
            np_img[coords[0], coords[1], :] = 0
        elif noise_type == NoiseType.POISSON:
            vals = len(np.unique(np_img))
            vals = 2 ** np.ceil(np.log2(vals))
            np_img = np.random.poisson(np_img * vals) / float(vals)
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    
    @staticmethod
    def _apply_color_shift(img: Image.Image, params: Dict) -> Image.Image:
        mode = params.get('mode', ColorShiftMode.BRIGHTNESS)
        amount = params.get('amount', 0.02)
        if mode == ColorShiftMode.BRIGHTNESS:
            enhancer = ImageEnhance.Brightness(img)
            factor = 1.0 + random.uniform(-amount, amount)
            img = enhancer.enhance(factor)
        elif mode == ColorShiftMode.CONTRAST:
            enhancer = ImageEnhance.Contrast(img)
            factor = 1.0 + random.uniform(-amount, amount)
            img = enhancer.enhance(factor)
        elif mode == ColorShiftMode.SATURATION:
            enhancer = ImageEnhance.Color(img)
            factor = 1.0 + random.uniform(-amount, amount)
            img = enhancer.enhance(factor)
        elif mode == ColorShiftMode.HUE:
            hsv = img.convert('HSV')
            h, s, v = hsv.split()
            h_data = np.array(h, dtype=np.uint8)
            shift = int(amount * 255)
            h_data = (h_data + shift) % 256
            h = Image.fromarray(h_data, mode='L')
            img = Image.merge('HSV', (h, s, v)).convert('RGBA')
        elif mode == ColorShiftMode.RGB_SHIFT:
            r, g, b, a = img.split()
            r_data = np.array(r, dtype=np.float32)
            g_data = np.array(g, dtype=np.float32)
            b_data = np.array(b, dtype=np.float32)
            shift_r = random.uniform(-amount*50, amount*50)
            shift_g = random.uniform(-amount*50, amount*50)
            shift_b = random.uniform(-amount*50, amount*50)
            r_data = np.clip(r_data + shift_r, 0, 255).astype(np.uint8)
            g_data = np.clip(g_data + shift_g, 0, 255).astype(np.uint8)
            b_data = np.clip(b_data + shift_b, 0, 255).astype(np.uint8)
            img = Image.merge('RGBA', (Image.fromarray(r_data), Image.fromarray(g_data), Image.fromarray(b_data), a))
        # Gamma i white balance są oddzielne
        return img
    
    @staticmethod
    def _apply_gamma(img: Image.Image, params: Dict) -> Image.Image:
        """Korekcja gamma: wartość gamma (1.0 = brak zmiany)."""
        gamma = params.get('gamma', 1.0)
        # Konwertuj do trybu z pojedynczym kanałem? Lepiej zrobić na każdym kanale osobno.
        r, g, b, a = img.split()
        # Funkcja gamma: out = 255 * (in/255)^(1/gamma)  (standardowa korekcja)
        # Dla gamma < 1 rozjaśnia, > 1 przyciemnia.
        r = r.point(lambda x: int(255 * (x/255) ** (1/gamma)))
        g = g.point(lambda x: int(255 * (x/255) ** (1/gamma)))
        b = b.point(lambda x: int(255 * (x/255) ** (1/gamma)))
        img = Image.merge('RGBA', (r, g, b, a))
        return img
    
    @staticmethod
    def _apply_white_balance(img: Image.Image, params: Dict) -> Image.Image:
        """Przesunięcie balansu bieli: zmiana kanałów R i B o niewielką wartość."""
        amount = params.get('amount', 0.0)  # wartość z zakresu np. -0.02..0.02
        # amount > 0 = cieplej (więcej czerwonego), amount < 0 = zimniej (więcej niebieskiego)
        r, g, b, a = img.split()
        r_data = np.array(r, dtype=np.float32)
        b_data = np.array(b, dtype=np.float32)
        # Przesunięcie: dla amount dodatniego dodajemy do R, odejmujemy od B
        r_shift = int(amount * 255)
        b_shift = -int(amount * 255)
        r_data = np.clip(r_data + r_shift, 0, 255).astype(np.uint8)
        b_data = np.clip(b_data + b_shift, 0, 255).astype(np.uint8)
        img = Image.merge('RGBA', (Image.fromarray(r_data), g, Image.fromarray(b_data), a))
        return img
    
    @staticmethod
    def _apply_zoom(img: Image.Image, params: Dict, target_res: Tuple[int, int]) -> Image.Image:
        zoom_factor = params.get('factor', 1.0)
        crop = params.get('crop', True)
        w, h = img.size
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if crop:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            right = left + w
            bottom = top + h
            img = img.crop((left, top, right, bottom))
        else:
            if zoom_factor < 1.0:
                canvas = Image.new("RGBA", target_res, (0, 0, 0, 0))
                paste_x = (target_res[0] - new_w) // 2
                paste_y = (target_res[1] - new_h) // 2
                canvas.paste(img, (paste_x, paste_y))
                img = canvas
            else:
                left = (new_w - target_res[0]) // 2
                top = (new_h - target_res[1]) // 2
                right = left + target_res[0]
                bottom = top + target_res[1]
                img = img.crop((left, top, right, bottom))
        return img
    
    @staticmethod
    def _apply_rotation(img: Image.Image, params: Dict, target_res: Tuple[int, int], bg_color) -> Image.Image:
        angle = params.get('angle', 0.0)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg_color + (255,))
        w, h = img.size
        left = (w - target_res[0]) // 2
        top = (h - target_res[1]) // 2
        right = left + target_res[0]
        bottom = top + target_res[1]
        img = img.crop((left, top, right, bottom))
        return img
    
    @staticmethod
    def _apply_flip(img: Image.Image, params: Dict) -> Image.Image:
        flip_mode = params.get('flip_mode', 'horizontal')
        if flip_mode == 'horizontal':
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    @staticmethod
    def _scale_cover(img: Image.Image, target_res: Tuple[int, int]) -> Image.Image:
        t_w, t_h = target_res
        img_w, img_h = img.size
        scale = max(t_w / img_w, t_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - t_w) // 2
        top = (new_h - t_h) // 2
        right = left + t_w
        bottom = top + t_h
        return img.crop((left, top, right, bottom))
    
    @staticmethod
    def _scale_contain(img: Image.Image, target_res: Tuple[int, int], bg_color) -> Image.Image:
        t_w, t_h = target_res
        img.thumbnail((t_w, t_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", target_res, bg_color + (255,))
        paste_x = (t_w - img.width) // 2
        paste_y = (t_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y), img if img.mode == 'RGBA' else None)
        return canvas
    
    @staticmethod
    def _scale_crop(img: Image.Image, target_res: Tuple[int, int]) -> Image.Image:
        t_w, t_h = target_res
        target_ratio = t_w / t_h
        img_ratio = img.width / img.height
        if img_ratio > target_ratio:
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        return img.resize(target_res, Image.Resampling.LANCZOS)

# ==============================================================================
# 4. SILNIK TEKSTOWY (bez zmian)
# ==============================================================================

class TextEngine:
    FONT_CACHE = {}
    
    @classmethod
    def get_font(cls, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        cache_key = f"{font_name}_{size}"
        if cache_key in cls.FONT_CACHE:
            return cls.FONT_CACHE[cache_key]
        font_files = {
            "League Gothic Regular": "LeagueGothic-Regular.otf",
            "League Gothic Condensed": "LeagueGothic-CondensedRegular.otf",
            "Impact": "impact.ttf",
            "Arial": "arial.ttf",
            "Roboto": "Roboto-Regular.ttf",
            "Montserrat": "Montserrat-Regular.ttf"
        }
        font_path = font_files.get(font_name, "arial.ttf")
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size)
        else:
            font = ImageFont.load_default()
        cls.FONT_CACHE[cache_key] = font
        return font
    
    @classmethod
    def render_text(
        cls,
        text: str,
        style: TextStyleConfig,
        resolution: Tuple[int, int] = OmegaCore.TARGET_RES
    ) -> Image.Image:
        if style.uppercase:
            text = text.upper()
        font_size = cls._auto_scale_font(text, style, resolution)
        font = cls.get_font(style.font_name, font_size)
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        base_pos = cls._calculate_position(text_width, text_height, style, resolution)
        pos = base_pos
        if style.random_position_jitter:
            jitter_range = style.position_jitter_range
            dx = random.randint(-jitter_range, jitter_range)
            dy = random.randint(-jitter_range, jitter_range)
            pos = (base_pos[0] + dx, base_pos[1] + dy)
        combined = Image.new("RGBA", resolution, (0, 0, 0, 0))
        if style.use_shadow:
            shadow_layer = Image.new("RGBA", resolution, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_offset_x = style.shadow_offset_x
            shadow_offset_y = style.shadow_offset_y
            if style.random_shadow_jitter:
                jitter_range = style.shadow_jitter_range
                shadow_offset_x += random.randint(-jitter_range, jitter_range)
                shadow_offset_y += random.randint(-jitter_range, jitter_range)
            shadow_pos = (pos[0] + shadow_offset_x, pos[1] + shadow_offset_y)
            shd_rgb = cls._hex_to_rgb(style.shadow_color)
            shadow_draw.text(shadow_pos, text, fill=(*shd_rgb, style.shadow_alpha), font=font)
            if style.shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(style.shadow_blur))
            combined = Image.alpha_composite(combined, shadow_layer)
        text_layer = Image.new("RGBA", resolution, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        text_rgb = cls._hex_to_rgb(style.font_color)
        stroke_rgb = cls._hex_to_rgb(style.stroke_color)
        if style.use_outline and style.stroke_width > 0:
            text_draw.text(
                pos, text,
                fill=text_rgb + (255,),
                font=font,
                stroke_width=style.stroke_width,
                stroke_fill=stroke_rgb + (255,)
            )
        else:
            text_draw.text(pos, text, fill=text_rgb + (255,), font=font)
        combined = Image.alpha_composite(combined, text_layer)
        return combined
    
    @classmethod
    def render_multiline(
        cls,
        lines: List[str],
        style: TextStyleConfig,
        resolution: Tuple[int, int] = OmegaCore.TARGET_RES
    ) -> Image.Image:
        if not lines:
            return Image.new("RGBA", resolution, (0, 0, 0, 0))
        font_size = cls._auto_scale_font(max(lines, key=len), style, resolution)
        font = cls.get_font(style.font_name, font_size)
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        line_bboxes = []
        max_width = 0
        total_height = 0
        for line in lines:
            bbox = dummy_draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            line_bboxes.append((w, h))
            max_width = max(max_width, w)
            total_height += h + style.line_spacing
        total_height -= style.line_spacing
        start_y = (resolution[1] - total_height) // 2
        combined = Image.new("RGBA", resolution, (0, 0, 0, 0))
        y_offset = start_y
        for i, line in enumerate(lines):
            w, h = line_bboxes[i]
            if style.multiline_align == "center":
                x = (resolution[0] - w) // 2
            elif style.multiline_align == "left":
                x = OmegaCore.SAFE_MARGIN
            else:
                x = resolution[0] - w - OmegaCore.SAFE_MARGIN
            if style.random_position_jitter:
                jitter_range = style.position_jitter_range
                dx = random.randint(-jitter_range, jitter_range)
                dy = random.randint(-jitter_range, jitter_range)
                line_pos = (x + dx, y_offset + dy)
            else:
                line_pos = (x, y_offset)
            line_img = cls._render_line_at(line, line_pos, style, font)
            combined = Image.alpha_composite(combined, line_img)
            y_offset += h + style.line_spacing
        return combined
    
    @classmethod
    def _render_line_at(cls, text: str, pos: Tuple[int, int], style: TextStyleConfig, font) -> Image.Image:
        res = OmegaCore.TARGET_RES
        combined = Image.new("RGBA", res, (0, 0, 0, 0))
        if style.use_shadow:
            shadow_layer = Image.new("RGBA", res, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_offset_x = style.shadow_offset_x
            shadow_offset_y = style.shadow_offset_y
            if style.random_shadow_jitter:
                jitter_range = style.shadow_jitter_range
                shadow_offset_x += random.randint(-jitter_range, jitter_range)
                shadow_offset_y += random.randint(-jitter_range, jitter_range)
            shadow_pos = (pos[0] + shadow_offset_x, pos[1] + shadow_offset_y)
            shd_rgb = cls._hex_to_rgb(style.shadow_color)
            shadow_draw.text(shadow_pos, text, fill=(*shd_rgb, style.shadow_alpha), font=font)
            if style.shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(style.shadow_blur))
            combined = Image.alpha_composite(combined, shadow_layer)
        text_layer = Image.new("RGBA", res, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        text_rgb = cls._hex_to_rgb(style.font_color)
        stroke_rgb = cls._hex_to_rgb(style.stroke_color)
        if style.use_outline and style.stroke_width > 0:
            text_draw.text(
                pos, text,
                fill=text_rgb + (255,),
                font=font,
                stroke_width=style.stroke_width,
                stroke_fill=stroke_rgb + (255,)
            )
        else:
            text_draw.text(pos, text, fill=text_rgb + (255,), font=font)
        combined = Image.alpha_composite(combined, text_layer)
        return combined
    
    @classmethod
    def _auto_scale_font(cls, text: str, style: TextStyleConfig, res: Tuple[int, int]) -> int:
        if not text:
            return style.font_size
        max_width = res[0] - (OmegaCore.SAFE_MARGIN * 2)
        font_size = style.font_size
        while font_size > 15:
            font = cls.get_font(style.font_name, font_size)
            dummy_img = Image.new("RGBA", (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            bbox = dummy_draw.textbbox((0, 0), text, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                break
            font_size -= 4
        return font_size
    
    @classmethod
    def _calculate_position(cls, w: int, h: int, style: TextStyleConfig, res: Tuple[int, int]) -> Tuple[int, int]:
        if style.position == TextPosition.TOP:
            y = OmegaCore.SAFE_MARGIN
        elif style.position == TextPosition.BOTTOM:
            y = res[1] - h - OmegaCore.SAFE_MARGIN
        elif style.position == TextPosition.CENTER:
            y = (res[1] - h) // 2
        else:
            return style.custom_position
        x = (res[0] - w) // 2
        return (x, y)
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ==============================================================================
# 5. SILNIK AUDIO (rozszerzony o pitch)
# ==============================================================================

class AudioProcessor:
    @staticmethod
    def load_audio(file_obj, target_duration: float, config: VideoConfig) -> Optional[AudioFileClip]:
        if file_obj is None:
            return None
        try:
            suffix = Path(file_obj.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_obj.getvalue())
                tmp_path = tmp.name
            audio = AudioFileClip(tmp_path)
            
            # Zmiana prędkości (speed)
            if config.enable_random_speed_change:
                speed_factor = random.uniform(*config.speed_change_range)
                audio = audio.fx(vfx.speedx, speed_factor)
            
            # Zmiana wysokości dźwięku (pitch) – uproszczone: używamy speedx z kompresją czasu? 
            # W moviepy nie ma bezpośredniego pitch shift, ale można użyć `audio.fx(vfx.pitch_shift, ...)`
            # To wymaga biblioteki `pydub` lub `librosa`. Dla uproszczenia pomijamy lub robimy speedx.
            # Możemy dodać opcję później.
            
            # Dostosowanie długości
            if audio.duration < target_duration:
                audio = audio.loop(duration=target_duration)
            else:
                audio = audio.subclip(0, target_duration)
            
            # Fade
            if config.audio_fade_in > 0:
                audio = audio.audio_fadein(config.audio_fade_in)
            if config.audio_fade_out > 0:
                audio = audio.audio_fadeout(config.audio_fade_out)
            
            # Głośność
            if config.audio_volume != 1.0:
                audio = audio.volumex(config.audio_volume)
            
            st.session_state.temp_files.append(tmp_path)
            return audio
        except Exception as e:
            logger.error(f"Błąd ładowania audio: {e}")
            return None

# ==============================================================================
# 6. MENEDŻER PROFILI (bez zmian)
# ==============================================================================

class ProfileManager:
    @staticmethod
    def save_profile(name: str, text_style: TextStyleConfig, video_config: VideoConfig):
        if 'config_profiles' not in st.session_state:
            st.session_state.config_profiles = {}
        st.session_state.config_profiles[name] = {
            'text_style': asdict(text_style),
            'video_config': asdict(video_config)
        }
        st.session_state.current_profile = name
        logger.info(f"Profil '{name}' zapisany")
    
    @staticmethod
    def load_profile(name: str) -> Tuple[Optional[TextStyleConfig], Optional[VideoConfig]]:
        profile = st.session_state.config_profiles.get(name)
        if not profile:
            return None, None
        text_style = TextStyleConfig(**profile['text_style'])
        if isinstance(text_style.position, str):
            text_style.position = TextPosition(text_style.position)
        video_config = VideoConfig(**profile['video_config'])
        if isinstance(video_config.transition, str):
            video_config.transition = TransitionType(video_config.transition)
        if 'noise_type' in profile['video_config'] and isinstance(profile['video_config']['noise_type'], str):
            video_config.noise_type = NoiseType(profile['video_config']['noise_type'])
        if 'color_shift_mode' in profile['video_config'] and isinstance(profile['video_config']['color_shift_mode'], str):
            video_config.color_shift_mode = ColorShiftMode(profile['video_config']['color_shift_mode'])
        if 'photo_duration_mode' in profile['video_config'] and isinstance(profile['video_config']['photo_duration_mode'], str):
            video_config.photo_duration_mode = PhotoDurationMode(profile['video_config']['photo_duration_mode'])
        if 'codec_profile' in profile['video_config'] and isinstance(profile['video_config']['codec_profile'], str):
            video_config.codec_profile = CodecProfile(profile['video_config']['codec_profile'])
        return text_style, video_config
    
    @staticmethod
    def delete_profile(name: str):
        if name in st.session_state.config_profiles:
            del st.session_state.config_profiles[name]
            logger.info(f"Profil '{name}' usunięty")
    
    @staticmethod
    def list_profiles() -> List[str]:
        return list(st.session_state.config_profiles.keys())

# ==============================================================================
# 7. GŁÓWNY SILNIK RENDERUJĄCY – z nowymi opcjami
# ==============================================================================

class RenderEngine:
    def __init__(self, job: RenderJob, output_dir: str = "output"):
        self.job = job
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.temp_files = []
    
    def _get_photo_duration(self) -> float:
        cfg = self.job.video_config
        if cfg.photo_duration_mode == PhotoDurationMode.FIXED:
            return cfg.photo_duration_fixed
        elif cfg.photo_duration_mode == PhotoDurationMode.RANGE:
            return random.uniform(cfg.photo_duration_min, cfg.photo_duration_max)
        else:  # LIST
            try:
                values = [float(x.strip()) for x in cfg.photo_duration_list.split(',') if x.strip()]
                if values:
                    return random.choice(values)
                else:
                    return cfg.photo_duration_fixed
            except:
                return cfg.photo_duration_fixed
    
    def _get_effective_fps(self, base_fps: int) -> float:
        """Zwraca FPS z uwzględnieniem losowej zmiany."""
        cfg = self.job.video_config
        if cfg.enable_random_fps:
            # Odchylenie o ±cfg.fps_variation procent
            variation = cfg.fps_variation / 100.0  # bo fps_variation to procent (0.1 = 0.1%)
            factor = random.uniform(1 - variation, 1 + variation)
            return base_fps * factor
        return float(base_fps)
    
    def _get_effective_bitrate(self) -> str:
        """Zwraca bitrate z ewentualną losową zmianą."""
        cfg = self.job.video_config
        if not cfg.vary_bitrate:
            return cfg.bitrate
        # Oczekujemy string np. "4000k"
        try:
            # Usuń 'k' i skonwertuj na int
            base = int(cfg.bitrate.replace('k', ''))
            variation = cfg.bitrate_variation  # 0.05 = 5%
            new_bitrate = base * random.uniform(1 - variation, 1 + variation)
            return f"{int(new_bitrate)}k"
        except:
            return cfg.bitrate
    
    def _get_ffmpeg_params(self) -> List[str]:
        """Zwraca dodatkowe parametry ffmpeg dla kodeka i profilu."""
        cfg = self.job.video_config
        params = []
        if cfg.codec_profile != CodecProfile.AUTO:
            params.extend(['-profile:v', cfg.codec_profile.value])
        return params
    
    def _strip_metadata(self, file_path: str) -> str:
        """Czyści metadane z pliku wideo, zwraca ścieżkę do nowego pliku."""
        temp_out = file_path + "_clean.mp4"
        cmd = [
            'ffmpeg', '-i', file_path,
            '-map_metadata', '-1',
            '-c', 'copy',
            '-y', temp_out
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(temp_out, file_path)  # podmieniamy oryginał
        except Exception as e:
            logger.error(f"Błąd czyszczenia metadanych: {e}")
            if os.path.exists(temp_out):
                os.remove(temp_out)
        return file_path
    
    def render(self, progress_callback=None) -> str:
        try:
            cfg = self.job.video_config
            style = self.job.text_style
            target_duration = random.uniform(cfg.target_duration_min, cfg.target_duration_max)
            
            # Określenie docelowego FPS
            effective_fps = self._get_effective_fps(cfg.fps)
            
            # Czy używamy okładki?
            use_cover = cfg.use_cover and self.job.cover_file is not None
            
            # Oblicz liczbę zdjęć
            if use_cover:
                cover_duration = self._get_photo_duration() * cfg.cover_duration_multiplier
                remaining = target_duration - cover_duration
                num_photos = max(1, int(remaining / self._get_photo_duration()))
            else:
                num_photos = max(1, int(target_duration / self._get_photo_duration()))
            
            # Wybór zdjęć
            photos = self.job.photo_files
            if cfg.randomize_photo_order:
                if cfg.repeat_photos_if_needed and len(photos) < num_photos:
                    selected_photos = random.choices(photos, k=num_photos)
                else:
                    selected_photos = random.sample(photos, min(num_photos, len(photos)))
            else:
                selected_photos = [photos[i % len(photos)] for i in range(num_photos)]
            
            clips = []
            
            # Przygotuj parametry anty-fingerprint dla obrazu
            noise_params = self._prepare_noise_params(cfg)
            color_shift_params = self._prepare_color_shift_params(cfg)
            gamma_params = self._prepare_gamma_params(cfg)
            wb_params = self._prepare_white_balance_params(cfg)
            zoom_params = self._prepare_zoom_params(cfg)
            rotation_params = self._prepare_rotation_params(cfg)
            flip_params = self._prepare_flip_params(cfg)
            
            # Okładka
            if use_cover:
                cover_img = ImageProcessor.process_image(
                    self.job.cover_file,
                    target_res=cfg.resolution,
                    mode="cover",
                    noise_params=noise_params,
                    color_shift_params=color_shift_params,
                    gamma_params=gamma_params,
                    white_balance_params=wb_params,
                    zoom_params=zoom_params,
                    rotation_params=rotation_params,
                    flip_params=flip_params
                )
                cover_duration = self._get_photo_duration() * cfg.cover_duration_multiplier
                cover_clip = ImageClip(cover_img).set_duration(cover_duration)
                clips.append(cover_clip)
            
            # Zdjęcia
            for i, photo in enumerate(selected_photos):
                # Opcjonalnie odśwież parametry dla każdego zdjęcia
                if cfg.enable_random_noise:
                    noise_params = self._prepare_noise_params(cfg)
                if cfg.enable_random_color_shift:
                    color_shift_params = self._prepare_color_shift_params(cfg)
                if cfg.enable_gamma_correction:
                    gamma_params = self._prepare_gamma_params(cfg)
                if cfg.enable_white_balance:
                    wb_params = self._prepare_white_balance_params(cfg)
                if cfg.enable_random_zoom:
                    zoom_params = self._prepare_zoom_params(cfg)
                if cfg.enable_random_rotation:
                    rotation_params = self._prepare_rotation_params(cfg)
                if cfg.enable_random_flip:
                    flip_params = self._prepare_flip_params(cfg)
                
                img = ImageProcessor.process_image(
                    photo,
                    target_res=cfg.resolution,
                    mode="cover",
                    noise_params=noise_params,
                    color_shift_params=color_shift_params,
                    gamma_params=gamma_params,
                    white_balance_params=wb_params,
                    zoom_params=zoom_params,
                    rotation_params=rotation_params,
                    flip_params=flip_params
                )
                photo_duration = self._get_photo_duration()
                clip = ImageClip(img).set_duration(photo_duration)
                clips.append(clip)
            
            # Konkatenacja
            if cfg.transition != TransitionType.NONE:
                final_clip = self._apply_transitions(clips, cfg)
            else:
                final_clip = concatenate_videoclips(clips, method="chain")
            
            # Tekst
            text_img = TextEngine.render_multiline(
                self.job.text_lines,
                style,
                resolution=cfg.resolution
            )
            text_clip = ImageClip(np.array(text_img)).set_duration(final_clip.duration)
            final = CompositeVideoClip([final_clip, text_clip], size=cfg.resolution)
            
            # Audio
            if self.job.audio_file:
                audio = AudioProcessor.load_audio(self.job.audio_file, final.duration, cfg)
                if audio:
                    final = final.set_audio(audio)
                    self.temp_files.extend(getattr(audio, 'audio_files', []))
            
            # Zapis
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_filename = f"OMEGA_{self.job.job_id}_{timestamp}.mp4"
            out_path = os.path.join(self.output_dir, out_filename)
            
            # Przygotuj parametry ffmpeg
            ffmpeg_params = self._get_ffmpeg_params()
            effective_bitrate = self._get_effective_bitrate()
            
            final.write_videofile(
                out_path,
                fps=effective_fps,
                codec=cfg.codec,
                audio_codec=cfg.audio_codec,
                preset=cfg.preset,
                bitrate=effective_bitrate,
                threads=4,
                ffmpeg_params=ffmpeg_params,
                logger=None
            )
            
            # Czyszczenie metadanych jeśli włączone
            if cfg.strip_metadata:
                out_path = self._strip_metadata(out_path)
            
            final.close()
            final_clip.close()
            for clip in clips:
                clip.close()
            
            return out_path
            
        except Exception as e:
            logger.exception(f"Błąd renderowania: {e}")
            raise
    
    def _prepare_noise_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_random_noise:
            return None
        return {
            'enabled': True,
            'type': cfg.noise_type,
            'intensity': cfg.noise_intensity
        }
    
    def _prepare_color_shift_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_random_color_shift:
            return None
        amount = random.uniform(-cfg.color_shift_range, cfg.color_shift_range)
        return {
            'enabled': True,
            'mode': cfg.color_shift_mode,
            'amount': amount
        }
    
    def _prepare_gamma_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_gamma_correction:
            return None
        gamma = random.uniform(*cfg.gamma_range)
        return {
            'enabled': True,
            'gamma': gamma
        }
    
    def _prepare_white_balance_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_white_balance:
            return None
        amount = random.uniform(*cfg.white_balance_range)
        return {
            'enabled': True,
            'amount': amount
        }
    
    def _prepare_zoom_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_random_zoom:
            return None
        factor = random.uniform(*cfg.zoom_range)
        return {
            'enabled': True,
            'factor': factor,
            'crop': cfg.zoom_crop
        }
    
    def _prepare_rotation_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_random_rotation:
            return None
        angle = random.uniform(*cfg.rotation_range)
        return {
            'enabled': True,
            'angle': angle
        }
    
    def _prepare_flip_params(self, cfg: VideoConfig) -> Optional[Dict]:
        if not cfg.enable_random_flip:
            return None
        if random.random() < cfg.flip_probability:
            return {
                'enabled': True,
                'flip_mode': random.choice(['horizontal', 'vertical'])
            }
        return {'enabled': False}
    
    def _apply_transitions(self, clips: List[VideoClip], cfg: VideoConfig) -> VideoClip:
        if len(clips) <= 1:
            return clips[0]
        # Uproszczona obsługa przejść – w pełnej wersji można rozbudować
        return concatenate_videoclips(clips, method="chain")

# ==============================================================================
# 8. INTERFEJS UŻYTKOWNIKA – z nowymi opcjami w zakładce Anti-Fingerprint
# ==============================================================================

def main():
    OmegaCore.setup_session()
    st.set_page_config(
        page_title=f"Ω OMEGA {OmegaCore.VERSION}",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    mpy_config.change_settings({"IMAGEMAGICK_BINARY": OmegaCore.get_magick_path()})
    
    # Inicjalizacja konfiguracji z obsługą starszych wersji
    if 'current_text_style' not in st.session_state:
        st.session_state.current_text_style = TextStyleConfig()
    else:
        existing = st.session_state.current_text_style
        if hasattr(existing, '__dict__'):
            d = existing.__dict__.copy()
        else:
            d = dict(existing)
        st.session_state.current_text_style = TextStyleConfig(**d)
    
    if 'current_video_config' not in st.session_state:
        st.session_state.current_video_config = VideoConfig()
    else:
        existing = st.session_state.current_video_config
        if hasattr(existing, '__dict__'):
            d = existing.__dict__.copy()
        else:
            d = dict(existing)
        st.session_state.current_video_config = VideoConfig(**d)
    
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    
    with st.sidebar:
        st.title("⚙️ OMEGA CONFIGURATOR")
        
        with st.expander("📁 Profile konfiguracji", expanded=False):
            profile_name = st.text_input("Nazwa profilu", value="default")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Zapisz profil", use_container_width=True):
                    ProfileManager.save_profile(
                        profile_name,
                        st.session_state.current_text_style,
                        st.session_state.current_video_config
                    )
                    st.success(f"Profil '{profile_name}' zapisany")
            with col2:
                profiles = ProfileManager.list_profiles()
                if profiles:
                    selected = st.selectbox("Wczytaj profil", profiles)
                    if st.button("📂 Wczytaj", use_container_width=True):
                        ts, vc = ProfileManager.load_profile(selected)
                        if ts and vc:
                            st.session_state.current_text_style = ts
                            st.session_state.current_video_config = vc
                            st.rerun()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Tekst", "🎬 Wideo", "🎵 Audio", "🖼️ Obrazy", "🛡️ Anti-Fingerprint"])
        
        with tab1:
            st.subheader("Styl tekstu")
            font_choices = ["League Gothic Regular", "League Gothic Condensed", "Impact", "Arial", "Roboto", "Montserrat"]
            st.session_state.current_text_style.font_name = st.selectbox(
                "Czcionka", font_choices,
                index=font_choices.index(st.session_state.current_text_style.font_name)
            )
            st.session_state.current_text_style.font_size = st.slider(
                "Rozmiar (max)", 20, 500, st.session_state.current_text_style.font_size
            )
            st.session_state.current_text_style.font_color = st.color_picker(
                "Kolor tekstu", st.session_state.current_text_style.font_color
            )
            st.session_state.current_text_style.stroke_width = st.slider(
                "Grubość obrysu", 0, 20, st.session_state.current_text_style.stroke_width
            )
            st.session_state.current_text_style.stroke_color = st.color_picker(
                "Kolor obrysu", st.session_state.current_text_style.stroke_color
            )
            
            st.divider()
            st.subheader("Cień")
            st.session_state.current_text_style.use_shadow = st.checkbox(
                "Użyj cienia", value=st.session_state.current_text_style.use_shadow
            )
            if st.session_state.current_text_style.use_shadow:
                st.session_state.current_text_style.shadow_offset_x = st.slider(
                    "Offset X", -100, 100, st.session_state.current_text_style.shadow_offset_x
                )
                st.session_state.current_text_style.shadow_offset_y = st.slider(
                    "Offset Y", -100, 100, st.session_state.current_text_style.shadow_offset_y
                )
                st.session_state.current_text_style.shadow_blur = st.slider(
                    "Rozmycie", 0, 50, st.session_state.current_text_style.shadow_blur
                )
                st.session_state.current_text_style.shadow_alpha = st.slider(
                    "Przezroczystość", 0, 255, st.session_state.current_text_style.shadow_alpha
                )
                st.session_state.current_text_style.shadow_color = st.color_picker(
                    "Kolor cienia", st.session_state.current_text_style.shadow_color
                )
            
            st.divider()
            st.subheader("Pozycja")
            pos_options = [p.value for p in TextPosition]
            current_pos = st.session_state.current_text_style.position.value
            selected_pos = st.selectbox("Pozycja", pos_options, index=pos_options.index(current_pos))
            st.session_state.current_text_style.position = TextPosition(selected_pos)
            
            if st.session_state.current_text_style.position == TextPosition.CUSTOM:
                col1, col2 = st.columns(2)
                with col1:
                    x = st.number_input("X", 0, 1920, st.session_state.current_text_style.custom_position[0])
                with col2:
                    y = st.number_input("Y", 0, 1080, st.session_state.current_text_style.custom_position[1])
                st.session_state.current_text_style.custom_position = (x, y)
            
            st.session_state.current_text_style.multiline_align = st.selectbox(
                "Wyrównanie wielolinijkowe", ["center", "left", "right"]
            )
            st.session_state.current_text_style.uppercase = st.checkbox(
                "Wielkie litery", value=st.session_state.current_text_style.uppercase
            )
            
            st.divider()
            st.subheader("🛡️ Anti-Fingerprint tekstu")
            st.session_state.current_text_style.random_position_jitter = st.checkbox(
                "Losowe przesunięcie tekstu (jitter)", 
                value=st.session_state.current_text_style.random_position_jitter
            )
            if st.session_state.current_text_style.random_position_jitter:
                st.session_state.current_text_style.position_jitter_range = st.slider(
                    "Maksymalne przesunięcie (piksele)", 1, 50, 
                    st.session_state.current_text_style.position_jitter_range
                )
            st.session_state.current_text_style.random_shadow_jitter = st.checkbox(
                "Losowe przesunięcie cienia",
                value=st.session_state.current_text_style.random_shadow_jitter
            )
            if st.session_state.current_text_style.random_shadow_jitter:
                st.session_state.current_text_style.shadow_jitter_range = st.slider(
                    "Zakres jittera cienia", 1, 20,
                    st.session_state.current_text_style.shadow_jitter_range
                )
        
        with tab2:
            st.subheader("Ustawienia wideo")
            st.session_state.current_video_config.fps = st.selectbox(
                "Klatki na sekundę", [24, 25, 30, 60], index=0
            )
            st.session_state.current_video_config.codec = st.selectbox(
                "Kodek", ["libx264", "libx265", "mpeg4"], index=0
            )
            st.session_state.current_video_config.preset = st.selectbox(
                "Preset (szybkość)", ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"], index=0
            )
            st.session_state.current_video_config.bitrate = st.selectbox(
                "Bitrate", ["2000k", "4000k", "6000k", "8000k"], index=1
            )
            
            st.divider()
            st.subheader("Czas trwania filmu")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.current_video_config.target_duration_min = st.number_input(
                    "Min (s)", 3.0, 20.0, st.session_state.current_video_config.target_duration_min, 0.01,
                    format="%.2f"
                )
            with col2:
                st.session_state.current_video_config.target_duration_max = st.number_input(
                    "Max (s)", 3.0, 20.0, st.session_state.current_video_config.target_duration_max, 0.01,
                    format="%.2f"
                )
            
            st.divider()
            st.subheader("Okładka")
            st.session_state.current_video_config.use_cover = st.checkbox(
                "Użyj okładek (jeśli przesłane)", value=st.session_state.current_video_config.use_cover
            )
            if st.session_state.current_video_config.use_cover:
                st.session_state.current_video_config.cover_duration_multiplier = st.slider(
                    "Mnożnik czasu okładki", 1.0, 5.0, st.session_state.current_video_config.cover_duration_multiplier, 0.1
                )
            
            st.divider()
            st.subheader("Czas trwania zdjęcia")
            mode_options = [m.value for m in PhotoDurationMode]
            current_mode = st.session_state.current_video_config.photo_duration_mode.value
            selected_mode = st.selectbox("Tryb", mode_options, index=mode_options.index(current_mode))
            st.session_state.current_video_config.photo_duration_mode = PhotoDurationMode(selected_mode)
            
            if st.session_state.current_video_config.photo_duration_mode == PhotoDurationMode.FIXED:
                st.session_state.current_video_config.photo_duration_fixed = st.slider(
                    "Stały czas (s)", 0.05, 1.0, st.session_state.current_video_config.photo_duration_fixed, 0.01
                )
            elif st.session_state.current_video_config.photo_duration_mode == PhotoDurationMode.RANGE:
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.current_video_config.photo_duration_min = st.number_input(
                        "Min (s)", 0.05, 1.0, st.session_state.current_video_config.photo_duration_min, 0.01
                    )
                with col2:
                    st.session_state.current_video_config.photo_duration_max = st.number_input(
                        "Max (s)", 0.05, 1.0, st.session_state.current_video_config.photo_duration_max, 0.01
                    )
            else:  # LIST
                st.session_state.current_video_config.photo_duration_list = st.text_input(
                    "Lista wartości (oddzielone przecinkami)", 
                    value=st.session_state.current_video_config.photo_duration_list
                )
            
            st.divider()
            st.subheader("Przejścia")
            trans_options = [t.value for t in TransitionType]
            current_trans = st.session_state.current_video_config.transition.value
            selected_trans = st.selectbox("Typ przejścia", trans_options, index=trans_options.index(current_trans))
            st.session_state.current_video_config.transition = TransitionType(selected_trans)
            st.session_state.current_video_config.transition_duration = st.slider(
                "Czas przejścia (s)", 0.0, 1.0, st.session_state.current_video_config.transition_duration, 0.05
            )
        
        with tab3:
            st.subheader("Ustawienia audio")
            st.session_state.current_video_config.audio_volume = st.slider(
                "Głośność", 0.0, 2.0, st.session_state.current_video_config.audio_volume, 0.1
            )
            st.session_state.current_video_config.audio_fade_in = st.slider(
                "Fade in (s)", 0.0, 3.0, st.session_state.current_video_config.audio_fade_in, 0.1
            )
            st.session_state.current_video_config.audio_fade_out = st.slider(
                "Fade out (s)", 0.0, 3.0, st.session_state.current_video_config.audio_fade_out, 0.1
            )
            
            st.divider()
            st.subheader("🛡️ Anti-Fingerprint audio")
            st.session_state.current_video_config.enable_random_pitch_shift = st.checkbox(
                "Losowa zmiana wysokości dźwięku (pitch)", 
                value=st.session_state.current_video_config.enable_random_pitch_shift
            )
            if st.session_state.current_video_config.enable_random_pitch_shift:
                col1, col2 = st.columns(2)
                with col1:
                    min_pitch = st.number_input("Min pitch", 0.9, 1.1, 
                                                st.session_state.current_video_config.pitch_shift_range[0], 0.01)
                with col2:
                    max_pitch = st.number_input("Max pitch", 0.9, 1.1,
                                                st.session_state.current_video_config.pitch_shift_range[1], 0.01)
                st.session_state.current_video_config.pitch_shift_range = (min_pitch, max_pitch)
            
            st.session_state.current_video_config.enable_random_speed_change = st.checkbox(
                "Losowa zmiana tempa (speed)",
                value=st.session_state.current_video_config.enable_random_speed_change
            )
            if st.session_state.current_video_config.enable_random_speed_change:
                col1, col2 = st.columns(2)
                with col1:
                    min_speed = st.number_input("Min speed", 0.95, 1.05,
                                                st.session_state.current_video_config.speed_change_range[0], 0.01)
                with col2:
                    max_speed = st.number_input("Max speed", 0.95, 1.05,
                                                st.session_state.current_video_config.speed_change_range[1], 0.01)
                st.session_state.current_video_config.speed_change_range = (min_speed, max_speed)
        
        with tab4:
            st.subheader("Przetwarzanie obrazów")
            img_mode = st.selectbox("Tryb skalowania", ["cover", "contain", "crop"])
            st.session_state.img_mode = img_mode
        
        with tab5:
            st.subheader("🛡️ Anti-Fingerprint obrazu")
            st.markdown("Subtelne, losowe modyfikacje każdego obrazu.")
            
            # Szum
            st.session_state.current_video_config.enable_random_noise = st.checkbox(
                "Dodaj losowy szum (grain)", value=st.session_state.current_video_config.enable_random_noise
            )
            if st.session_state.current_video_config.enable_random_noise:
                noise_types = [nt.value for nt in NoiseType]
                current_noise = st.session_state.current_video_config.noise_type.value
                selected_noise = st.selectbox("Typ szumu", noise_types, index=noise_types.index(current_noise))
                st.session_state.current_video_config.noise_type = NoiseType(selected_noise)
                st.session_state.current_video_config.noise_intensity = st.slider(
                    "Intensywność szumu (0.005 = 0.5%)", 0.0, 0.02, 
                    st.session_state.current_video_config.noise_intensity, 0.001,
                    format="%.3f"
                )
            
            # Korekta kolorów (jasność, kontrast, nasycenie)
            st.session_state.current_video_config.enable_random_color_shift = st.checkbox(
                "Losowa korekta kolorów (jasność/kontrast/nasycenie)", 
                value=st.session_state.current_video_config.enable_random_color_shift
            )
            if st.session_state.current_video_config.enable_random_color_shift:
                color_modes = [cm.value for cm in ColorShiftMode if cm.value not in ['gamma', 'white_balance']]
                current_mode = st.session_state.current_video_config.color_shift_mode.value
                # Zabezpieczenie przed starymi wartościami
                if current_mode not in color_modes:
                    current_mode = 'brightness'
                selected_mode = st.selectbox("Tryb korekty", color_modes, index=color_modes.index(current_mode))
                st.session_state.current_video_config.color_shift_mode = ColorShiftMode(selected_mode)
                st.session_state.current_video_config.color_shift_range = st.slider(
                    "Zakres zmian (±)", 0.0, 0.1, 
                    st.session_state.current_video_config.color_shift_range, 0.005,
                    format="%.3f"
                )
            
            # Gamma
            st.session_state.current_video_config.enable_gamma_correction = st.checkbox(
                "Korekcja gamma (subtelna)", value=st.session_state.current_video_config.enable_gamma_correction
            )
            if st.session_state.current_video_config.enable_gamma_correction:
                col1, col2 = st.columns(2)
                with col1:
                    min_gamma = st.number_input("Min gamma", 0.95, 1.05, 
                                                st.session_state.current_video_config.gamma_range[0], 0.01)
                with col2:
                    max_gamma = st.number_input("Max gamma", 0.95, 1.05,
                                                st.session_state.current_video_config.gamma_range[1], 0.01)
                st.session_state.current_video_config.gamma_range = (min_gamma, max_gamma)
            
            # Balans bieli (temperatura)
            st.session_state.current_video_config.enable_white_balance = st.checkbox(
                "Balans bieli (temperatura)", value=st.session_state.current_video_config.enable_white_balance
            )
            if st.session_state.current_video_config.enable_white_balance:
                col1, col2 = st.columns(2)
                with col1:
                    min_wb = st.number_input("Min przesunięcie", -0.05, 0.05, 
                                                st.session_state.current_video_config.white_balance_range[0], 0.005,
                                                format="%.3f")
                with col2:
                    max_wb = st.number_input("Max przesunięcie", -0.05, 0.05,
                                                st.session_state.current_video_config.white_balance_range[1], 0.005,
                                                format="%.3f")
                st.session_state.current_video_config.white_balance_range = (min_wb, max_wb)
            
            # Zoom
            st.session_state.current_video_config.enable_random_zoom = st.checkbox(
                "Losowy zoom (skala)", value=st.session_state.current_video_config.enable_random_zoom
            )
            if st.session_state.current_video_config.enable_random_zoom:
                col1, col2 = st.columns(2)
                with col1:
                    min_zoom = st.number_input("Min zoom", 0.95, 1.05, 
                                                st.session_state.current_video_config.zoom_range[0], 0.01)
                with col2:
                    max_zoom = st.number_input("Max zoom", 0.95, 1.05,
                                                st.session_state.current_video_config.zoom_range[1], 0.01)
                st.session_state.current_video_config.zoom_range = (min_zoom, max_zoom)
                st.session_state.current_video_config.zoom_crop = st.checkbox(
                    "Przytnij (zamiast dodawania tła)", value=st.session_state.current_video_config.zoom_crop
                )
            
            # Obrót
            st.session_state.current_video_config.enable_random_rotation = st.checkbox(
                "Losowy obrót", value=st.session_state.current_video_config.enable_random_rotation
            )
            if st.session_state.current_video_config.enable_random_rotation:
                col1, col2 = st.columns(2)
                with col1:
                    min_angle = st.number_input("Min kąt (°)", -2.0, 2.0, 
                                                st.session_state.current_video_config.rotation_range[0], 0.1)
                with col2:
                    max_angle = st.number_input("Max kąt (°)", -2.0, 2.0,
                                                st.session_state.current_video_config.rotation_range[1], 0.1)
                st.session_state.current_video_config.rotation_range = (min_angle, max_angle)
            
            # Odbicie
            st.session_state.current_video_config.enable_random_flip = st.checkbox(
                "Losowe odbicie lustrzane", value=st.session_state.current_video_config.enable_random_flip
            )
            if st.session_state.current_video_config.enable_random_flip:
                st.session_state.current_video_config.flip_probability = st.slider(
                    "Prawdopodobieństwo odbicia", 0.0, 0.5, 
                    st.session_state.current_video_config.flip_probability, 0.05
                )
            
            st.divider()
            st.subheader("🛡️ Anti-Fingerprint czasowy")
            
            # Losowy FPS
            st.session_state.current_video_config.enable_random_fps = st.checkbox(
                "Losowa zmiana FPS (np. 30 → 29.97-30.03)", 
                value=st.session_state.current_video_config.enable_random_fps
            )
            if st.session_state.current_video_config.enable_random_fps:
                st.session_state.current_video_config.fps_variation = st.slider(
                    "Odchylenie (%)", 0.0, 0.5, 
                    st.session_state.current_video_config.fps_variation, 0.01,
                    format="%.2f"
                )
            
            # Zmiana prędkości (już jest w audio, ale dotyczy całego filmu)
            # (jest w zakładce audio)
            
            # Decimate (usuwanie klatek) – opcjonalne
            st.session_state.current_video_config.enable_decimate = st.checkbox(
                "Usuwanie co N-tej klatki (eksperymentalne)", 
                value=st.session_state.current_video_config.enable_decimate
            )
            if st.session_state.current_video_config.enable_decimate:
                st.session_state.current_video_config.decimate_every = st.number_input(
                    "Usuń co n-tą klatkę", 10, 500, 
                    st.session_state.current_video_config.decimate_every, 10
                )
            
            st.divider()
            st.subheader("🛡️ Anti-Fingerprint kontenera")
            
            # Profil kodeka
            profile_options = [p.value for p in CodecProfile]
            current_profile = st.session_state.current_video_config.codec_profile.value
            selected_profile = st.selectbox("Profil kodeka", profile_options, index=profile_options.index(current_profile))
            st.session_state.current_video_config.codec_profile = CodecProfile(selected_profile)
            
            # Zmienny bitrate
            st.session_state.current_video_config.vary_bitrate = st.checkbox(
                "Losowa zmiana bitrate (VBR z odchyleniem)", 
                value=st.session_state.current_video_config.vary_bitrate
            )
            if st.session_state.current_video_config.vary_bitrate:
                st.session_state.current_video_config.bitrate_variation = st.slider(
                    "Odchylenie bitrate (±%)", 0.0, 0.2, 
                    st.session_state.current_video_config.bitrate_variation, 0.01,
                    format="%.2f"
                )
            
            # Czyszczenie metadanych
            st.session_state.current_video_config.strip_metadata = st.checkbox(
                "Usuń wszystkie metadane z pliku", 
                value=st.session_state.current_video_config.strip_metadata
            )
        
        st.divider()
        
        default_txts = (
            "Most unique spreadsheet rn\nIg brands ain't safe\nPOV: You created best ig brands spreadsheet\n"
            "Best archive spreadsheet rn\nArchive fashion ain't safe\nBest ig brands spreadsheet oat.\n"
            "Best archive fashion spreadsheet rn.\nEven ig brands ain't safe\nPOV: you have best spreadsheet on tiktok\n"
            "pov: you found best spreadsheet\nSwagest spreadsheet ever\nSwagest spreadsheet in 2026\n"
            "Coldest spreadsheet rn.\nNo more gatekeeping this spreadsheet\nUltimate archive clothing vault\n"
            "Only fashion sheet needed\nBest fashion sheet oat\nIG brands ain't safe\n"
            "I found the holy grail of spreadsheets\nTook me 3 months to create best spreadsheet\n"
            "I’m actually done gatekeeping this\nWhy did nobody tell me about this sheet earlier?\n"
            "Honestly, best finds i’ve ever seen\npov: you’re not gatekeeping your sources anymore\n"
            "pov: your fits are about to get 10x better\npov: you found the spreadsheet everyone was looking for\n"
            "me after finding this archive sheet:\nThis spreadsheet is actually crazy\n"
            "archive pieces you actually need\nSpreadsheet just drooped"
        )
        raw_texts = st.text_area("Baza tekstów (każda linia to osobny tekst)", default_txts, height=150)
        texts_list = [t.strip() for t in raw_texts.split('\n') if t.strip()]
        
        with st.expander("👁️ LIVE PREVIEW", expanded=False):
            preview_text = st.text_input("Tekst testowy", "OMEGA PREVIEW")
            if preview_text:
                preview_img = TextEngine.render_text(
                    preview_text,
                    st.session_state.current_text_style,
                    resolution=OmegaCore.TARGET_RES
                )
                bg = Image.new("RGB", OmegaCore.TARGET_RES, (30, 30, 30))
                bg.paste(preview_img, (0, 0), preview_img)
                st.image(bg, caption="Podgląd na żywo", use_container_width=True)
    
    # === GŁÓWNY INTERFEJS ===
    st.title(f"Ω OMEGA {OmegaCore.VERSION} – MASOWY GENERATOR FILMÓW")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        u_c = st.file_uploader(
            "📁 Okładki (pliki startowe)",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            accept_multiple_files=True
        )
    with col2:
        u_p = st.file_uploader(
            "🖼️ Zdjęcia (do wstawienia)",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            accept_multiple_files=True
        )
    with col3:
        u_m = st.file_uploader(
            "🎵 Muzyka (opcjonalnie)",
            type=['mp3', 'wav', 'm4a', 'aac', 'ogg'],
            accept_multiple_files=True
        )
    
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        chunk_size = st.number_input("Liczba filmów w paczce ZIP", min_value=1, max_value=100, value=20)
    with col_adv2:
        output_dir = st.text_input("Katalog wyjściowy", value="output")
    
    if st.button("🚀 ROZPOCZNIJ PRODUKCJĘ MASOWĄ", type="primary", use_container_width=True):
        if not u_p:
            st.error("Wgraj co najmniej jedno zdjęcie!")
        else:
            st.session_state.temp_dir = tempfile.mkdtemp(prefix="omega_")
            st.session_state.v_results = []
            st.session_state.zip_files = []
            
            progress_bar = st.progress(0, text="Inicjalizacja...")
            status_text = st.empty()
            
            cfg = st.session_state.current_video_config
            use_cover = cfg.use_cover and u_c
            
            # Określamy liczbę filmów do wygenerowania
            if use_cover:
                total_jobs = len(u_c)
                cover_files = u_c
            else:
                total_jobs = 1
                cover_files = [None]
            
            video_paths = []
            
            for idx in range(total_jobs):
                status_text.text(f"Renderowanie filmu {idx+1} z {total_jobs}...")
                
                cover_file = cover_files[idx] if use_cover else None
                audio_file = random.choice(u_m) if u_m else None
                
                num_lines = random.randint(1, 3)
                selected_texts = random.sample(texts_list, min(num_lines, len(texts_list)))
                
                job = RenderJob(
                    cover_file=cover_file,
                    photo_files=u_p,
                    audio_file=audio_file,
                    text_lines=selected_texts,
                    text_style=st.session_state.current_text_style,
                    video_config=cfg,
                    job_id=f"JOB{idx+1:03d}"
                )
                
                engine = RenderEngine(job, output_dir=output_dir)
                try:
                    out_path = engine.render()
                    video_paths.append(out_path)
                    st.session_state.v_results.append(out_path)
                    st.session_state.temp_files.extend(engine.temp_files)
                except Exception as e:
                    st.error(f"Błąd renderowania filmu {idx+1}: {e}")
                    logger.exception("Błąd renderowania")
                
                progress_bar.progress((idx + 1) / total_jobs)
            
            status_text.text("Pakowanie filmów do archiwów...")
            os.makedirs("zips", exist_ok=True)
            for i in range(0, len(video_paths), chunk_size):
                chunk = video_paths[i:i + chunk_size]
                part_num = i // chunk_size + 1
                zip_name = f"zips/OMEGA_PART_{part_num}.zip"
                with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_STORED) as zf:
                    for video in chunk:
                        zf.write(video, arcname=os.path.basename(video))
                st.session_state.zip_files.append(zip_name)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Produkcja zakończona!")
            st.success(f"Wygenerowano {len(video_paths)} filmów w {len(st.session_state.zip_files)} paczkach.")
    
    if st.session_state.zip_files:
        st.divider()
        st.subheader("📥 Gotowe paczki ZIP")
        cols = st.columns(len(st.session_state.zip_files))
        for idx, zip_path in enumerate(st.session_state.zip_files):
            with open(zip_path, "rb") as f:
                cols[idx].download_button(
                    label=f"📦 Pobierz PART {idx+1}",
                    data=f,
                    file_name=os.path.basename(zip_path),
                    use_container_width=True,
                    key=f"download_{idx}"
                )
        
        if st.button("🗑️ Wyczyść pliki tymczasowe i wyniki", use_container_width=True):
            for f in st.session_state.v_results:
                try: os.remove(f)
                except: pass
            for f in st.session_state.zip_files:
                try: os.remove(f)
                except: pass
            for f in st.session_state.temp_files:
                try: os.remove(f)
                except: pass
            OmegaCore.cleanup_temp()
            st.session_state.v_results = []
            st.session_state.zip_files = []
            st.session_state.temp_files = []
            st.rerun()

if __name__ == "__main__":
    main()
