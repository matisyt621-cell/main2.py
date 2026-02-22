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

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmegaCore:
    """Główna klasa konfiguracyjna aplikacji Omega Video Generator."""
    
    VERSION = "V13.0 PRO (1000-LINER)"
    TARGET_RES = (1080, 1920)  # 9:16
    SAFE_MARGIN = 90  # margines boczny dla tekstu
    SUPPORTED_IMG_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    SUPPORTED_AUDIO_FORMATS = ('.mp3', '.wav', '.m4a', '.aac', '.ogg')
    
    @staticmethod
    def setup_session():
        """Inicjalizacja zmiennych sesyjnych."""
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
        """Zwraca ścieżkę do ImageMagick dla różnych systemów."""
        if os.name == 'posix':
            # Linux/Mac
            common_paths = ["/usr/bin/convert", "/usr/local/bin/convert"]
            for path in common_paths:
                if os.path.exists(path):
                    return path
            return "/usr/bin/convert"
        else:
            # Windows
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
        """Sprzątanie katalogu tymczasowego."""
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            st.session_state.temp_dir = None
            logger.info("Katalog tymczasowy usunięty")

# ==============================================================================
# 2. MODELE DANYCH (dataclass)
# ==============================================================================

class TextPosition(Enum):
    """Pozycje tekstu na ekranie."""
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    CUSTOM = "custom"

class TransitionType(Enum):
    """Typy przejść między klipami."""
    NONE = "none"
    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"

@dataclass
class TextStyleConfig:
    """Konfiguracja stylu tekstu."""
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
    animation_in: str = "none"  # fade, slide, zoom
    animation_out: str = "none"
    animation_duration: float = 0.5

@dataclass
class VideoConfig:
    """Globalna konfiguracja generowania filmów."""
    resolution: Tuple[int, int] = (1080, 1920)
    fps: int = 24
    codec: str = "libx264"
    audio_codec: str = "aac"
    preset: str = "ultrafast"
    bitrate: str = "4000k"
    target_duration_min: float = 8.5
    target_duration_max: float = 9.8
    cover_duration_multiplier: float = 3.0  # okładka będzie trwać speed * multiplier
    photo_duration: float = 0.15  # domyślna szybkość (jeśli nie użyto suwaka)
    transition: TransitionType = TransitionType.NONE
    transition_duration: float = 0.2
    randomize_photo_order: bool = True
    repeat_photos_if_needed: bool = True
    audio_fade_in: float = 0.5
    audio_fade_out: float = 0.5
    audio_volume: float = 1.0
    add_timestamp: bool = False
    output_dir: str = "output"

@dataclass
class RenderJob:
    """Pojedyncze zadanie renderowania."""
    cover_file: Any  # obiekt pliku Streamlit
    photo_files: List[Any]
    audio_file: Optional[Any]
    text_lines: List[str]
    text_style: TextStyleConfig
    video_config: VideoConfig
    job_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

# ==============================================================================
# 3. SILNIK GRAFICZNY – ZAAWANSOWANA OBSŁUGA OBRAZÓW
# ==============================================================================

class ImageProcessor:
    """Przetwarzanie obrazów: skalowanie, przycinanie, filtry, efekty."""
    
    @staticmethod
    def process_image(
        file_obj,
        target_res: Tuple[int, int] = OmegaCore.TARGET_RES,
        mode: str = "cover",  # cover, contain, crop
        background_color: Tuple[int, int, int] = (0, 0, 0),
        apply_blur: bool = False,
        blur_radius: float = 0,
        enhance_contrast: float = 1.0,
        enhance_sharpness: float = 1.0,
        grayscale: bool = False
    ) -> np.ndarray:
        """Główna funkcja przetwarzania obrazu."""
        try:
            # Wczytanie i konwersja
            file_bytes = file_obj.getvalue()
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img).convert("RGBA")
            
            # Zastosowanie filtrów przed skalowaniem
            if grayscale:
                img = img.convert("L").convert("RGBA")
            
            if enhance_contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(enhance_contrast)
            
            if enhance_sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(enhance_sharpness)
            
            # Skalowanie do docelowej rozdzielczości
            if mode == "cover":
                img = ImageProcessor._scale_cover(img, target_res)
            elif mode == "contain":
                img = ImageProcessor._scale_contain(img, target_res, background_color)
            elif mode == "crop":
                img = ImageProcessor._scale_crop(img, target_res)
            
            # Opcjonalne rozmycie tła (jeśli tryb contain, można dodać rozmyte tło)
            if apply_blur and blur_radius > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Konwersja do RGB (moviepy wymaga RGB)
            return np.array(img.convert("RGB"))
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania obrazu: {e}")
            # Zwróć czarny obraz w razie błędu
            return np.zeros((target_res[1], target_res[0], 3), dtype=np.uint8)
    
    @staticmethod
    def _scale_cover(img: Image.Image, target_res: Tuple[int, int]) -> Image.Image:
        """Skaluje obraz tak, aby wypełnił cały obszar (przycinając nadmiar)."""
        t_w, t_h = target_res
        img_w, img_h = img.size
        scale = max(t_w / img_w, t_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Przytnij do środka
        left = (new_w - t_w) // 2
        top = (new_h - t_h) // 2
        right = left + t_w
        bottom = top + t_h
        img = img.crop((left, top, right, bottom))
        return img
    
    @staticmethod
    def _scale_contain(img: Image.Image, target_res: Tuple[int, int], bg_color) -> Image.Image:
        """Skaluje obraz, aby zmieścił się w obszarze, dodając tło."""
        t_w, t_h = target_res
        img.thumbnail((t_w, t_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", target_res, bg_color + (255,))
        paste_x = (t_w - img.width) // 2
        paste_y = (t_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y), img if img.mode == 'RGBA' else None)
        return canvas
    
    @staticmethod
    def _scale_crop(img: Image.Image, target_res: Tuple[int, int]) -> Image.Image:
        """Przycina obraz do proporcji docelowej bez skalowania (jeśli większy)."""
        t_w, t_h = target_res
        target_ratio = t_w / t_h
        img_ratio = img.width / img.height
        
        if img_ratio > target_ratio:
            # obraz szerszy – przytnij szerokość
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # obraz wyższy – przytnij wysokość
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        
        # Skaluj do docelowej rozdzielczości
        return img.resize(target_res, Image.Resampling.LANCZOS)
    
    @staticmethod
    def apply_ken_burns(img: np.ndarray, duration: float, fps: int, zoom_scale: float = 0.1) -> VideoClip:
        """Tworzy klip wideo z efektem powolnego zoomu (Ken Burns)."""
        from moviepy.video.fx.resize import resize
        
        def make_frame(t):
            progress = t / duration
            zoom = 1 + zoom_scale * progress
            h, w = img.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            # Oblicz przesunięcie, aby zachować środek
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
            
            # Skaluj obraz
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Przytnij do oryginalnego rozmiaru
            pil_img = pil_img.crop((x_offset, y_offset, x_offset + w, y_offset + h))
            return np.array(pil_img)
        
        return VideoClip(make_frame, duration=duration).set_fps(fps)

# ==============================================================================
# 4. SILNIK TEKSTOWY – AUTO-SCALE, WIELE WARSTW, ANIMACJE
# ==============================================================================

class TextEngine:
    """Zaawansowane renderowanie tekstu z auto-scale, cieniami i animacjami."""
    
    FONT_CACHE = {}
    
    @classmethod
    def get_font(cls, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Pobiera czcionkę z cache lub ładuje z pliku."""
        cache_key = f"{font_name}_{size}"
        if cache_key in cls.FONT_CACHE:
            return cls.FONT_CACHE[cache_key]
        
        # Mapowanie nazw na pliki
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
            # Fallback do domyślnej
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
        """Renderuje pojedyńczą linię tekstu z auto-scale."""
        # Przygotowanie tekstu
        if style.uppercase:
            text = text.upper()
        
        # Auto-scale czcionki
        font_size = cls._auto_scale_font(text, style, resolution)
        font = cls.get_font(style.font_name, font_size)
        
        # Pomiar tekstu
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Pozycja
        pos = cls._calculate_position(text_width, text_height, style, resolution)
        
        # Warstwy
        combined = Image.new("RGBA", resolution, (0, 0, 0, 0))
        
        if style.use_shadow:
            shadow_layer = Image.new("RGBA", resolution, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_pos = (pos[0] + style.shadow_offset_x, pos[1] + style.shadow_offset_y)
            
            # Konwersja koloru cienia
            shd_rgb = cls._hex_to_rgb(style.shadow_color)
            shadow_draw.text(shadow_pos, text, fill=(*shd_rgb, style.shadow_alpha), font=font)
            
            if style.shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(style.shadow_blur))
            
            combined = Image.alpha_composite(combined, shadow_layer)
        
        # Tekst główny z obrysem
        text_layer = Image.new("RGBA", resolution, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        
        # Konwersja koloru tekstu
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
        """Renderuje wiele linii tekstu (jedna pod drugą)."""
        if not lines:
            return Image.new("RGBA", resolution, (0, 0, 0, 0))
        
        # Auto-scale dla każdej linii? Najlepiej dla całego bloku
        # Uproszczenie: używamy tej samej czcionki dla wszystkich linii
        font_size = cls._auto_scale_font(max(lines, key=len), style, resolution)
        font = cls.get_font(style.font_name, font_size)
        
        # Pomiar każdej linii
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
        total_height -= style.line_spacing  # ostatni bez odstępu
        
        # Pozycja startowa (wyśrodkowana w pionie)
        start_y = (resolution[1] - total_height) // 2
        
        combined = Image.new("RGBA", resolution, (0, 0, 0, 0))
        y_offset = start_y
        
        for i, line in enumerate(lines):
            w, h = line_bboxes[i]
            # Pozycja w poziomie
            if style.multiline_align == "center":
                x = (resolution[0] - w) // 2
            elif style.multiline_align == "left":
                x = OmegaCore.SAFE_MARGIN
            else:  # right
                x = resolution[0] - w - OmegaCore.SAFE_MARGIN
            
            # Renderuj pojedynczą linię
            line_img = cls.render_text(line, style, resolution)
            # Wytnij tylko obszar tekstu (aby nie nakładać cieni wielokrotnie)
            # Prościej: narysuj na warstwie, ale musimy kontrolować pozycję
            # Użyjemy osobnej funkcji do rysowania linii w konkretnym miejscu
            combined = Image.alpha_composite(combined, cls._render_line_at(line, (x, y_offset), style, font))
            y_offset += h + style.line_spacing
        
        return combined
    
    @classmethod
    def _render_line_at(cls, text: str, pos: Tuple[int, int], style: TextStyleConfig, font) -> Image.Image:
        """Renderuje linię tekstu w podanej pozycji (pomija auto-scale)."""
        res = OmegaCore.TARGET_RES
        combined = Image.new("RGBA", res, (0, 0, 0, 0))
        
        if style.use_shadow:
            shadow_layer = Image.new("RGBA", res, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_pos = (pos[0] + style.shadow_offset_x, pos[1] + style.shadow_offset_y)
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
        """Automatycznie dobiera rozmiar czcionki, aby tekst zmieścił się w marginesach."""
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
        """Oblicza pozycję tekstu na podstawie stylu."""
        if style.position == TextPosition.TOP:
            y = OmegaCore.SAFE_MARGIN
        elif style.position == TextPosition.BOTTOM:
            y = res[1] - h - OmegaCore.SAFE_MARGIN
        elif style.position == TextPosition.CENTER:
            y = (res[1] - h) // 2
        else:  # CUSTOM
            return style.custom_position
        
        x = (res[0] - w) // 2
        return (x, y)
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ==============================================================================
# 5. SILNIK AUDIO
# ==============================================================================

class AudioProcessor:
    """Przetwarzanie audio: przycinanie, fade, zapętlanie."""
    
    @staticmethod
    def load_audio(file_obj, target_duration: float, config: VideoConfig) -> Optional[AudioFileClip]:
        """Ładuje audio, przycina do docelowej długości, dodaje fade."""
        if file_obj is None:
            return None
        
        try:
            # Zapis tymczasowy
            suffix = Path(file_obj.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_obj.getvalue())
                tmp_path = tmp.name
            
            audio = AudioFileClip(tmp_path)
            
            # Sprawdź długość
            if audio.duration < target_duration:
                # Zapętl jeśli krótsze
                audio = audio.loop(duration=target_duration)
            else:
                # Przytnij jeśli dłuższe
                audio = audio.subclip(0, target_duration)
            
            # Fade
            if config.audio_fade_in > 0:
                audio = audio.audio_fadein(config.audio_fade_in)
            if config.audio_fade_out > 0:
                audio = audio.audio_fadeout(config.audio_fade_out)
            
            # Głośność
            if config.audio_volume != 1.0:
                audio = audio.volumex(config.audio_volume)
            
            # Usuń tymczasowy plik po załadowaniu (nie możemy, bo audio jest używane)
            # Zostanie usunięty przy cleanup
            st.session_state.temp_files.append(tmp_path)
            return audio
            
        except Exception as e:
            logger.error(f"Błąd ładowania audio: {e}")
            return None

# ==============================================================================
# 6. MENEDŻER PROJEKTÓW I PROFILI
# ==============================================================================

class ProfileManager:
    """Zarządzanie profilami konfiguracji."""
    
    @staticmethod
    def save_profile(name: str, text_style: TextStyleConfig, video_config: VideoConfig):
        """Zapisuje profil do sesji."""
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
        """Ładuje profil z sesji."""
        profile = st.session_state.config_profiles.get(name)
        if not profile:
            return None, None
        
        text_style = TextStyleConfig(**profile['text_style'])
        # Konwersja enum
        if isinstance(text_style.position, str):
            text_style.position = TextPosition(text_style.position)
        
        video_config = VideoConfig(**profile['video_config'])
        if isinstance(video_config.transition, str):
            video_config.transition = TransitionType(video_config.transition)
        
        return text_style, video_config
    
    @staticmethod
    def delete_profile(name: str):
        """Usuwa profil."""
        if name in st.session_state.config_profiles:
            del st.session_state.config_profiles[name]
            logger.info(f"Profil '{name}' usunięty")
    
    @staticmethod
    def list_profiles() -> List[str]:
        """Lista dostępnych profili."""
        return list(st.session_state.config_profiles.keys())

# ==============================================================================
# 7. GŁÓWNY SILNIK RENDERUJĄCY
# ==============================================================================

class RenderEngine:
    """Koordynuje cały proces renderowania filmów."""
    
    def __init__(self, job: RenderJob, output_dir: str = "output"):
        self.job = job
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.temp_files = []
    
    def render(self, progress_callback=None) -> str:
        """Renderuje pojedynczy film, zwraca ścieżkę do pliku."""
        try:
            cfg = self.job.video_config
            style = self.job.text_style
            target_duration = random.uniform(cfg.target_duration_min, cfg.target_duration_max)
            
            # Oblicz liczbę zdjęć
            cover_duration = cfg.photo_duration * cfg.cover_duration_multiplier
            remaining = target_duration - cover_duration
            num_photos = max(1, int(remaining / cfg.photo_duration))
            
            # Wybór zdjęć
            photos = self.job.photo_files
            if cfg.randomize_photo_order:
                if cfg.repeat_photos_if_needed and len(photos) < num_photos:
                    selected_photos = random.choices(photos, k=num_photos)
                else:
                    selected_photos = random.sample(photos, min(num_photos, len(photos)))
            else:
                # Sekwencyjnie z cyklem
                selected_photos = [photos[i % len(photos)] for i in range(num_photos)]
            
            # Tworzenie klipów
            clips = []
            
            # Okładka
            cover_img = ImageProcessor.process_image(
                self.job.cover_file,
                target_res=cfg.resolution,
                mode="cover"
            )
            cover_clip = ImageClip(cover_img).set_duration(cover_duration)
            clips.append(cover_clip)
            
            # Zdjęcia
            for i, photo in enumerate(selected_photos):
                img = ImageProcessor.process_image(
                    photo,
                    target_res=cfg.resolution,
                    mode="cover",
                    apply_blur=False
                )
                clip = ImageClip(img).set_duration(cfg.photo_duration)
                clips.append(clip)
            
            # Konkatenacja z przejściami
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
            
            # Kompozyt
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
            
            final.write_videofile(
                out_path,
                fps=cfg.fps,
                codec=cfg.codec,
                audio_codec=cfg.audio_codec,
                preset=cfg.preset,
                bitrate=cfg.bitrate,
                threads=4,
                logger=None
            )
            
            # Czyszczenie
            final.close()
            final_clip.close()
            for clip in clips:
                clip.close()
            
            return out_path
            
        except Exception as e:
            logger.exception(f"Błąd renderowania: {e}")
            raise
    
    def _apply_transitions(self, clips: List[VideoClip], cfg: VideoConfig) -> VideoClip:
        """Stosuje przejścia między klipami."""
        if len(clips) <= 1:
            return clips[0]
        
        transition_map = {
            TransitionType.FADE: lambda c1, c2: c1.crossfadeout(cfg.transition_duration).crossfadein(cfg.transition_duration),
            TransitionType.SLIDE_LEFT: lambda c1, c2: self._slide_transition(c1, c2, 'left', cfg.transition_duration),
            TransitionType.SLIDE_RIGHT: lambda c1, c2: self._slide_transition(c1, c2, 'right', cfg.transition_duration),
            TransitionType.SLIDE_UP: lambda c1, c2: self._slide_transition(c1, c2, 'up', cfg.transition_duration),
            TransitionType.SLIDE_DOWN: lambda c1, c2: self._slide_transition(c1, c2, 'down', cfg.transition_duration),
        }
        
        func = transition_map.get(cfg.transition)
        if func is None:
            return concatenate_videoclips(clips, method="chain")
        
        # Sklejanie z przejściami
        from moviepy.video.compositing.transitions import slide_in, slide_out
        
        final_clips = [clips[0]]
        for i in range(1, len(clips)):
            # To wymaga bardziej zaawansowanej obsługi – uproszczone
            # W praktyce lepiej użyć CompositeVideoClip z przejściami
            # Dla uproszczenia zastosujemy zwykłą konkatenację
            # (w pełnej wersji można by zaimplementować)
            final_clips.append(clips[i])
        
        return concatenate_videoclips(final_clips, method="chain")
    
    def _slide_transition(self, clip1, clip2, direction, duration):
        """Tworzy przejście przesuwne (uproszczone)."""
        # W praktyce wymaga compositingu; na potrzeby przykładu zwracamy proste połączenie
        return concatenate_videoclips([clip1, clip2], method="chain")

# ==============================================================================
# 8. INTERFEJS UŻYTKOWNIKA – STREAMLIT
# ==============================================================================

def main():
    OmegaCore.setup_session()
    st.set_page_config(
        page_title=f"Ω OMEGA {OmegaCore.VERSION}",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Konfiguracja ImageMagick
    mpy_config.change_settings({"IMAGEMAGICK_BINARY": OmegaCore.get_magick_path()})
    
    # Inicjalizacja domyślnych stylów
    if 'current_text_style' not in st.session_state:
        st.session_state.current_text_style = TextStyleConfig()
    if 'current_video_config' not in st.session_state:
        st.session_state.current_video_config = VideoConfig()
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    
    # === SIDEBAR – KONFIGURACJA ===
    with st.sidebar:
        st.title("⚙️ OMEGA CONFIGURATOR")
        
        # Zarządzanie profilami
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
        
        # Zakładki konfiguracji
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Tekst", "🎬 Wideo", "🎵 Audio", "🖼️ Obrazy"])
        
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
            st.subheader("Czas trwania")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.current_video_config.target_duration_min = st.number_input(
                    "Min (s)", 3.0, 20.0, st.session_state.current_video_config.target_duration_min, 0.1
                )
            with col2:
                st.session_state.current_video_config.target_duration_max = st.number_input(
                    "Max (s)", 3.0, 20.0, st.session_state.current_video_config.target_duration_max, 0.1
                )
            
            st.session_state.current_video_config.photo_duration = st.slider(
                "Czas trwania zdjęcia (s)", 0.05, 1.0, st.session_state.current_video_config.photo_duration, 0.01
            )
            st.session_state.current_video_config.cover_duration_multiplier = st.slider(
                "Mnożnik czasu okładki", 1.0, 5.0, st.session_state.current_video_config.cover_duration_multiplier, 0.1
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
        
        with tab4:
            st.subheader("Przetwarzanie obrazów")
            img_mode = st.selectbox("Tryb skalowania", ["cover", "contain", "crop"])
            st.session_state.img_mode = img_mode  # To nie jest w VideoConfig, trzeba dodać
            # Można dodać więcej opcji
        
        st.divider()
        
        # Lista tekstów
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
        
        # Live preview
        with st.expander("👁️ LIVE PREVIEW", expanded=False):
            preview_text = st.text_input("Tekst testowy", "OMEGA PREVIEW")
            if preview_text:
                preview_img = TextEngine.render_text(
                    preview_text,
                    st.session_state.current_text_style,
                    resolution=OmegaCore.TARGET_RES
                )
                # Tło szare dla kontrastu
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
    
    # Dodatkowe opcje
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        chunk_size = st.number_input("Liczba filmów w paczce ZIP", min_value=1, max_value=100, value=20)
    with col_adv2:
        output_dir = st.text_input("Katalog wyjściowy", value="output")
    
    if st.button("🚀 ROZPOCZNIJ PRODUKCJĘ MASOWĄ", type="primary", use_container_width=True):
        if not u_c or not u_p:
            st.error("Wgraj co najmniej jedną okładkę i jedno zdjęcie!")
        else:
            # Przygotowanie katalogu tymczasowego
            st.session_state.temp_dir = tempfile.mkdtemp(prefix="omega_")
            st.session_state.v_results = []
            st.session_state.zip_files = []
            
            # Pasek postępu
            progress_bar = st.progress(0, text="Inicjalizacja...")
            status_text = st.empty()
            
            # Generowanie filmów
            total_jobs = len(u_c)
            video_paths = []
            
            for idx, cover_file in enumerate(u_c):
                status_text.text(f"Renderowanie filmu {idx+1} z {total_jobs}...")
                
                # Wybór losowego audio (jeśli dostępne)
                audio_file = random.choice(u_m) if u_m else None
                
                # Losowa liczba linii tekstu (1-3)
                num_lines = random.randint(1, 3)
                selected_texts = random.sample(texts_list, min(num_lines, len(texts_list)))
                
                # Utworzenie zadania
                job = RenderJob(
                    cover_file=cover_file,
                    photo_files=u_p,
                    audio_file=audio_file,
                    text_lines=selected_texts,
                    text_style=st.session_state.current_text_style,
                    video_config=st.session_state.current_video_config,
                    job_id=f"JOB{idx+1:03d}"
                )
                
                # Renderowanie
                engine = RenderEngine(job, output_dir=output_dir)
                try:
                    out_path = engine.render()
                    video_paths.append(out_path)
                    st.session_state.v_results.append(out_path)
                    st.session_state.temp_files.extend(engine.temp_files)
                except Exception as e:
                    st.error(f"Błąd renderowania filmu {idx+1}: {e}")
                    logger.exception("Błąd renderowania")
                
                # Aktualizacja paska postępu
                progress_bar.progress((idx + 1) / total_jobs)
            
            # Pakowanie do ZIP
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
    
    # Sekcja pobierania
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
        
        # Przycisk czyszczenia
        if st.button("🗑️ Wyczyść pliki tymczasowe i wyniki", use_container_width=True):
            # Usuń wygenerowane pliki
            for f in st.session_state.v_results:
                try:
                    os.remove(f)
                except:
                    pass
            for f in st.session_state.zip_files:
                try:
                    os.remove(f)
                except:
                    pass
            # Usuń pliki tymczasowe
            for f in st.session_state.temp_files:
                try:
                    os.remove(f)
                except:
                    pass
            OmegaCore.cleanup_temp()
            st.session_state.v_results = []
            st.session_state.zip_files = []
            st.session_state.temp_files = []
            st.rerun()

# ==============================================================================
# 9. WEJŚCIE APLIKACJI
# ==============================================================================

if __name__ == "__main__":
    main()
