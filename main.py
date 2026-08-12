import streamlit as st
import os, gc, random, io, zipfile, hashlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from moviepy.editor import ImageClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import moviepy.config as mpy_config

try:
    import imageio_ffmpeg
    mpy_config.change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
except:
    pass

# ==============================================================================
# 1. KONFIGURACJA RDZENIA
# ==============================================================================

class OmegaCore:
    VERSION = "V15.0 STABILNA (STREAMLIT CLOUD)"
    TARGET_RES = (1080, 1920)
    SAFE_MARGIN = 90

    @staticmethod
    def setup_session():
        keys = ['v_results', 'zip_files']
        for key in keys:
            if key not in st.session_state:
                st.session_state[key] = []

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


# --- CACHE ZDEKODOWANYCH ZDJĘĆ ---
# Limit celowo NISKI (20, nie 300 jak w wersji pod VPS) - Streamlit Cloud ma
# ~1GB RAM na CAŁĄ appkę, więc duży cache obrazów w pamięci mógłby sam w sobie
# wywołać crash z braku pamięci.
_DECODE_CACHE = {}
_DECODE_CACHE_LIMIT = 20


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
    """BEZ ZMIAN logiki skalowania/kadrowania."""
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
    """Silnik Auto-Scale - BEZ ŻADNYCH ZMIAN."""
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


# --- KOMPRESJA DUŻYCH ZDJĘĆ (>1MB) - zostaje, pomaga też przy RAM ---
_MAX_STORED_WIDTH = 1600
_COMPRESS_THRESHOLD_BYTES = 1024 * 1024


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
        return content


def _save_uploads(files, subdir, base_dir="temp/uploads", compress_images=False):
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
                ext = ".jpg"
        path = os.path.join(dir_path, f"{h}{ext}")
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(stored_content)
        paths.append(path)
    return paths


def _render_one_video(cover_path, photo_paths, music_path, text, cfg,
                       cov_dur, photo_dur, res_mod, fps_mod,
                       v_bitrate_mod, a_bitrate_mod, bright_mod, gamma_mod, out_path):
    """
    Renderuje JEDEN film - wywoływane bezpośrednio w pętli, sekwencyjnie.
    ŚWIADOMIE bez ProcessPoolExecutor/subprocess: na darmowym Streamlit Cloud
    wieloprocesowość robiła więcej szkody (crash) niż pożytku (szybkość).
    Logika samego renderowania identyczna jak we wcześniejszych wersjach.
    """
    cov_arr = process_image_916(cover_path, res_mod)
    cov_arr = apply_image_adjustments(cov_arr, bright_mod, gamma_mod)
    clips = [ImageClip(cov_arr).set_duration(cov_dur)]

    for p in photo_paths:
        img_arr = process_image_916(p, res_mod)
        img_arr = apply_image_adjustments(img_arr, bright_mod, gamma_mod)
        clips.append(ImageClip(img_arr).set_duration(photo_dur))

    base = concatenate_videoclips(clips, method="chain")

    t_arr = np.array(draw_text_pancerny(text, cfg, res=res_mod))
    txt_clip = ImageClip(t_arr).set_duration(base.duration)

    final = CompositeVideoClip([base, txt_clip], size=res_mod)

    if music_path:
        aud = AudioFileClip(music_path)
        final = final.set_audio(aud.subclip(0, min(aud.duration, final.duration)))

    final.write_videofile(
        out_path,
        fps=fps_mod,
        codec="libx264",
        audio_codec="aac",
        bitrate=None if v_bitrate_mod is None else f"{v_bitrate_mod}k",
        audio_bitrate=None if a_bitrate_mod is None else f"{a_bitrate_mod}k",
        threads=2,
        logger=None,
        preset="ultrafast"
    )

    final.close()
    base.close()
    for c in clips:
        c.close()
    gc.collect()
    return out_path


# ==============================================================================
# 2. INTERFEJS
# ==============================================================================

OmegaCore.setup_session()
st.set_page_config(page_title="Ω OMEGA V15.0", layout="wide")
mpy_config.change_settings({"IMAGEMAGICK_BINARY": OmegaCore.get_magick_path()})

PACK_SIZE = 20  # zawsze 20 filmów w paczce ZIP

with st.sidebar:
    st.title("⚙️ KONFIGURACJA")

    f_font = st.selectbox("Czcionka", ["League Gothic Regular", "League Gothic Condensed", "Impact"])
    f_size = st.slider("Max Wielkość", 20, 500, 83)
    t_color = st.color_picker("Kolor tekstu", "#FFFFFF")
    s_width = st.slider("Obrys", 0, 20, 3)

    st.header("🌑 CIEŃ")
    shd_x = st.slider("Cień X", -100, 100, 15)
    shd_y = st.slider("Cień Y", -100, 100, 15)
    shd_alpha = st.slider("Alpha", 0, 255, 200)

    cfg = {
        'font_path': get_font_path(f_font), 'f_size': f_size, 't_color': t_color,
        's_width': s_width, 's_color': "#000000", 'shd_x': shd_x, 'shd_y': shd_y,
        'shd_blur': 8, 'shd_alpha': shd_alpha, 'shd_color': "#000000"
    }

    st.header("👁️ LIVE PREVIEW")
    sim_bg = Image.new("RGB", OmegaCore.TARGET_RES, (15, 15, 15))
    draw_sim = ImageDraw.Draw(sim_bg)
    draw_sim.rectangle([0, 625, 1080, 1295], fill=(0, 255, 0))
    t_lay = draw_text_pancerny("LIVE PREVIEW TEST", cfg)
    sim_bg.paste(t_lay, (0, 0), t_lay)
    st.image(sim_bg, caption="Podgląd reaguje na suwaki!", use_container_width=True)

    st.divider()
    st.subheader("🎬 PRODUKCJA")
    speed_options = st.multiselect(
        "🎞️ Dozwolone szybkości przejść (s)",
        options=[0.1, 0.11, 0.12, 0.15, 0.2, 0.25, 0.3],
        default=[0.1, 0.12, 0.15, 0.2]
    )
    if not speed_options:
        speed_options = [0.1, 0.12, 0.15, 0.2]

    st.caption(f"📦 Filmy pakowane po {PACK_SIZE} w ZIP (stała wartość)")
    st.caption("⚡ Renderowanie sekwencyjne (1 film na raz) - stabilne na darmowym Streamlit Cloud. Jeśli crashuje, spróbuj wybrać wyżej wartości szybkości powyżej (mniej zdjęć = mniej RAM na film).")

    st.divider()
    with st.expander("🛡️ ANTY-TIKTOK (opcje)"):
        enable_anti = st.checkbox("Włącz techniki anty-detekcyjne", value=False)

        col1, col2 = st.columns(2)
        with col1:
            res_shift = st.checkbox("Zmiana rozdzielczości o 2px", value=True)
            fps_random = st.checkbox("Losowy FPS (np. 29.97)", value=True)
            video_bitrate_random = st.checkbox("Losowy bitrate wideo", value=True)
        with col2:
            audio_bitrate_random = st.checkbox("Losowy bitrate audio", value=True)
            brightness_tweak = st.checkbox("Modyfikacja jasności (+/-1%)", value=True)
            gamma_tweak = st.checkbox("Modyfikacja gamma", value=True)

        default_fps = st.selectbox("Bazowe FPS", [24, 30, 60], index=1)
        default_video_bitrate = st.number_input("Bazowy bitrate wideo (kb/s)", value=5000, step=100)
        default_audio_bitrate = st.number_input("Bazowy bitrate audio (kb/s)", value=192, step=16)

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
    raw_texts = st.text_area("Baza Tekstów", default_txts, height=150)
    texts_list = [t.strip() for t in raw_texts.split('\n') if t.strip()]

st.title(f"Ω OMEGA {OmegaCore.VERSION}")

c1, c2, c3 = st.columns(3)
with c1: u_c = st.file_uploader("Okładki", type=['png','jpg','jpeg'], accept_multiple_files=True)
with c2: u_p = st.file_uploader("Zdjęcia (Bulk)", type=['png','jpg','jpeg'], accept_multiple_files=True)
with c3: u_m = st.file_uploader("Muzyka (MP3)", type=['mp3'], accept_multiple_files=True)

if st.button("🚀 URUCHOM PRODUKCJĘ", use_container_width=True):
    if not u_c or not u_p:
        st.error("Wgraj okładki i zdjęcia!")
    else:
        st.session_state.v_results = []
        st.session_state.zip_files = []
        with st.status("🎬 Renderowanie...", expanded=True) as status:
            os.makedirs("temp", exist_ok=True)
            os.makedirs("output", exist_ok=True)

            st.write("💾 Zapisywanie plików (duże zdjęcia >1MB kompresowane automatycznie)...")
            cover_paths = _save_uploads(u_c, "covers", compress_images=True)
            photo_paths = _save_uploads(u_p, "photos", compress_images=True)
            music_paths = _save_uploads(u_m, "music") if u_m else []

            video_paths = []
            for idx, cover_path in enumerate(cover_paths):
                current_speed = random.choice(speed_options)
                target_dur = random.uniform(8.5, 9.8)
                cov_dur = current_speed * 3
                num_photos = int((target_dur - cov_dur) / current_speed)
                if num_photos < 1:
                    num_photos = 1

                res_mod = OmegaCore.TARGET_RES
                fps_mod = 24
                v_bitrate_mod = None
                a_bitrate_mod = None
                bright_mod = 1.0
                gamma_mod = 1.0

                if enable_anti:
                    if res_shift:
                        res_mod = (res_mod[0] + random.choice([-2, 0, 2]),
                                   res_mod[1] + random.choice([-2, 0, 2]))
                    fps_mod = default_fps
                    if fps_random:
                        if default_fps == 30:
                            fps_mod = random.uniform(29.97, 30.03)
                        elif default_fps == 60:
                            fps_mod = random.uniform(59.94, 60.06)
                        else:
                            fps_mod = default_fps + random.uniform(-0.05, 0.05)
                    if video_bitrate_random:
                        v_bitrate_mod = int(default_video_bitrate * random.uniform(0.98, 1.02))
                    if audio_bitrate_random:
                        a_bitrate_mod = int(default_audio_bitrate * random.uniform(0.98, 1.02))
                    if brightness_tweak:
                        bright_mod = random.uniform(0.99, 1.01)
                    if gamma_tweak:
                        gamma_mod = random.uniform(0.99, 1.01)
                else:
                    fps_mod = 24

                sample_paths = random.sample(photo_paths, min(num_photos, len(photo_paths)))
                music_path = random.choice(music_paths) if music_paths else None
                text_choice = random.choice(texts_list)
                out_path = os.path.join("output", f"OMEGA_VIDEO_{idx+1}.mp4")

                st.write(f"🎞️ Renderowanie filmu {idx+1}/{len(cover_paths)}...")
                try:
                    result = _render_one_video(
                        cover_path, sample_paths, music_path, text_choice, cfg,
                        cov_dur, current_speed, res_mod, fps_mod,
                        v_bitrate_mod, a_bitrate_mod, bright_mod, gamma_mod, out_path
                    )
                    video_paths.append(result)
                    st.session_state.v_results.append(result)
                    st.write(f"✅ Gotowe: film {idx+1}/{len(cover_paths)}")
                except Exception as e:
                    st.write(f"❌ Błąd filmu {idx+1}: {e}")

            st.write(f"📦 Pakowanie po {PACK_SIZE} filmów...")
            for i in range(0, len(video_paths), PACK_SIZE):
                chunk = video_paths[i:i + PACK_SIZE]
                part_num = i // PACK_SIZE + 1
                zip_name = f"OMEGA_PART_{part_num}.zip"
                with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_STORED) as z:
                    for v in chunk:
                        if os.path.exists(v):
                            z.write(v, arcname=os.path.basename(v))
                st.session_state.zip_files.append(zip_name)

            status.update(label="✅ PRODUKCJA ZAKOŃCZONA!", state="complete")

if st.session_state.zip_files:
    st.divider()
    st.subheader(f"📥 Gotowe paczki (po {PACK_SIZE} filmów):")
    cols = st.columns(len(st.session_state.zip_files))
    for idx, zip_path in enumerate(st.session_state.zip_files):
        if os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                cols[idx].download_button(
                    label=f"📂 Pobierz PART {idx+1}",
                    data=f,
                    file_name=zip_path,
                    use_container_width=True,
                    key=f"dl_{idx}"
                )

    if st.button("🗑️ WYCZYŚĆ SERWER (Usuń pliki)"):
        for f in st.session_state.v_results + st.session_state.zip_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        st.session_state.v_results = []
        st.session_state.zip_files = []
        st.rerun()
