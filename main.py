import streamlit as st
import os, gc, random, io, zipfile, hashlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from moviepy.editor import ImageClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import moviepy.config as mpy_config
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# 0. Automatyczne ustawienie ffmpeg (dzięki imageio-ffmpeg)
# ==============================================================================
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    mpy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
except:
    # Jeśli się nie uda, moviepy spróbuje użyć ffmpeg z systemu (jeśli istnieje)
    pass

# ==============================================================================
# 1. KONFIGURACJA RDZENIA OMEGA (z dodatkami)
# ==============================================================================

class OmegaCore:
    VERSION = "V13.1 ANTY-TIKTOK (SZYBKA PRODUKCJA)"
    TARGET_RES = (1080, 1920)
    SAFE_MARGIN = 90  # Margines boczny dla tekstu (Auto-Scale)

    @staticmethod
    def setup_session():
        # Inicjalizacja list
        keys = ['v_covers', 'v_photos', 'v_music', 'v_results', 'zip_files']
        for key in keys:
            if key not in st.session_state:
                st.session_state[key] = []
        # Dodatkowe ustawienia
        if 'pack_size' not in st.session_state:
            st.session_state.pack_size = 70  # domyślnie 70 filmów na paczkę

    @staticmethod
    def get_magick_path():
        if os.name == 'posix': return "/usr/bin/convert"
        return r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"


# ==============================================================================
# 2. SILNIK GRAFICZNY I AUTO-SCALE (logika bez zmian, zmieniony tylko input)
# ==============================================================================

def get_font_path(font_selection):
    font_files = {
        "League Gothic Regular": "LeagueGothic-Regular.otf",
        "League Gothic Condensed": "LeagueGothic-CondensedRegular.otf",
        "Impact": "impact.ttf"
    }
    target = font_files.get(font_selection)
    if target and os.path.exists(target): return os.path.abspath(target)
    return "arial.ttf"


# --- OPTYMALIZACJA 2: CACHE ZDEKODOWANYCH ZDJĘĆ ---
# Każdy proces roboczy (worker) ProcessPoolExecutor ma własny, oddzielny słownik
# w pamięci. Te same zdjęcia są losowane do wielu różnych filmów - bez cache
# każde użycie oznaczałoby ponowne wczytanie pliku z dysku, EXIF-transpose
# i konwersję RGB. Cache trzyma już zdekodowany obraz PIL (przed skalowaniem
# do docelowej rozdzielczości, bo ta może się różnić między filmami o ±2px
# przez opcję "Zmiana rozdzielczości o 2px").
_DECODE_CACHE = {}
_DECODE_CACHE_LIMIT = 300  # zabezpieczenie przed nieograniczonym wzrostem pamięci procesu


def _load_decoded_image(path):
    cached = _DECODE_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "rb") as f:
        file_bytes = f.read()
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if len(_DECODE_CACHE) >= _DECODE_CACHE_LIMIT:
        # Prosty mechanizm ograniczający rozmiar cache (usuwa najstarszy wpis)
        _DECODE_CACHE.pop(next(iter(_DECODE_CACHE)))
    _DECODE_CACHE[path] = img
    return img


def process_image_916(path, target_res=OmegaCore.TARGET_RES):
    """
    UWAGA: sygnatura zmieniona z file_obj (UploadedFile) na path (str) -
    wymagane, aby zadania dało się przekazywać do procesów roboczych
    (ProcessPoolExecutor nie potrafi zserializować obiektów UploadedFile).
    Logika skalowania/kadrowania obrazu pozostaje IDENTYCZNA jak wcześniej.
    """
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
    """Silnik Auto-Scale: Zmniejsza czcionkę, aby tekst nie wystawał poza marginesy. (BEZ ZMIAN)"""
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


# ==============================================================================
# 3. DODATKOWE FUNKCJE ANTY-DETEKCYJNE (BEZ ZMIAN)
# ==============================================================================

def apply_image_adjustments(img_array, brightness=1.0, gamma=1.0):
    """Modyfikuje jasność i gamma obrazu (tablica numpy)"""
    img = Image.fromarray(img_array)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    if gamma != 1.0:
        # szybka korekcja gamma przez LUT
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)


# ==============================================================================
# 3B. FUNKCJE POMOCNICZE DO ZAPISU UPLOADÓW I RENDEROWANIA RÓWNOLEGŁEGO
# ==============================================================================

def _save_uploads(files, subdir, base_dir="temp/uploads"):
    """
    OPTYMALIZACJA 3: Zapisuje wgrane pliki na dysk RAZ, kluczując nazwę pliku
    hashem MD5 zawartości. Dzięki temu:
    - identyczne pliki (np. ta sama piosenka wylosowana wielokrotnie) są
      zapisywane tylko raz,
    - do procesów roboczych przekazujemy proste, pickle'owalne ścieżki (str)
      zamiast niepickle'owalnych obiektów UploadedFile.
    """
    dir_path = os.path.join(base_dir, subdir)
    os.makedirs(dir_path, exist_ok=True)
    paths = []
    for f in files:
        content = f.getvalue()
        h = hashlib.md5(content).hexdigest()
        ext = os.path.splitext(f.name)[1] or ".bin"
        path = os.path.join(dir_path, f"{h}{ext}")
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(content)
        paths.append(path)
    return paths


def _render_single_video(job):
    """
    OPTYMALIZACJA 1: Renderowanie pojedynczego filmu - funkcja wywoływana
    równolegle w osobnych procesach przez ProcessPoolExecutor.
    Cała logika budowania klipów, tekstu i zapisu jest IDENTYCZNA jak w
    oryginalnej pętli sekwencyjnej - zmienia się tylko to, że wiele takich
    wywołań dzieje się jednocześnie na różnych rdzeniach CPU.
    """
    try:
        cfg = job['cfg']
        res_mod = job['res_mod']

        # Okładka
        cov_arr = process_image_916(job['cover_path'], res_mod)
        cov_arr = apply_image_adjustments(cov_arr, job['bright_mod'], job['gamma_mod'])
        clips = [ImageClip(cov_arr).set_duration(job['cov_dur'])]

        # Zdjęcia
        for p in job['photo_paths']:
            img_arr = process_image_916(p, res_mod)
            img_arr = apply_image_adjustments(img_arr, job['bright_mod'], job['gamma_mod'])
            clips.append(ImageClip(img_arr).set_duration(job['photo_dur']))

        base = concatenate_videoclips(clips, method="chain")

        # Tekst (auto-scale) - bez zmian
        t_arr = np.array(draw_text_pancerny(job['text'], cfg, res=res_mod))
        txt_clip = ImageClip(t_arr).set_duration(base.duration)

        final = CompositeVideoClip([base, txt_clip], size=res_mod)

        # Audio (plik już zapisany wcześniej na dysku - patrz _save_uploads)
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
            # OPTYMALIZACJA 4: liczba wątków ffmpeg na zadanie jest dzielona przez
            # liczbę równoległych renderów, żeby nie przeciążyć CPU wieloma
            # procesami walczącymi jednocześnie o te same rdzenie.
            threads=job['threads'],
            logger=None,
            preset="ultrafast"
        )

        # Sprzątanie pamięci - wykonywane wewnątrz procesu roboczego
        final.close()
        base.close()
        gc.collect()
        return out_name
    except Exception as e:
        return f"__ERROR__:{job.get('idx')}:{e}"


# ==============================================================================
# 4. INTERFEJS I LIVE PREVIEW (bez zmian w wyglądzie, +1 nowe pole)
# ==============================================================================

OmegaCore.setup_session()
st.set_page_config(page_title="Ω OMEGA V13.1", layout="wide")
mpy_config.change_settings({"IMAGEMAGICK_BINARY": OmegaCore.get_magick_path()})

with st.sidebar:
    st.title("⚙️ KONFIGURACJA")

    # --- PODSTAWOWE USTAWIENIA TEKSTU (bez zmian) ---
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
    # Symulacja 2.5x: Góra 625px, Dół 625px, Środek zielony
    draw_sim.rectangle([0, 625, 1080, 1295], fill=(0, 255, 0))

    t_lay = draw_text_pancerny("LIVE PREVIEW TEST", cfg)
    sim_bg.paste(t_lay, (0, 0), t_lay)
    st.image(sim_bg, caption="Podgląd reaguje na suwaki!", use_container_width=True)

    st.divider()

    # --- NOWE USTAWIENIA PRODUKCJI ---
    st.subheader("🎬 PRODUKCJA")
    # Wybór dozwolonych prędkości (multiselect)
    speed_options = st.multiselect(
        "🎞️ Dozwolone szybkości przejść (s)",
        options=[0.1, 0.11, 0.12, 0.15, 0.2, 0.25, 0.3],
        default=[0.1, 0.12, 0.15, 0.2]
    )
    if not speed_options:
        speed_options = [0.1, 0.12, 0.15, 0.2]  # zabezpieczenie

    # Rozmiar paczki ZIP
    pack_size = st.number_input(
        "📦 Filmy na paczkę ZIP",
        min_value=1, max_value=100,
        value=int(st.session_state.pack_size),
        step=1
    )
    st.session_state.pack_size = int(pack_size)

    # --- NOWE: liczba równoległych renderowań (OPTYMALIZACJA WYDAJNOŚCI) ---
    cpu_count = os.cpu_count() or 4
    parallel_workers = st.number_input(
        "⚡ Równoległe renderowania (procesy)",
        min_value=1, max_value=max(1, cpu_count),
        value=max(1, min(4, cpu_count)),
        step=1,
        help="Ile filmów renderować jednocześnie. Więcej = szybciej, ale mocniej obciąża CPU/RAM."
    )

    st.divider()

    # --- USTAWIENIA ANTY-DETEKCYJNE (opcjonalne) ---
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

        # Bazowe wartości (do modyfikacji)
        default_fps = st.selectbox("Bazowe FPS", [24, 30, 60], index=1)
        default_video_bitrate = st.number_input("Bazowy bitrate wideo (kb/s)", value=5000, step=100)
        default_audio_bitrate = st.number_input("Bazowy bitrate audio (kb/s)", value=192, step=16)

    st.divider()

    # --- BAZA TEKSTÓW (bez zmian) ---
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

# ==============================================================================
# 5. SILNIK PRODUKCJI (RÓWNOLEGŁY)
# ==============================================================================

st.title(f"Ω OMEGA {OmegaCore.VERSION}")

c1, c2, c3 = st.columns(3)
with c1: u_c = st.file_uploader("Okładki", type=['png','jpg','jpeg'], accept_multiple_files=True)
with c2: u_p = st.file_uploader("Zdjęcia (Bulk)", type=['png','jpg','jpeg'], accept_multiple_files=True)
with c3: u_m = st.file_uploader("Muzyka (MP3)", type=['mp3'], accept_multiple_files=True)

if st.button("🚀 URUCHOM PRODUKCJĘ MASOWĄ", use_container_width=True):
    if not u_c or not u_p:
        st.error("Wgraj okładki i zdjęcia!")
    else:
        st.session_state.v_results = []
        st.session_state.zip_files = []
        with st.status("🎬 Renderowanie...", expanded=True) as status:
            if not os.path.exists("temp"): os.makedirs("temp")

            # --- OPTYMALIZACJA 3: zapisujemy WSZYSTKIE uploady na dysk RAZ ---
            # (deduplikacja po hashu treści + brak wielokrotnego getvalue()/zapisu
            # tego samego pliku muzycznego w każdej iteracji jak wcześniej)
            st.write("💾 Zapisywanie wgranych plików (z deduplikacją)...")
            cover_paths = _save_uploads(u_c, "covers")
            photo_paths = _save_uploads(u_p, "photos")
            music_paths = _save_uploads(u_m, "music") if u_m else []

            # --- Przygotowanie parametrów WSZYSTKICH zadań ---
            # Losowanie zostaje w głównym wątku (sekwencyjnie), dokładnie tak
            # samo jak w oryginale - zmienia się tylko to, że właściwe
            # renderowanie (write_videofile) uruchamiamy potem równolegle.
            threads_per_job = max(1, cpu_count // int(parallel_workers))

            jobs = []
            for idx, cover_path in enumerate(cover_paths):
                current_speed = random.choice(speed_options)

                # --- TIME GUARD: długość filmu 8.5-9.8s (bez zmian) ---
                target_dur = random.uniform(8.5, 9.8)
                cov_dur = current_speed * 3
                num_photos = int((target_dur - cov_dur) / current_speed)
                if num_photos < 1:
                    num_photos = 1

                # --- Parametry anty-detekcyjne (bez zmian logiki) ---
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

                jobs.append({
                    'idx': idx,
                    'cover_path': cover_path,
                    'photo_paths': sample_paths,
                    'music_path': music_path,
                    'text': text_choice,
                    'cfg': cfg,
                    'cov_dur': cov_dur,
                    'photo_dur': current_speed,
                    'res_mod': res_mod,
                    'fps_mod': fps_mod,
                    'v_bitrate_mod': v_bitrate_mod,
                    'a_bitrate_mod': a_bitrate_mod,
                    'bright_mod': bright_mod,
                    'gamma_mod': gamma_mod,
                    'out_name': f"OMEGA_VIDEO_{idx+1}.mp4",
                    'threads': threads_per_job,
                })

                st.write(
                    f"🎞️ Przygotowano film {idx+1}/{len(cover_paths)} | "
                    f"Prędkość: {current_speed}s | Czas: {target_dur:.1f}s | Zdjęć: {num_photos}"
                )

            # --- OPTYMALIZACJA 1: RÓWNOLEGŁE RENDEROWANIE ---
            st.write(f"⚡ Renderowanie {len(jobs)} filmów równolegle ({parallel_workers} procesy naraz)...")
            done_count = 0
            with ProcessPoolExecutor(max_workers=int(parallel_workers)) as executor:
                future_to_idx = {executor.submit(_render_single_video, job): job['idx'] for job in jobs}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    done_count += 1
                    if isinstance(result, str) and result.startswith("__ERROR__"):
                        st.write(f"❌ Błąd filmu {idx+1}: {result}")
                    else:
                        st.session_state.v_results.append(result)
                        st.write(f"✅ Gotowe: {result} ({done_count}/{len(jobs)})")

            # --- Pakowanie według wybranego rozmiaru (bez zmian) ---
            st.write(f"📦 Dzielenie na paczki po {st.session_state.pack_size} filmów...")
            chunk_size = st.session_state.pack_size
            for i in range(0, len(st.session_state.v_results), chunk_size):
                chunk = st.session_state.v_results[i:i + chunk_size]
                part_num = (i // chunk_size) + 1
                zip_n = f"OMEGA_PART_{part_num}.zip"

                with zipfile.ZipFile(zip_n, 'w', compression=zipfile.ZIP_STORED) as z:
                    for f in chunk:
                        if os.path.exists(f): z.write(f)
                st.session_state.zip_files.append(zip_n)

            status.update(label="✅ PRODUKCJA I PAKOWANIE ZAKOŃCZONE!", state="complete")

# ==============================================================================
# 6. SEKCJA POBIERANIA (bez zmian)
# ==============================================================================
if st.session_state.zip_files:
    st.divider()
    st.subheader(f"📥 Gotowe paczki (po {st.session_state.pack_size} filmów):")
    cols = st.columns(len(st.session_state.zip_files))
    for idx, zip_path in enumerate(st.session_state.zip_files):
        with open(zip_path, "rb") as f:
            cols[idx].download_button(
                label=f"📂 Pobierz PART {idx+1}",
                data=f,
                file_name=zip_path,
                use_container_width=True
            )

    if st.button("🗑️ WYCZYŚĆ SERWER (Usuń pliki)"):
        for f in st.session_state.v_results + st.session_state.zip_files:
            if os.path.exists(f): os.remove(f)
        st.session_state.v_results = []
        st.session_state.zip_files = []
        st.rerun()
