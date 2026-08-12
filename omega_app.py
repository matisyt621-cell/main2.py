"""
OMEGA - interfejs Streamlit.
WYMAGA plików omega_engine.py i omega_worker.py w TYM SAMYM folderze.
Uruchamianie: streamlit run omega_app.py

Ten plik NIE renderuje już filmów sam - tylko przygotowuje zadanie i
uruchamia omega_worker.py jako całkowicie niezależny proces w tle (Opcja 2).
Dzięki temu można wygenerować bardzo dużą liczbę filmów bez blokowania UI
i bez ryzyka przerwania produkcji przez zamknięcie karty przeglądarki.
"""

import streamlit as st
import os, sys, time, json, random, subprocess, shutil
from PIL import Image, ImageDraw

from omega_engine import OmegaCore, get_font_path, draw_text_pancerny, _save_uploads

# ==============================================================================
# STAŁE / ŚCIEŻKI
# ==============================================================================
PACK_SIZE = 20  # NA ŻYCZENIE: zawsze 20 filmów w paczce ZIP (wcześniej 70, nie do zmiany w UI)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKER_SCRIPT = os.path.join(_THIS_DIR, "omega_worker.py")
_JOBS_DIR = os.path.join(_THIS_DIR, "temp", "jobs")
_LAST_JOB_POINTER = os.path.join(_JOBS_DIR, "last_job.json")
os.makedirs(_JOBS_DIR, exist_ok=True)


def _launch_worker(config_path):
    """Uruchamia render jako OSOBNY, w pełni niezależny proces (Opcja 2)."""
    cmd = [sys.executable, _WORKER_SCRIPT, "--config", config_path]
    log_path = config_path.replace(".json", ".log")
    log_file = open(log_path, "w")
    kwargs = {"cwd": _THIS_DIR, "stdout": log_file, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


def _read_status(status_path):
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ==============================================================================
# KONFIGURACJA STRONY (musi być pierwszą komendą Streamlit)
# ==============================================================================
st.set_page_config(page_title="Ω OMEGA V14.0", layout="wide")

# ==============================================================================
# SESJA - odzyskiwanie ostatniego zadania po odświeżeniu strony/nowej sesji
# ==============================================================================
if 'zip_files' not in st.session_state:
    st.session_state.zip_files = []

if 'active_status_path' not in st.session_state:
    st.session_state.active_status_path = None
    st.session_state.active_output_dir = None
    if os.path.exists(_LAST_JOB_POINTER):
        try:
            with open(_LAST_JOB_POINTER, "r", encoding="utf-8") as f:
                pointer = json.load(f)
            if os.path.exists(pointer.get("status_path", "")):
                st.session_state.active_status_path = pointer["status_path"]
                st.session_state.active_output_dir = pointer.get("output_dir")
        except Exception:
            pass

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

    cpu_count = os.cpu_count() or 4
    parallel_workers = st.number_input(
        "⚡ Równoległe renderowania (procesy)",
        min_value=1, max_value=max(1, cpu_count),
        value=max(1, min(4, cpu_count)),
        step=1,
        help=f"Twoja maszyna ma {cpu_count} rdzeni CPU - to jest górny limit tego pola."
    )

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

# ==============================================================================
# GŁÓWNY WIDOK
# ==============================================================================

st.title(f"Ω OMEGA {OmegaCore.VERSION}")

# --- PANEL POSTĘPU (jeśli jest aktywne/ostatnie zadanie w tle) ---
if st.session_state.active_status_path:
    status = _read_status(st.session_state.active_status_path)
    if status:
        st.divider()
        st.subheader("📡 Status renderowania w tle")
        total = max(status.get("total", 0), 1)
        done = status.get("done", 0)
        st.progress(min(done / total, 1.0), text=f"{done}/{status.get('total', 0)} filmów gotowych")

        if status.get("errors"):
            with st.expander(f"⚠️ Błędy ({len(status['errors'])})"):
                for e in status["errors"]:
                    st.write(e)

        if status.get("state") == "done":
            st.success("✅ Produkcja i pakowanie zakończone!")
            st.session_state.zip_files = status.get("zip_files", [])
        else:
            st.caption("Renderowanie działa w tle jako niezależny proces — możesz zamknąć tę kartę, produkcja się nie zatrzyma. Strona odświeży się automatycznie.")
            time.sleep(3)
            st.rerun()
    st.divider()

# --- UPLOAD ---
c1, c2, c3 = st.columns(3)
with c1: u_c = st.file_uploader("Okładki", type=['png','jpg','jpeg'], accept_multiple_files=True)
with c2: u_p = st.file_uploader("Zdjęcia (Bulk)", type=['png','jpg','jpeg'], accept_multiple_files=True)
with c3: u_m = st.file_uploader("Muzyka (MP3)", type=['mp3'], accept_multiple_files=True)

if st.button("🚀 URUCHOM PRODUKCJĘ MASOWĄ (w tle)", use_container_width=True):
    if not u_c or not u_p:
        st.error("Wgraj okładki i zdjęcia!")
    else:
        with st.status("🎬 Przygotowywanie zadania...", expanded=True) as prep_status:
            st.write("💾 Zapisywanie plików (duże zdjęcia >1MB są automatycznie kompresowane)...")
            cover_paths = _save_uploads(u_c, "covers", base_dir=os.path.join(_THIS_DIR, "temp", "uploads"), compress_images=True)
            photo_paths = _save_uploads(u_p, "photos", base_dir=os.path.join(_THIS_DIR, "temp", "uploads"), compress_images=True)
            music_paths = _save_uploads(u_m, "music", base_dir=os.path.join(_THIS_DIR, "temp", "uploads")) if u_m else []

            threads_per_job = max(1, cpu_count // int(parallel_workers))

            job_id = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(_THIS_DIR, "temp", "outputs", job_id)
            os.makedirs(output_dir, exist_ok=True)

            # --- Przygotowanie parametrów WSZYSTKICH zadań (logika BEZ ZMIAN) ---
            jobs = []
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

                jobs.append({
                    'idx': idx,
                    'cover_path': cover_path,
                    'photo_paths': sample_paths,
                    'music_path': music_path,
                    'text': text_choice,
                    'cfg': cfg,
                    'cov_dur': cov_dur,
                    'photo_dur': current_speed,
                    'res_mod': list(res_mod),
                    'fps_mod': fps_mod,
                    'v_bitrate_mod': v_bitrate_mod,
                    'a_bitrate_mod': a_bitrate_mod,
                    'bright_mod': bright_mod,
                    'gamma_mod': gamma_mod,
                    'out_name': os.path.join(output_dir, f"OMEGA_VIDEO_{idx+1}.mp4"),
                    'threads': threads_per_job,
                })

            st.write(f"📝 Przygotowano {len(jobs)} zadań renderowania.")

            # --- Zapis konfiguracji zadania + wystartowanie NIEZALEŻNEGO procesu ---
            status_path = os.path.join(_JOBS_DIR, f"{job_id}_status.json")
            config_path = os.path.join(_JOBS_DIR, f"{job_id}_config.json")

            job_config = {
                "jobs": jobs,
                "pack_size": PACK_SIZE,
                "parallel_workers": int(parallel_workers),
                "status_path": status_path,
                "output_dir": output_dir,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(job_config, f)
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({"total": len(jobs), "done": 0, "results": [], "errors": [], "state": "starting", "zip_files": []}, f)

            _launch_worker(config_path)

            with open(_LAST_JOB_POINTER, "w", encoding="utf-8") as f:
                json.dump({"status_path": status_path, "output_dir": output_dir}, f)

            st.session_state.active_status_path = status_path
            st.session_state.active_output_dir = output_dir

            prep_status.update(label="✅ Zadanie wystartowało w tle! Renderowanie trwa niezależnie od tej karty.", state="complete")

        st.rerun()

# ==============================================================================
# SEKCJA POBIERANIA
# ==============================================================================
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
                    file_name=os.path.basename(zip_path),
                    use_container_width=True,
                    key=f"dl_{idx}"
                )

    if st.button("🗑️ WYCZYŚĆ SERWER (Usuń pliki)"):
        output_dir = st.session_state.get('active_output_dir')
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        st.session_state.zip_files = []
        st.session_state.active_status_path = None
        st.session_state.active_output_dir = None
        if os.path.exists(_LAST_JOB_POINTER):
            os.remove(_LAST_JOB_POINTER)
        st.rerun()
