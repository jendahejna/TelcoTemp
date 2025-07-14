"""
Modul hlavní aplikace: konfigurace Flask serveru pro obsluhu obrázků a řízení cyklického zpracování dat.
"""
import os
import threading
from werkzeug.utils import secure_filename
from flask import Flask, request, send_from_directory, abort
from dotenv import load_dotenv
from log import setup_logger
from initialization import (
    initialize_app,
    DB_CONFIG,
    CZECH_DATA_PATH,
    wait_for_next_hour,
    TIF_PATH,
)
from data_processing.data_processing import process_data_round

# Načtení proměnných prostředí
load_dotenv()

# Konfigurace z prostředí
FRONTEND_IP = os.getenv("FRONTEND_IP")
IMAGE_DIR   = os.getenv("IMAGE_DIR", os.path.join(os.getcwd(), "images"))
ALLOWED_EXT = set(os.getenv("ALLOWED_EXT", "png").split(","))
FLASK_HOST  = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT  = int(os.getenv("FLASK_PORT", "5000"))

# Nastavení loggerů
backend_logger = setup_logger("backend_logger", "app.log")
comm_logger    = setup_logger("comm_logger",    "http.log")

app = Flask(__name__)


@app.before_request
def restrict_ip():
    """
    Omezí přístup k vybraným endpointům pouze na specifikovanou IP adresu.
    Pokud je požadavek na '/images/' nebo '/test-directory' z jiné IP než FRONTEND_IP, vrátí 403.
    """
    if request.endpoint in ("get_image", "test_directory"):
        if request.remote_addr != FRONTEND_IP:
            abort(403)


@app.before_request
def log_request():
    """
    Zaznamená příchozí HTTP požadavky do logu.
    """
    comm_logger.info(f"{request.method} {request.url} from {request.remote_addr}")


@app.after_request
def log_response(response):
    """
    Zaznamená odchozí HTTP odpovědi do logu.
    """
    comm_logger.info(f"{response.status_code} to {request.remote_addr} for {request.url}")
    return response


def is_allowed_file(filename: str) -> bool:
    """
    Ověří, zda má soubor povolenou příponu podle ALLOWED_EXT.

    Vrací True, pokud přípona souboru (část za poslední tečkou) patří mezi povolené.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


@app.route("/images/<path:filename>")
def get_image(filename):
    """
    Poskytne soubor z adresáře IMAGE_DIR.
    Pokud soubor neexistuje nebo přípona není povolená, vrátí odpovídající chyba (403/404).
    """
    filename = secure_filename(filename)
    if not is_allowed_file(filename):
        abort(403)
    try:
        return send_from_directory(IMAGE_DIR, filename)
    except FileNotFoundError:
        abort(404)


@app.route("/test-directory")
def test_directory():
    """
    Vrátí seznam názvů souborů v adresáři IMAGE_DIR jako text.
    Pokud nastane jakákoliv chyba, vrátí 500.
    """
    try:
        return str(os.listdir(IMAGE_DIR))
    except Exception:
        abort(500)


def run_flask_app():
    """
    Spustí Flask HTTP server na definovaném hostu a portu.
    """
    app.run(host=FLASK_HOST, port=FLASK_PORT)


def data_processing_loop():
    """
    Neustále provádí zpracování dat v hodinových cyklech.

    1. Inicializuje aplikaci (DB, geodata, elevace).
    2. V nekonečné smyčce zavolá process_data_round a poté počká na začátek další hodiny.
    """
    db_ops, geo_proc, czech_rep, elevation_data, lon_elev, lat_elev = initialize_app(
        DB_CONFIG, CZECH_DATA_PATH, TIF_PATH
    )
    while True:
        process_data_round(db_ops, geo_proc, czech_rep, elevation_data, lon_elev, lat_elev)
        wait_for_next_hour()


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    backend_logger.info("Backend processing started")
    data_processing_loop()
