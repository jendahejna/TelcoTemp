"""
Modul pro inicializaci aplikace: načtení konfigurace, připojení k databázi a přípravu geografických a výškových dat.

Obsahuje:
    - Funkci `wait_for_next_hour` pro pauzu do začátku další hodiny.
    - Funkci `initialize_app` pro vytvoření DB připojení s SSL, načtení metadat a geografických dat.
"""
import logging
from sqlalchemy import create_engine
from database_operations.database_operations import DatabaseOperations
from spatial_processing.geographical_processing import GeographicalProcessing
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from time import sleep

# Načtení proměnných prostředí pro připojení k databázi
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "ssl_ca": os.getenv("DB_SSL_CA"),
    "ssl_cert": os.getenv("DB_SSL_CERT"),
    "ssl_key": os.getenv("DB_SSL_KEY"),
}

CZECH_DATA_PATH = "country_data/czech_republic.json"
TIF_PATH = "country_data/elevation_data.tif"


def wait_for_next_hour():
    """
    Pozastaví běh programu až do začátku příští hodiny.

    Vypočítá čas zbývající do další celé hodiny a usne po tuto dobu.
    """
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sleep((next_hour - now).seconds)


def initialize_app(config, data_path, tif_path):
    """
    Inicializuje prostředí aplikace:
      1. Vytvoří SSL připojení k MySQL databázi pomocí SQLAlchemy.
      2. Ověří spojení a zaloguje výsledek.
      3. Vytvoří instance pro DB operace a geografické zpracování.
      4. Načte a převede GeoJSON s hranicemi státu.
      5. Načte rastrová data výšek a transformuje souřadnice.

    Parametry:
        config (dict): Konfigurace databázového připojení se záznamy 'user', 'password', 'host', 'port', 'ssl_ca', 'ssl_cert', 'ssl_key'.
        data_path (str): Cesta k souboru GeoJSON s hranicemi státu.
        tif_path (str): Cesta k GeoTIFF souboru s výškovými daty.

    Vrací:
        tuple: (db_ops, geo_proc, czech_rep, elevation_data, lon_elev, lat_elev)
            db_ops: instance DatabaseOperations pro DB operace
            geo_proc: instance GeographicalProcessing pro geoprosessing
            czech_rep: GeoDataFrame s hranicemi státu
            elevation_data: 2D numpy.ndarray výškových dat
            lon_elev, lat_elev: 2D numpy.ndarray souřadnic geoprostoru odpovídající elevation_data
    """
    engine = create_engine(
        f"mysql+mysqlconnector://"
        f"{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}?"
        f"ssl_ca={config['ssl_ca']}&"
        f"ssl_cert={config['ssl_cert']}&"
        f"ssl_key={config['ssl_key']}"
    )
    try:
        with engine.connect() as conn:
            logging.getLogger(__name__).info("Připojení k DB s SSL proběhlo v pořádku.")
    except Exception as e:
        logging.getLogger(__name__).error(f"SSL připojení selhalo: {e}")
        raise

    db_ops = DatabaseOperations(engine)
    geo_proc = GeographicalProcessing()

    state = geo_proc.load_country_data(data_path)
    czech_rep = geo_proc.json_to_geodataframe(state)
    elevation_data, lon_elev, lat_elev = geo_proc.load_elevation_data(tif_path)

    return db_ops, geo_proc, czech_rep, elevation_data, lon_elev, lat_elev
