import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import json
import numpy as np
from sqlalchemy.exc import IntegrityError
import os
from dotenv import load_dotenv

backend_logger = logging.getLogger('backend_logger')
load_dotenv()


class DatabaseOperations:
    """
    Třída pro operace s databází: získávání metadat pro mikrovlnné spoje a ukládání výsledků interpolace.
    """

    def __init__(self, engine):
        """
        Inicializuje instanci DatabaseOperations.

        Parametry:
            engine: SQLAlchemy Engine pro připojení k databázi.
        """
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine)

    def get_metadata(self, df):
        """
        Načte metadata (souřadnice, azimuth, link_id) pro každý záznam ve vstupním DataFrame.

        Parametry:
            df (pandas.DataFrame): DataFrame obsahující sloupec 'IP' s adresou senzoru.

        Postup:
            1. Pro každý řádek v df získá odpovídající záznam z tabulky cml_metadata.links.
            2. Určí správný azimuth a site_id na základě IP adresy.
            3. Z tabulky cml_metadata.sites načte zeměpisné souřadnice (X_coordinate, Y_coordinate).
            4. Vytvoří seznamy latitudes, longitudes, azimuths a links.

        Vrací:
            latitudes (list): seznam zeměpisných šířek každého senzoru.
            longitudes (list): seznam zeměpisných délek každého senzoru.
            azimuths (list): seznam azimutů každého senzoru.
            links (list): seznam ID spojení (link_id) pro každý senzor.
        """
        sites = []
        azimuths = []
        latitudes = []
        longitudes = []
        links = []
        devices = 0
        with self.Session() as session:
            for index, row in df.iterrows():
                ip_address = row["IP"].strip()

                try:
                    result = session.execute(
                        text(
                            "SELECT ID, site_A, site_B, azimuth_A, azimuth_B, "
                            "IP_address_A, IP_address_B FROM cml_metadata.links "
                            "WHERE IP_address_A=:ip OR IP_address_B=:ip"
                        ),
                        {"ip": ip_address},
                    ).fetchone()

                    if result:
                        devices += 1
                        (
                            link_id,
                            site_a,
                            site_b,
                            azimuth_a,
                            azimuth_b,
                            ip_address_a,
                            ip_address_b,
                        ) = result

                        if ip_address == ip_address_a:
                            azimuth = azimuth_a
                            site_id = site_a
                        elif ip_address == ip_address_b:
                            azimuth = azimuth_b
                            site_id = site_b
                        else:
                            azimuth, site_id = None, None
                            backend_logger.warning(
                                f"Azimuth and Site_ID cannot be assigned for IP: {ip_address}"
                            )

                        site_result = session.execute(
                            text(
                                "SELECT X_coordinate, Y_coordinate FROM cml_metadata.sites "
                                "WHERE id=:site_id"
                            ),
                            {"site_id": site_id},
                        ).fetchone()

                        if site_result:
                            longitude, latitude = site_result
                        else:
                            latitude, longitude = None, None
                            backend_logger.warning(
                                f"No coordinates found for site ID: {site_id}"
                            )
                    else:
                        # Pokud není nalezeno žádné metadata, přeskočíme
                        link_id = None
                        site_id = None
                        azimuth = None
                        latitude = None
                        longitude = None

                    azimuths.append(azimuth)
                    sites.append(site_id)
                    links.append(link_id)
                    latitudes.append(latitude)
                    longitudes.append(longitude)

                except Exception as e:
                    backend_logger.error(f"Error in get_metadata for IP {ip_address}: {e}")
                    continue

        backend_logger.info(f"Completed get_metadata method for {devices} devices.")
        return latitudes, longitudes, azimuths, links

    def realtime_writer(self, image_name, unique_links_list, current_datetime, grid_z):
        """
        Zapíše parametry interpolované teplotní mřížky do tabulky realtime_temperature_grids.

        Parametry:
            image_name (str): název souboru s obrázkem mřížky.
            unique_links_list (list[int]): seznam ID linek, pro které byla data vypočtena.
            current_datetime (datetime): čas provedení výpočtu.
            grid_z (numpy.ndarray): matice predikovaných teplot.
        """
        unique_links_list = [int(link) for link in unique_links_list]
        TEMP_MIN = round(np.nanmin(grid_z.ravel()))
        TEMP_MAX = round(np.nanmax(grid_z.ravel()))

        with self.Session() as session:
            try:
                session.execute(
                    text(
                        """
                        INSERT INTO telcorain_output.realtime_temperature_grids 
                        (time, links, image_name, TEMP_MIN, TEMP_MAX) 
                        VALUES (:time, :links, :image_name, :TEMP_MIN, :TEMP_MAX)
                        """
                    ),
                    {
                        "time": current_datetime,
                        "links": json.dumps(unique_links_list),
                        "image_name": image_name,
                        "TEMP_MIN": TEMP_MIN,
                        "TEMP_MAX": TEMP_MAX,
                    },
                )
                session.commit()
                backend_logger.info(
                    f"Interpolation data from {current_datetime} successfully recorded."
                )
            except IntegrityError as e:
                session.rollback()
                backend_logger.warning(f"Duplicate entry error: {e}")
            except Exception as e:
                session.rollback()
                backend_logger.error(f"Error in realtime_writer: {e}")

    def save_parameters(self, current_datetime, grid_x, grid_y):
        """
        Uloží parametry generované teplotní mřížky do tabulky realtime_temperature_parameters.

        Parametry:
            current_datetime (datetime): čas zahájení výpočtu.
            grid_x (numpy.ndarray): matice X souřadnic mřížky.
            grid_y (numpy.ndarray): matice Y souřadnic mřížky.
        """
        with self.Session() as session:
            try:
                X_MIN = round(np.nanmin(grid_x.ravel()), 4)
                X_MAX = round(np.nanmax(grid_x.ravel()), 4)
                Y_MIN = round(np.nanmin(grid_y.ravel()), 4)
                Y_MAX = round(np.nanmax(grid_y.ravel()), 4)
                retention = 43200
                timestep = 1800
                X_COUNT = Y_COUNT = 500
                images_URL = os.getenv("IMAGES_URL")

                session.execute(
                    text(
                        """
                        INSERT INTO telcorain_output.realtime_temperature_parameters
                        (started, retention, timestep, X_MIN, X_MAX, Y_MIN, Y_MAX, 
                         X_COUNT, Y_COUNT, images_URL)
                        VALUES (:started, :retention, :timestep, :X_MIN, :X_MAX, :Y_MIN, :Y_MAX, 
                                :X_COUNT, :Y_COUNT, :images_URL)
                        """
                    ),
                    {
                        "started": current_datetime,
                        "retention": retention,
                        "timestep": timestep,
                        "X_MIN": X_MIN,
                        "X_MAX": X_MAX,
                        "Y_MIN": Y_MIN,
                        "Y_MAX": Y_MAX,
                        "X_COUNT": X_COUNT,
                        "Y_COUNT": Y_COUNT,
                        "images_URL": images_URL,
                    },
                )
                session.commit()
                backend_logger.info(
                    f"Parameters data for {current_datetime} successfully recorded."
                )
            except Exception as e:
                session.rollback()
                backend_logger.error(f"Error in save_parameters: {e}")