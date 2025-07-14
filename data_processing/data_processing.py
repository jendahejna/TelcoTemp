import datetime
import gc
import logging
import traceback

import pandas as pd
from scipy.spatial import cKDTree

from data_processing.ml_modeling import temperature_predict
from database_operations.data_extraction import get_data
from interpolation.interpolation import spatial_interpolation
from spatial_processing.visualization import map_plotting

backend_logger = logging.getLogger("backend_logger")
first_run = True


def collect_data_summary(df):
    """
    Vytvoří souhrn dat z DataFrame pro potřeby vizualizace.

    Parametry:
        df: pandas.DataFrame obsahující sloupec 'Link_ID' a 'Time'.

    Vrací:
        unique_links_list: list unikátních ID linek.
        image_name: str, název obrázku ve formátu 'YYYY-MM-DD_HHMM.png'.
        image_time: datetime, čas zaokrouhlený nahoru na celou hodinu.
    """
    unique_links = df["Link_ID"].unique()
    unique_links_list = list(unique_links)

    image_time = pd.to_datetime(df["Time"].iloc[0]).ceil("h")
    image_hour = image_time.strftime("%Y-%m-%d_%H%M")
    image_name = f"{image_hour}.png"

    return unique_links_list, image_name, image_time


import pandas as pd
import numpy as np
from scipy.interpolate import griddata


def prepare_data(
        df,
        latitudes,
        longitudes,
        azimuths,
        links,
        elevation_data=None,
        lon_elev=None,
        lat_elev=None
):
    """
    Připraví a obohatí DataFrame o potřebné sloupce pro predikci teploty.

    Parametry:
        df: pandas.DataFrame s původními daty.
        latitudes: sekvence (list nebo array) zeměpisných šířek.
        longitudes: sekvence (list nebo array) zeměpisných délek.
        azimuths: sekvence (list nebo array) azimutů.
        links: sekvence (list) ID linek.
        elevation_data: 2D pole (array) s hodnotami nadmořské výšky.
        lon_elev: 2D pole (array) dlouhých dlaždic výškového modelu.
        lat_elev: 2D pole (array) širokých dlaždic výškového modelu.

    Vrací:
        pandas.DataFrame s přidanými sloupci:
          - 'Azimuth', 'Latitude', 'Longitude', 'Link_ID'
          - 'Time' převedený na časové pásmo Europe/Prague bez lokalizace
          - 'Hour' (hodina dne) a 'Day' (pořadové číslo dne v roce)
          - 'Elevation' (nadmořská výška) – pokud jsou k dispozici data
    """

    df["Azimuth"] = azimuths
    df["Latitude"] = latitudes
    df["Longitude"] = longitudes
    df["Link_ID"] = links

    df = df.dropna()

    df["Time"] = (
        pd.to_datetime(df["Time"], utc=True)
        .dt.tz_convert("Europe/Prague")
        .dt.tz_localize(None)
    )
    df["Hour"] = df["Time"].dt.hour
    df["Day"] = df["Time"].dt.dayofyear

    if elevation_data is not None and lon_elev is not None and lat_elev is not None:
        points = np.column_stack((lon_elev.ravel(), lat_elev.ravel()))
        values = elevation_data.ravel()
        df["Elevation"] = griddata(
            points,
            values,
            (df["Longitude"], df["Latitude"]),
            method="nearest"
        )
    else:
        print("Výšková data nebyla poskytnuta, sloupec 'Elevation' nebude přidán.")

    return df


def anomaly_detection(
        df: pd.DataFrame,
        z_threshold: float = 2.0,
        residual_threshold: float = 6.0,
        min_neighbors: int = 5,
        radius_km: float = 20,
        combine_mode: str = "union",
) -> pd.DataFrame:
    """
    Kombinuje detekci anomálií pomocí tří přístupů:
      1. Detekce na základě hodnoty Uptime.
      2. Globální detekce na základě z-skóre (teplota a elevace).
      3. Lokální detekce na základě rozdílu mezi teplotou senzoru a průměrnou teplotou okolních měření.

    Parametry:
      df: Vstupní DataFrame, který musí obsahovat sloupce
          'Temperature_MW', 'Elevation', 'Latitude', 'Longitude' (a případně 'Uptime').
      z_threshold: Prah pro z-skóre při globální detekci.
      residual_threshold: Prah pro rozdíl mezi teplotou a lokálním průměrem.
      min_neighbors: Minimální počet sousedů pro lokální vyhodnocení.
      radius_km: Poloměr v kilometrech pro hledání sousedních bodů.
      combine_mode: 'union' pro logické NEBO, 'intersection' pro logické A.

    Vrací:
      DataFrame bez anomálních záznamů.
    """
    removed = restarts = num_global_anomalies = total_local_anomalies = unevaluated = 0

    # 1. Filtrace podle Uptime
    if "Uptime" in df.columns and "IP" in df.columns:
        df = df.sort_values(["IP", "Time"])
        df["uptime_diff"] = df.groupby("IP")["Uptime"].diff()
        restarted = df.loc[df["uptime_diff"] < 0, "IP"].unique()
        before = len(df)
        df = df[~df["IP"].isin(restarted)].copy()
        removed = before - len(df)
        restarts = len(restarted)
        df.drop(columns=["uptime_diff"], inplace=True)

    # 2. Kontrola existence nutných sloupců
    for col in ["Temperature_MW", "Elevation", "Latitude", "Longitude"]:
        if col not in df.columns:
            raise KeyError(f"Chybí požadovaný sloupec '{col}' potřebný pro detekci anomálií.")

    # 3. Globální detekce anomálií pomocí z-skóre
    temp_mean = df["Temperature_MW"].mean()
    temp_std = df["Temperature_MW"].std()
    elev_mean = df["Elevation"].mean()
    elev_std = df["Elevation"].std()

    df["z_temp"] = (df["Temperature_MW"] - temp_mean) / temp_std
    df["z_elev"] = (df["Elevation"] - elev_mean) / elev_std

    global_anomaly_mask = (
                                  df["z_temp"].abs() > z_threshold
                          ) & (df["z_elev"].abs() <= z_threshold)
    num_global_anomalies = int(global_anomaly_mask.sum())

    # 4. Lokální detekce anomálií pomocí reziduí z interpolace
    coords = df[["Latitude", "Longitude"]].values
    tree = cKDTree(coords)
    local_flags = []

    for idx, (lat, lon, temp) in enumerate(df[["Latitude", "Longitude", "Temperature_MW"]].values):
        dists, indices = tree.query(
            [lat, lon], k=len(df), distance_upper_bound=radius_km / 111
        )
        neighbors = indices[(indices != idx) & (dists != float("inf"))]
        if len(neighbors) >= min_neighbors:
            local_avg = df.iloc[neighbors]["Temperature_MW"].mean()
            if abs(temp - local_avg) > residual_threshold:
                total_local_anomalies += 1
                local_flags.append(True)
            else:
                local_flags.append(False)
        else:
            unevaluated += 1
            local_flags.append(False)

    df["local_anomaly"] = local_flags

    # 5. Kombinace výsledků
    if combine_mode == "union":
        combined_mask = global_anomaly_mask | df["local_anomaly"]
    elif combine_mode == "intersection":
        combined_mask = global_anomaly_mask & df["local_anomaly"]
    else:
        raise ValueError("Neznámý režim kombinace. Použijte 'union' nebo 'intersection'.")

    total_combined = int(combined_mask.sum())

    # 6. Odstranění anomálií a dočasných sloupců
    df_clean = df[~combined_mask].copy()
    df_clean.drop(columns=["z_temp", "z_elev", "local_anomaly"], inplace=True)

    # 7. Logovací shrnutí
    backend_logger.info(
        "anomaly_detection_summary uptime_removed=%d restarts=%d "
        "global_anomalies=%d local_anomalies=%d unevaluated=%d "
        "combined_anomalies=%d combine_mode=%s",
        removed,
        restarts,
        num_global_anomalies,
        total_local_anomalies,
        unevaluated,
        total_combined,
        combine_mode,
    )

    return df_clean


def process_data_round(db_ops, geo_proc, czech_rep, elevation_data, lon_elev, lat_elev):
    """
    Provede jeden cyklus zpracování dat od načtení až po vygenerování teplotních výsledků.

    Parametry:
      db_ops: Objekt pro operace s databází.
      geo_proc: Objekt pro prostorové zpracování dat.
      czech_rep: Geometrie České republiky pro vizualizaci.
      elevation_data: Raster nebo matice výškových dat.
      lon_elev: Pole délek pro výšková data.
      lat_elev: Pole šířek pro výšková data.
    """
    global first_run

    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    backend_logger.info(f"Calculation started on {start_datetime}")

    try:
        df = get_data()
        latitudes, longitudes, azimuths, links = db_ops.get_metadata(df)

        df = prepare_data(df, latitudes, longitudes, azimuths, links, elevation_data, lon_elev, lat_elev)
        unique_links, image_name, image_time = collect_data_summary(df)
        df = anomaly_detection(df)
        df = temperature_predict(df)

        grid_x, grid_y, grid_z = spatial_interpolation(
            df, czech_rep, geo_proc, elevation_data, lon_elev, lat_elev
        )

        db_ops.realtime_writer(image_name, unique_links, image_time, grid_z)
        db_ops.save_parameters(start_datetime, grid_x, grid_y)
        map_plotting(grid_x, grid_y, grid_z, czech_rep, image_name)

    except Exception as e:
        backend_logger.error(
            f"Error during data processing round: {e}\n{traceback.format_exc()}"
        )

    finally:
        if "df" in locals():
            del df
        gc.collect()

    end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    backend_logger.info(f"Calculation ended on {end_datetime}. Waiting for another round…")
