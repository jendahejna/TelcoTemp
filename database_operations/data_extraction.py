import logging
from influxdb_client import InfluxDBClient
import pandas as pd
from astral.sun import sun
from astral import LocationInfo
import pytz
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

"""
Modul pro získávání a základní předzpracování dat z InfluxDB a určení denního světla.

Obsahuje:
  - Konstanty pro připojení k InfluxDB a parametry lokace.
  - Funkci `is_daylight` pro zjištění, zda je daný čas ve světle.
  - Funkci `get_data` pro načtení, agregaci a transformaci dat z InfluxDB do pandas DataFrame.
"""

load_dotenv()
token = os.getenv("INFLUX_TOKEN")
url = os.getenv("INFLUX_URL_PUBLIC")
org = os.getenv("ORG")
bucket = os.getenv("BUCKET")
backend_logger = logging.getLogger('backend_logger')
technology = os.getenv("TECHNOLOGY")
# Konstanty pro výpočet denního světla
LAT = 49.8175
LNG = 15.4730
PRAGUE_TZ = pytz.timezone("Europe/Prague")

# Funkce pro zjištění denního světla
def is_daylight(time):
    """
    Určí, zda zadaný čas spadá do období denního světla pro definovanou polohu.

    Parametry:
        time (datetime.datetime): Čas (s lokalizací Europe/Prague) k vyhodnocení.

    Návratová hodnota:
        int: 1 pokud je čas mezi východem a západem slunce, jinak 0.
    """
    location = LocationInfo(latitude=LAT, longitude=LNG)
    s = sun(location.observer, date=time.date())
    sunrise = s["sunrise"].astimezone(PRAGUE_TZ)
    sunset = s["sunset"].astimezone(PRAGUE_TZ)
    return 1 if sunrise <= time <= sunset else 0


# Funkce pro získání dat
def get_data(retry_count=3):
    """
    Načte a předzpracuje data z InfluxDB, včetně výpočtu UNIX času a indikátoru denního světla.

    Parametry:
        retry_count (int): Počet pokusů o načtení dat v případě chyby (výchozí 3).

    Návratová hodnota:
        pandas.DataFrame: DataFrame se sloupci:
            - Time: čas záznamu (datetime)
            - Unix: UNIX timestamp (int)
            - Temperature_MW: průměrná teplota za 5 minut (float)
            - Signal: průměrná přijímaná úroveň signálu za 5 minut (float)
            - IP: adresa senzoru (str)
            - sun: indikator denního světla (1 = den, 0 = noc)
    """
    for attempt in range(retry_count):
        try:
            with InfluxDBClient(url=url, token=token) as client:
                now_utc = datetime.utcnow()
                rounded_down_hour_utc = now_utc - timedelta(minutes=now_utc.minute, seconds=now_utc.second)

                query = f"""
                from(bucket: \"{bucket}\")
                  |> range(start: - 1h)
                  |> filter(fn: (r) => r["_measurement"] == "{technology}")
                  |> filter(fn: (r) => r["_field"] == "Teplota" or r["_field"] == "PrijimanaUroven" or r["_field"] == "Uptime")
                  |> aggregateWindow(every: 5m, fn: mean)
                  |> group(columns: ["_measurement", "_field", "agent_host"])
                """

                result = client.query_api().query(org=org, query=query)

                # Přepracované zpracování dat
                data = [
                    {
                        "Time": record.get_time(),
                        "Measurement": record.values["_field"],
                        "Value": record.get_value(),
                        "IP": record.values["agent_host"],
                    }
                    for table in result
                    for record in table.records
                ]

                df = pd.DataFrame(data)
                df_pivot = df.pivot_table(
                    index=["Time", "IP"], columns="Measurement", values="Value"
                ).reset_index()
                df_pivot["Time"] = pd.to_datetime(df_pivot["Time"])
                df_pivot["Unix"] = df_pivot["Time"].astype("int64") // 10**9
                df_pivot["sun"] = df_pivot["Time"].apply(is_daylight)
                new_column_order = ["Time", "Unix", "Teplota", "PrijimanaUroven", "Uptime", "IP", "sun"]
                df_final = df_pivot[new_column_order].rename(
                    columns={"Teplota": "Temperature_MW", "PrijimanaUroven": "Signal"}
                )
                logging.info(f"Data úspěšně získána: {len(df_final)} záznamů")
                return df_final
        except Exception as e:
            backend_logger.warning(f"Nepodařilo se získat data: {e}")
            df_final = pd.DataFrame()
    return df_final

