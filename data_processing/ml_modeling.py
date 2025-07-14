import joblib
from tensorflow.keras.models import load_model

# Načte předtrénovaný scaler pro škálování vstupních dat
scaler = joblib.load("neural/scaler.joblib")

def temperature_predict(df):
    """
    Provádí predikci teploty pomocí LSTM modelu:

    Parametry:
        df (pandas.DataFrame): DataFrame obsahující sloupce:
            - 'Temperature_MW': aktuální naměřená teplota
            - 'sun': sluneční záření
            - 'Hour': hodina dne
            - 'Day': pořadové číslo dne v roce
            - 'Signal': signál senzoru
            - 'Azimuth': azimut senzoru

    Postup:
        1. Vybere sloupce ve správném pořadí a vytvoří matici X.
        2. Aplikuje předem natrénovaný scaler na X.
        3. Přeformuje data do tvaru (počet vzorků, počet rysů, 1) pro LSTM.
        4. Načte předtrénovaný LSTM model bez kompilace.
        5. Provede predikci a naplní výsledky do sloupce 'Predicted_Temperature'.
        6. Agreguje výsledky průměrem pro každou kombinaci Hour, IP, Latitude, Longitude.

    Vrací:
        pandas.DataFrame: DataFrame se sloupci ['Hour', 'IP', 'Latitude', 'Longitude', 'Predicted_Temperature'],
                          kde 'Predicted_Temperature' je průměrná predikce pro každý senzor dané hodiny a lokace.
    """
    col_order = ['Temperature_MW', 'sun', 'Hour', 'Day', 'Signal', 'Azimuth']
    X = df[col_order]
    X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    model = load_model("neural/lstm.keras", compile=False)
    predicted_temperatures = model.predict(X_reshaped).flatten()
    df["Predicted_Temperature"] = predicted_temperatures
    df = (
        df.groupby(["Hour", "IP", "Latitude", "Longitude"])["Predicted_Temperature"]
        .mean()
        .reset_index()
    )
    return df

