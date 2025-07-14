"""
Modul pro prostorovou interpolaci predikovaných teplot pomocí regresního krigingu.

Funkce:
    spatial_interpolation: Vytvoří pravidelnou mřížku nad zadanou oblastí, interpoluje elevaci
    pro vstupní data, trénuje model regresního krigování a interpoluje teplotu na mřížce.
"""
import numpy as np
from scipy.interpolate import griddata
from pykrige.rk import RegressionKriging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import logging

backend_logger = logging.getLogger('backend_logger')


def spatial_interpolation(
        df,
        rep,
        geo_proc,
        elevation_data,
        lon_elev,
        lat_elev,
        variogram_model='spherical',
        nlags=40,
        regression_model_type='linear'
):
    """
    Provádí prostorovou interpolaci předpovězených teplot nad zadanou oblastí.

    Parametry:
        df (pandas.DataFrame): DataFrame se sloupci 'Longitude', 'Latitude', 'Predicted_Temperature'.
        rep: geopandas.GeoDataFrame nebo geometrie oblasti (např. obrys ČR) pro definici hranic.
        geo_proc: Objekt poskytující metodu create_mask(rep, grid_x, grid_y) pro vygenerování masky pro oblast.
        elevation_data (numpy.ndarray): matice elevací pro body definované lon_elev a lat_elev.
        lon_elev (numpy.ndarray): matice délkových souřadnic odpovídající elevation_data.
        lat_elev (numpy.ndarray): matice šířkových souřadnic odpovídající elevation_data.
        variogram_model (str): typ variogramového modelu pro krigování ('spherical', 'exponential', ...).
        nlags (int): počet sousedních bodů použité pro interpolaci v Regression Kriging.
        regression_model_type (str): typ regresního modelu pro trendovou složku. Možné hodnoty:
            - 'linear': LinearRegression
            - 'random_forest': RandomForestRegressor
            - 'gradient_boosting': GradientBoostingRegressor
            - 'svr': Support Vector Regression (SVR)

    Postup:
        1. Vygeneruje pravidelnou mřížku (grid_x, grid_y) nad oblastí rep.
        2. Vytvoří binární masku bodů uvnitř oblasti pomocí geo_proc.
        3. Extrahuje platné body a jejich predikované teploty z df.
        4. Interpoluje elevaci vstupních bodů pomocí griddata.
        5. Při přítomnosti NaN v elevaci je nahradí průměrem.
        6. Vybere regresní model podle regression_model_type.
        7. Inicializuje a natrénuje Regression Kriging (rk.fit).
        8. Interpoluje elevaci celé mřížky a predikuje teplotu (rk.predict).
        9. Aplikuje masku, body mimo oblast označí NaN.

    Vrací:
        grid_x (numpy.ndarray): 2D matice X souřadnic mřížky.
        grid_y (numpy.ndarray): 2D matice Y souřadnic mřížky.
        grid_predicted_temp (numpy.ndarray): 2D matice interpolovaných teplot,
            kde body mimo oblast jsou NaN.

    Výjimky:
        ValueError: pokud se neshoduje délka vstupních polí nebo neznámý regression_model_type.
        Ostatní výjimky jsou zalogovány a znovu vyhozeny.
    """
    backend_logger.info(
        "spatial_interpolation function started. Using regression model: %s", regression_model_type
    )
    try:

        bounds = rep.total_bounds
        grid_x, grid_y = np.mgrid[
                         bounds[0]:bounds[2]:500j,
                         bounds[1]:bounds[3]:500j
                         ]
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        mask = geo_proc.create_mask(rep, grid_x, grid_y)
        backend_logger.debug("Grid and mask created.")

        valid_points = (
                ~df['Longitude'].isna() &
                ~df['Latitude'].isna() &
                ~df['Predicted_Temperature'].isna()
        )
        lon = df.loc[valid_points, 'Longitude'].values
        lat = df.loc[valid_points, 'Latitude'].values
        temp = df.loc[valid_points, 'Predicted_Temperature'].values

        points = np.c_[lon_elev.ravel(), lat_elev.ravel()]
        values = elevation_data.ravel()
        valid_elev = griddata(points, values, (lon, lat), method='linear')
        backend_logger.debug("Elevation interpolated for input data.")

        if np.isnan(valid_elev).any():
            valid_elev = np.nan_to_num(valid_elev, nan=np.nanmean(valid_elev))
            backend_logger.debug(
                "NaN values in elevation replaced with the mean value."
            )

        if not (len(lon) == len(lat) == len(temp) == len(valid_elev)):
            backend_logger.error(
                "Data dimensions mismatch: lon=%d, lat=%d, temp=%d, elev=%d",
                len(lon), len(lat), len(temp), len(valid_elev)
            )
            raise ValueError(
                "Dimensions of coordinates, temperature, and elevation do not match. Check input data."
            )

        # Step 3: Choose regression model
        if regression_model_type == 'linear':
            regression_model = LinearRegression()
        elif regression_model_type == 'random_forest':
            regression_model = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
        elif regression_model_type == 'gradient_boosting':
            regression_model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
        elif regression_model_type == 'svr':
            regression_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            backend_logger.error(
                "Unknown regression model type: %s", regression_model_type
            )
            raise ValueError(
                f"Unknown regression model type: {regression_model_type}"
            )
        backend_logger.debug(
            "Regression model '%s' selected successfully.", regression_model_type
        )

        X = valid_elev.reshape(-1, 1)
        rk = RegressionKriging(
            regression_model=regression_model,
            variogram_model=variogram_model,
            n_closest_points=nlags
        )
        backend_logger.info("Regression Kriging model initialized.")

        rk.fit(X, np.c_[lon, lat], temp)
        backend_logger.info("Regression Kriging model trained successfully.")

        grid_elev = griddata(points, values, (grid_x, grid_y), method='linear')
        backend_logger.debug("Elevation grid interpolated.")

        if np.isnan(grid_elev).any():
            grid_elev = np.nan_to_num(grid_elev, nan=np.nanmean(valid_elev))
            backend_logger.debug(
                "NaN values in grid elevation replaced with the mean value."
            )

        grid_predicted_temp = rk.predict(
            grid_elev.reshape(-1, 1), np.c_[grid_x.ravel(), grid_y.ravel()]
        )
        grid_predicted_temp = grid_predicted_temp.reshape(grid_x.shape)
        backend_logger.info("Temperature prediction on grid completed.")

        grid_predicted_temp = np.where(
            mask.reshape(grid_x.shape), grid_predicted_temp, np.nan
        )
        backend_logger.info("Mask applied. spatial_interpolation function finished.")

        return grid_x, grid_y, grid_predicted_temp

    except Exception as e:
        backend_logger.exception(
            "Exception in spatial_interpolation function: %s", e
        )
        raise
