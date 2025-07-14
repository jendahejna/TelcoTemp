"""
Modul pro geografické zpracování: převod GeoJSON do GeoDataFrame, tvorba masky oblasti,
načítání hranic země a výškových dat.
"""
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import json
import rasterio
from pyproj import Transformer

class GeographicalProcessing:
    """
    Poskytuje metody pro převod a zpracování geografických dat.

    Metody:
        json_to_geodataframe: Převod GeoJSON struktury na GeoDataFrame se souřadnicemi.
        create_mask: Vytvoří masku mřížky bodů uvnitř zadané oblasti.
        load_country_data: Načte GeoJSON soubor s hranicemi státu.
        load_elevation_data: Načte rastrová data výšek a převede pixely na geografické souřadnice.
    """

    def json_to_geodataframe(self, json_data):
        """
        Převádí GeoJSON data na GeoDataFrame s geometriemi polygonů.

        Parametry:
            json_data (dict): Struktura GeoJSON obsahující klíč 'features' s geometriemi polygonů.

        Vrací:
            geopandas.GeoDataFrame: GeoDataFrame se sloupcem 'geometry' obsahujícím Polygon objekty
                                   v CRS EPSG:4326.
        """
        geometries = []

        for feature in json_data["features"]:
            poly = Polygon(feature["geometry"]["coordinates"][0])
            geometries.append(poly)

        gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        return gdf

    def create_mask(self, czech_rep, grid_x, grid_y):
        """
        Vytvoří masku pro body ležící uvnitř zadané oblasti (polygonu).

        Parametry:
            czech_rep (GeoDataFrame nebo GeoSeries): Geometrie oblasti (např. obrys ČR).
            grid_x (numpy.ndarray): 2D matice X souřadnic bodů.
            grid_y (numpy.ndarray): 2D matice Y souřadnic bodů.

        Vrací:
            numpy.ndarray: Boolean maska stejného tvaru jako grid_x/grid_y,
                           kde True označuje body uvnitř oblasti.
        """
        mask = np.zeros_like(grid_x, dtype=bool)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                point = Point(grid_x[i, j], grid_y[i, j])
                mask[i, j] = czech_rep.contains(point).any()
        return mask

    def load_country_data(self, country_file_path):
        """
        Načte GeoJSON soubor s hranicemi státu.

        Parametry:
            country_file_path (str): Cesta k souboru .json obsahujícím GeoJSON data.

        Vrací:
            dict: Načtená struktura GeoJSON jako Python slovník.
        """
        with open(country_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def load_elevation_data(self, tif_path):
        """
        Načte rastrová data výšek (GeoTIFF) a převede pixely na zeměpisné souřadnice.

        Parametry:
            tif_path (str): Cesta k souboru GeoTIFF s výškovými daty v CRS EPSG:3045.

        Postup:
            1. Otevře rastrová data pomocí rasterio.
            2. Vytvoří mřížku pixelových souřadnic (x_pixels, y_pixels).
            3. Pomocí transformace rasteru vypočte souřadnice lon, lat v původním CRS.
            4. Převod na WGS84 (EPSG:4326) pomocí pyproj Transformer.
            5. Nahradí nodata hodnoty NaN.

        Vrací:
            elevation_data (numpy.ndarray): 2D matice výšek.
            lon (numpy.ndarray): 2D matice zeměpisných délek (EPSG:4326).
            lat (numpy.ndarray): 2D matice zeměpisných šířek (EPSG:4326).
        """
        with rasterio.open(tif_path) as src:
            transformer = Transformer.from_crs("EPSG:3045", "EPSG:4326", always_xy=True)
            transform_matrix = src.transform
            width = src.width
            height = src.height

            x_pixels, y_pixels = np.meshgrid(np.arange(width), np.arange(height))
            x_coords, y_coords = rasterio.transform.xy(transform_matrix, y_pixels, x_pixels)
            lon, lat = transformer.transform(x_coords, y_coords)

            elevation_data = src.read(1)
            nodata_value = -3.4028234663852886e+38
            elevation_data = np.where(elevation_data == nodata_value, np.nan, elevation_data)

        return elevation_data, lon, lat
