# TelcoTemp Backend

This repository contains the source code for the TelcoTemp backend system, which is designed for opportunistic temperature sensing and visualization of current air temperature distribution across the Czech Republic. The system leverages opportunistic measurements from a network of commercial microwave links.

## Project Overview

Traditional meteorological stations often face high operational costs, leading to insufficiently dense measurement networks. TelcoTemp addresses this by utilizing commercial microwave links, commonly used for data transmission in telecommunication networks, as an efficient way to expand the sensing network without additional investment. These devices are sensitive to ambient meteorological conditions, allowing for indirect estimation of ambient temperature at their locations.

This project focuses on the development of a software platform that uses operational data from commercial microwave links to predict and visualize the current temperature distribution in the Czech Republic.

## Key Features

The TelcoTemp backend performs the following core functions through its source code:

* **Data Acquisition:** Implements logic to gather operational data from telecommunication devices from time-series and relational databases.
* **Temperature Prediction:** Utilizes a Long Short-Term Memory (LSTM) neural network model, implemented in the code, to predict ambient temperature.
* **Spatial Interpolation:** Executes spatial interpolation of predicted temperatures using the regression kriging method, which incorporates elevation data for enhanced accuracy.
* **Temperature Map Generation:** Generates 2D maps of temperature distribution across the Czech Republic as image files.
* **Database Communication Logic:** Manages the logic for interacting with InfluxDB to retrieve raw sensor data and with MariaDB to fetch device metadata and store computational results.
* **Anomaly Detection:** Incorporates an algorithm within the data processing pipeline to identify and remove anomalous data points that could negatively affect prediction accuracy.
* **HTTP Server:** Provides an HTTP interface to serve the generated temperature maps to a frontend application.

## Technologies Used

The project is built primarily with Python and leverages several key libraries and databases within its codebase:

* **Backend Framework:** Python, Flask
* **Neural Networks:** TensorFlow, Keras (for LSTM model implementation)
* **Data Handling:** Pandas, NumPy
* **Geospatial Processing:** GeoPandas, Rasterio, PyProj, Shapely (for geographical operations and handling digital elevation model data)
* **Kriging:** PyKrige (for Regression Kriging implementation)
* **Databases:**
    * InfluxDB (for interacting with time-series operational data)
    * MariaDB (for interacting with metadata and storing calculation results)
* **Visualization:** Matplotlib (for programmatic generation of temperature maps)
* **Logging:** Python's built-in `logging` module with `RotatingFileHandler` for managing application and HTTP logs.
* **Environment Management:** `python-dotenv` (for loading environment variables from `.env` files).

## Architecture

The TelcoTemp platform is designed with a modular architecture, with the backend serving as the central processing unit. The core components from a code perspective include:

* **TelcoTemp Backend:** This Python application leads the entire data processing workflow. It contains modules for data extraction, machine learning modeling, spatial interpolation, and visualization. It also hosts an HTTP server to interface with the frontend.
* **InfluxDB Integration:** The backend includes code to connect to and query InfluxDB, retrieving time-series data from commercial microwave links network.
* **MariaDB Integration:** The backend's code interacts with MariaDB to retrieve device-specific metadata (like geographical coordinates and azimuths) and to store the results and parameters of the generated temperature maps.
* **Frontend Interface:** While the frontend is a separate application (developed in React), the backend provides specific HTTP endpoints for it to request and receive generated temperature maps.

The backend's internal processes and data flow are managed through Python scripts, ensuring continuous updates of temperature maps.

## Data Processing Workflow

The backend automates the data processing in regular, cyclic operations:

1.  **Current Data Loading:** The backend initiates each cycle by querying InfluxDB to obtain the most recent measurements from telecommunication devices, typically aggregating data into 5-minute averages over the last hour.
2.  **Metadata Association:** For each data record retrieved, the backend's code queries MariaDB to associate specific metadata, such as geographical coordinates and azimuths, using device IP addresses as identifiers.
3.  **Data Preprocessing and Transformation:** Before being fed into the prediction model, the dataset is cleaned (e.g., handling missing values) and enriched. New features are derived, such as `hour of the day`, `day of the year`, a binary `sun` indicator (day/night based on location and time), and `elevation` data extracted from a pre-loaded digital elevation model. This step prepares a consistent input format for the neural network.
4.  **Temperature Prediction:** The preprocessed data is then fed into the trained LSTM neural network model, which predicts the ambient temperature for each record. This adds a new column with predicted temperatures to the dataset.
5.  **Spatial Interpolation:** The predicted point temperatures are transformed into a continuous temperature field covering the Czech Republic. This is achieved using a regression kriging implementation that estimates values on a regular grid, taking into account the influence of elevation. The output consists of three matrices: X-coordinates, Y-coordinates, and the corresponding interpolated temperatures.
6.  **Result Persistence:** Before visualization, the backend's code saves metadata about the generated map (e.g., timestamp, list of devices used, min/max temperatures, image filename) and interpolation parameters to MariaDB.
7.  **Temperature Map Visualization:** The final step involves rendering the interpolated temperature grid into a PNG image file using Matplotlib. These image files are saved to a directory accessible via the HTTP server for the frontend to retrieve.

After completing these steps, the backend enters a waiting period before initiating the next processing cycle, ensuring continuous and updated temperature map generation.

## How to Run

1.  **Configuration File:** To run the application, you will need a `.env` configuration file and also cert files placed in the project's root directory. This file contains sensitive information like database credentials and environment settings. To obtain this file, please contact the system administrator at **musilp94@gmail.com**.
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # For Linux/macOS
    source venv/bin/activate
    # For Windows
    venv\Scripts\activate.bat
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Start the Application:**
    ```bash
    python main.py
    ```

**Author:** Jan Hejna
