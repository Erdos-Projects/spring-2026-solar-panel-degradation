# spring-2026-solar-panel-degradation
Team project: spring-2026-solar-panel-degradation

**Predicting Solar Panel Yields**

**Erdos Institute Data Science Bootcamp Spring 2026**

**Project Team: Varun Gudibanda, Hongki Jung, Jacob Kapita, Zhihao Mu, Huachen Sun**

**Motivation:**
Solar facilities have found their way onto the power grids as a sustainable method of power generation. Prior to building a solar facility, predicting the AC power yield from the facility will assess whether that location is sufficient for the area’s power needs. Also, being able to predict the power that will be generated from currently built facilities will also help assess overall trends in power generation. Furthermore, solar energy is traded in day-ahead and real-time markets so accurate power predictions offer a competitive edge to market participants. We seek a data-driven framework by which we can use weather data at a particular location to make predictions of the AC power yield from a solar facility. In particular, given time series data of the weather, we want to forecast the AC power output over time.

**Dataset:**
The National Laboratory of the Rockies (formerly the National Renewable Energy Laboratory) released the NREL 2023 Solar Data Prize Dataset which contains photovoltaic, environmental, and irradiance data along with their AC power output for a large number of solar facilities around the country. These datasets provide time series data in 15 minute increments for the ambient temperature, wind direction, wind speed, irradiance (power from the sun), and the AC power output. We select a particular dataset corresponding to a facility in Arbuckle, CA which has nearly 8 years of data from 2017-2024. Although the datasets provide the data in 15 minute increments, there is missing data for some of these features at particular times. We clean the data by filling gaps of less than 30 minutes with the previous time point’s data. Any other times with NaNs are deleted from the dataset. The number of deleted rows did not constitute a significant number of data points. 

**Methodology:**
Explored the following methods/models:

-Linear Regression

-XGBoost

-Multi-Layer Perceptron

-Random Forest

-Prophet

-LightGBM

**Results:**
Models were assessed and compared via their mean square errors. Amongst these models, XGBoost was seen to perform the best with lowest mean squared error. The data for XGBoost was supplanted with additional features for the hour of the day and the month, which was seen to improve the model. Hyperparameter optimization via randomized search allowed us to find an optimal XGBoost model. This final model showed the most important feature by far was the irradiance with the hour of the day and month as the next most important features. This is consistent because the irradiance is how the solar panels actually produce any power. The hour of the day provides information about daily changes and the month allows the model to learn seasonal changes. The next most important feature after these was the air temperature, which corresponds to the efficiency of a solar panel being temperature dependent with higher efficiencies at lower temperatures. We did notice that XGBoost, though it does well with predicting the overall trends, routinely underestimates the peak power output in a day. 

-----------------------------------
Description of Files:

Training and Test Data from the OEDI 2107 system from the NREL Solar Data Prize Dataset. Contains cleaned environmental data and AC output data:

test_data.csv

train_data.csv




Model Exploration:

solar_panel_2107_ts_split.ipynb - Linear Regression

SolarPanel_LightGBM.ipynb - LightGBM

solarpanels_prophet.ipynb - Prophet Model

xgboost_2107_final.ipynb - XGBoost





Old Files:

NREL_data.ipynb - Initial NREL OEDI data exploration

pvlib_test.ipynb - Initial attempt to use pvlib to calculate performance ratios
