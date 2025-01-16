import pandas as pd
import os
import numpy as np
from taipy import Gui


#Loading electricity data
data_elec = pd.read_csv("electricity_cleaned_reduced.csv", index_col="timestamp", parse_dates=True)
building_name = "Panther_office_Hannah"
data_frame_elec = pd.DataFrame((data_elec[building_name].truncate(before = '2017-01-01')).ffill())

#Since the orignal file is too large to upload on GitHub we will save the new required file.
# data_frame_elec.to_csv("electricity_cleaned_reduced.csv")

#Loading weather data
data_weather = pd.read_csv("weather_reduced.csv", index_col="timestamp", parse_dates=True)
data_frame_weather = pd.DataFrame(data_weather.truncate(before = "2017-01-01"))
# data_frame_weather.to_csv("weather_reduced.csv")
weather_hourly = data_frame_weather.resample("h").mean()
weather_hourly_nooutlier = weather_hourly[weather_hourly > -40]
weather_hourly_nooutlier_nogaps = weather_hourly_nooutlier.ffill()
#
temperature = weather_hourly_nooutlier_nogaps["airTemperature"]


training_months = [3,4,5,6,8,9]
test_months = [7]
training_data = data_frame_elec[data_frame_elec.index.month.isin(training_months)]
test_data = data_frame_elec[data_frame_elec.index.month.isin(test_months)]


#Encoding
train_features = pd.concat([pd.get_dummies(training_data.index.hour),
                                     pd.get_dummies(training_data.index.dayofweek),
                                     pd.DataFrame(temperature[temperature.index.month.isin(training_months)].values)], axis=1).dropna()

from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler

model = HuberRegressor().fit(np.array(train_features), np.array(training_data.values))

# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=42)
# model.fit(np.array(train_features), np.array(training_data.values))

test_features = np.array(pd.concat([pd.get_dummies(test_data.index.hour),
                                    pd.get_dummies(test_data.index.dayofweek),
                                    pd.DataFrame(temperature[temperature.index.month.isin(test_months)].values)], axis=1).dropna())

predictions = model.predict(test_features)



def chart_data(y_actual, y_predicted):
    data = pd.DataFrame({"Index": range(len(y_actual)),
                         "Actual": y_actual.flatten(),
                         "Predicted": y_predicted.flatten()})
    return data

prepared_chart = chart_data(test_data.values, predictions)
#Calculating errors:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2, mae

rmse, mape, r2, mae = evaluate_model_performance(test_data.values, predictions)


page = """
<|text-center|
<h1> Energy Consumption </h1>

<h5> Predicting energy consumption of a building in a city using regressor and neural network based on various factors.</h5>

<a href = "https://www.kaggle.com/code/chuanfutan/energy-consumption-forecasting-project/notebook#Introduction-to-Machine-Learning-for-the-Built-Environment:-Energy-Consumption-Forecasting"> Link to Kaggle dataset </a>

<a href = "https://github.com/ArjunTewari/Energy_Consumption.git"> Link to GitHub repository </a>

<h3> Predicted vs Actual Values for huber regressor : </h3>

<|{prepared_chart}|chart|type=line|x = Index|y[1]=Actual|y[2]=Predicted|title = Actual vs Predicted values|>

<h3> Metrics of the model </h3>
gi
<|layout|columns = 1 1 1 1|
# MAE : <|metric|value={mae:.2f}|>
# RMSE : <|metric|value={rmse:.2f}|>
# MAPE : <|metric|value={mape:.2f}%|>
# R² : <|metric|value={r2:.2f}|>
|>
>
"""

if __name__== "__main__":
    app = Gui(page)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False, debug=False)