The project is to compare AR1 model with Deep Learning model performance in time series modeling. The focus is on financial risk factor (RF) time series.

# FX Spot Time Series Modelling

## Candidate Models
For simplicity, we start with FX spot time series. It is simple because the RF is a scalar, instead of a term structure like interest rates.

### AR1 model
For AR1 model we consider the following:

1. Estimate the AR1 model parameters from weedly data points, use Wednesday data only to avoid holidays as much as possible.
2. We consider a 3 year period for calibration.
3. The AR1 model parameters are estimated by a maximum likelihood method, assuming Normal residuals, this is equivalend to Ordinary Least Square estimation.

### Deep Learning Model: LSTM (Long Short-Term Memory) Network

For the deep learning approach, we propose using an LSTM (Long Short-Term Memory) neural network, which is well-suited for modeling sequential and time series data due to its ability to capture long-term dependencies.

For the LSTM model, we consider the following:

1. **Data Preparation**: Use the same weekly (Wednesday) data points as for the AR1 model, ensuring a fair comparison. The data is normalized (e.g., using z-score normalization) before being fed into the network.
2. **Training Period**: The LSTM is trained on a rolling 3-year window, matching the calibration period of the AR1 model.
3. **Model Architecture**: The LSTM network consists of one or more LSTM layers, possibly followed by dense layers. The architecture is kept simple to avoid overfitting, given the limited data.
4. **Training Procedure**: The model is trained to predict the next week's RF value given a sequence of previous weeks (e.g., using the past 4-8 weeks as input). The loss function is Mean Squared Error (MSE), and early stopping is used to prevent overfitting.
5. **Evaluation**: The out-of-sample predictive performance of the LSTM is compared to the AR1 model using standard metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

This setup allows for a direct comparison between the classical AR1 approach and a modern deep learning technique for time series forecasting.

## Model Performance

Test and compare model performance by performing back testing.

### Backtesting Methodology
Models are backtested with the following approach:
1. Select 50 hop dates, each being 10 days aparts.
2. On each hop date, calibrate the candidate models to the 3 year window immediately before the hop dates.
3. Using the calibrated models, simulate and predict the RF value in 10 days time.
4. Use 1000 simulation
5. compare the simulation to the realised RF value and performa a binomial test at the 5th, 15th, 85th and 95th centile accross the hop dates respectively, and compute the test p-values for each of the centile tested.

