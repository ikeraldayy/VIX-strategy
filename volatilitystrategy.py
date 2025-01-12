import datetime
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import backtrader as bt
from arch import arch_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_data(stocks, start, end):
    # Download the data using yfinance
    stockData = yf.download(stocks, start=start, end=end)
    
    # Rename columns to match backtrader's expectations if necessary
    stockData.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Ensure the index is a datetime type (should already be by default)
    stockData.index = pd.to_datetime(stockData.index)
    
    return stockData


end_date = datetime.datetime.now().replace(tzinfo=None) #Timezone out

start_date = yf.download('^VIX', period="max").index[0] + datetime.timedelta(days=5000)

estimation_start_date = start_date - datetime.timedelta(days=9000)

dataSPY = get_data('SPY', start_date, end_date)
VIX = get_data('^VIX', start_date, end_date)

dataSPY_estimation = get_data('SPY', estimation_start_date, end_date)
#%%

def diagnostics(results):
    # Extract residuals
    residuals = results.resid
    std_residuals = residuals / results.conditional_volatility

    # 1. Residuals Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(std_residuals)
    plt.title('Standardized Residuals')

    # 2. Histogram of Residuals
    plt.subplot(2, 2, 2)
    plt.hist(std_residuals, bins=30, edgecolor='k')
    plt.title('Histogram of Standardized Residuals')

    # 3. Q-Q Plot for Normality
    plt.subplot(2, 2, 3)
    sm.qqplot(std_residuals, line='s', ax=plt.gca())
    plt.title('Q-Q Plot')

    # 4. ACF Plot (Autocorrelation)
    plt.subplot(2, 2, 4)
    sm.graphics.tsa.plot_acf(std_residuals, lags=20, ax=plt.gca())
    plt.title('Autocorrelation (ACF) of Residuals')

    plt.tight_layout()
    plt.show()

#test autocorrelation
def ljung_box_test(results, lags=20):
    # Standardized residuals
    std_residuals = results.resid.dropna() / results.conditional_volatility.dropna()

    # Perform Ljung-Box test
    lb_test = acorr_ljungbox(std_residuals, lags=lags, return_df=True)
    print(lb_test)


def arch_lm_test(results):
    residuals = results.resid.dropna()
    test_stat, p_value, _, _ = het_arch(residuals)
    print(f"ARCH-LM Test p-value: {p_value}")

#test normality
def jarque_bera_test(results):
    residuals = results.resid
    jb_stat, p_value = jarque_bera(residuals)
    print(f"Jarque-Bera Test p-value: {p_value}")


def forecast_garch_volatility(data, p=1, q=2):
    """
    Forecast next-period realized volatility using an EGARCH model.

    Parameters:
    data (pd.DataFrame): DataFrame containing price data with 'Close' column.
    p (int): Lag order for GARCH terms.
    q (int): Lag order for ARCH terms.
    o (int): Lag order for asymmetric terms (EGARCH).

    Returns:
    pd.Series: Forecasted volatility for each period.
    """
    # Calculate log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100  # Rescale to percentage
    data['realized_vol'] = data['log_return'].rolling(window=21).std()
    returns = data['log_return'].dropna()
    volatility = data['realized_vol'].dropna()

    # Fit EGARCH model
    model = arch_model(returns, mean='AR', lags=1, vol='Garch', p=p, q=q, dist='t')
    results = model.fit(disp='off')
    
    print(results.summary())  # Displays model details like coefficients and diagnostics
    print("\nModel Parameters:")
    print(results.params)  # Displays parameter values only

    # Diagnostics
    diagnostics(results)
    ljung_box_test(results)
    arch_lm_test(results)
    jarque_bera_test(results)

   # Forecast volatility for each day
    forecast = results.forecast(start=returns.index[0])
    forecasted_volatility = np.sqrt(forecast.variance.iloc[:, 0])
    #forecasted_volatility = forecast.iloc[:,0]
    # Realized volatility
    realized_volatility = returns.rolling(window=21).std()
    realized_volatility = realized_volatility[forecasted_volatility.index]



    # Calculate Mean Squared Error (MSE)
    mse_egarch = ((forecasted_volatility - realized_volatility) ** 2).mean()
    print(f"MSE (EGARCH): {mse_egarch}")
    
    dates = ['2011-10-10', '2017-05-06', '2023-03-16', '2024-06-13']
    for date in dates:
        if date in forecasted_volatility.index:
            print(f"{date}: Forecasted = {forecasted_volatility[date]:.2f}, Realized = {realized_volatility[date]:.2f}")

    baseline_forecast = realized_volatility.shift(1)
    mse_baseline = ((baseline_forecast - realized_volatility) ** 2).mean()
    print(f"MSE (Baseline): {mse_baseline}")

    # Compare forecasts visually
    plt.figure(figsize=(12, 6))
    plt.plot(forecasted_volatility, label='EGARCH Forecasted Volatility')
    plt.plot(baseline_forecast, label='Baseline Forecasted Volatility')
    plt.plot(realized_volatility, label='Realized Volatility')
    plt.title('Forecasted vs Realized Volatility (EGARCH vs Baseline)')
    plt.legend()
    plt.show()
    
    return forecasted_volatility

vol_forecast = forecast_garch_volatility(dataSPY_estimation)
print(vol_forecast)
#%%

def forecast_randomforest_volatility(data):
    
    # Calculate log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100  # Rescale to percentage
    data['realized_vol'] = data['log_return'].rolling(window=21).std()
    
    for i in range(1, 6):  # Use past 5 days' volatilities as features
        data[f'lag_{i}'] = data['realized_vol'].shift(i)

    data.dropna(inplace=True)

    # Features (X) and target (y)
    X = data[[f'lag_{i}' for i in range(1, 6)]]
    y = data['realized_vol']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Realized Volatility (Actual)')
    plt.plot(y_pred, label='Predicted Volatility (ML Model)')
    plt.title('Machine Learning Volatility Forecast vs Realized Volatility')
    plt.legend()
    plt.show()
    
    return y_pred

pred = forecast_randomforest_volatility(dataSPY_estimation)
#%%

def forecast_randomforest_next_volatility(data, date):
    """
    Predict the next-period realized volatility based on past 5 volatilities using RandomForest.

    Parameters:
    data (pd.DataFrame): DataFrame containing price data with 'Close' column.
    date (str): Date in 'YYYY-MM-DD' format to use as the starting point for prediction.

    Returns:
    float: Predicted realized volatility for the next tradeable day.
    """
    
    # Make a copy to prevent modifying original data
    data = data.copy()

    # Calculate log returns and realized volatility
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100  # Rescale to percentage
    data['realized_vol'] = data['log_return'].rolling(window=21).std()
    
    # Generate lag features for past 5 volatilities
    for i in range(0, 5):
        data[f'lag_{i}'] = data['realized_vol'].shift(i)

    # Drop rows with NaN values
    data = data.dropna().copy()

    # Features (X) and target (y)
    X = data[[f'lag_{i}' for i in range(0, 5)]]
    y = data['realized_vol']

    # Train-test split
    X_train = X.iloc[:-1]  # Train on all but the last row
    y_train = y.iloc[:-1]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the provided date
    if date not in data.index:
        raise ValueError(f"Date {date} not found in data index!")

    # Use the specified date for prediction
    date_index = data.index.get_loc(date)
    if date_index < 5:
        raise ValueError("Not enough data before the given date for lag features!")

    # Test input features
    X_test = X.iloc[[date_index]]  # Use the specific row corresponding to the date
    predicted_volatility = model.predict(X_test)[0]

    return predicted_volatility

forecast_randomforest_next_volatility(dataSPY_estimation, '2024-12-27')

#%%

def RSI(data, periods, date):
    delta = data['Close'].diff(1)
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=periods).mean()    
    avg_loss = pd.Series(loss).rolling(window=periods).mean()
    
    rs = avg_gain/avg_loss
    rsi = 100 - (100 / (1 + rs))
    date_index = data.index.get_loc(date)
    rsi_today = rsi[date_index]
    return rsi_today

rsi = RSI(dataSPY_estimation, 8, '2024-12-27')
print(rsi)

#%%

def dispersion(data, date):
    
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100  # Rescale to percentage
    data['dispersion'] = data['log_return'].rolling(window=21).std()
    if date not in data.index:
        raise ValueError(f"Date {date} not found in data index!")
    date_index = data.index.get_loc(date)
    if date_index < 5:
        raise ValueError("Not enough data before the given date for lag features!")
    dispersion_today = data.loc[date]['dispersion']
    return dispersion_today

dispersion1 = dispersion(VIX, '2024-12-27')
print(dispersion1)

#%%

from fredapi import Fred

def treasury_rates_slope(start_date, end_date):
    
    import requests
    
    api_key = '50ae35102d18c632aea56dba92f97b85'
    fred = Fred(api_key)
    
    tickers = {
    '3M': 'DGS3MO',  # 3-Month
    '2Y': 'DGS2',    # 2-Year
    '10Y': 'DGS10',  # 10-Year
    '30Y': 'DGS30'   # 30-Year
}

    data = pd.DataFrame({
        label: fred.get_series(ticker, observation_start=start_date.strftime('%Y-%m-%d'), observation_end = end_date)
        for label, ticker in tickers.items()
    })
    
    data = data.dropna(how='all')
    
    data1 = data['10Y']-data['2Y']
    data2 = data['30Y']-data['3M'] 
    
    today_rate = data.iloc[-1]
    
    spread1 = today_rate['10Y']-today_rate['2Y']
    spread2 = today_rate['30Y']-today_rate['3M']

    return data1,data2

print(treasury_rates_slope(start_date, '2024-12-26'))

#%%

#correlation matrix for all of the signals
dates = sorted(list(set(dataSPY_estimation.index).intersection(VIX.index)))  # Common dates sorted

dataSPY_estimation['log_return'] = np.log(dataSPY_estimation['Close'] / dataSPY_estimation['Close'].shift(1)) * 100  # Rescale to percentage
dataSPY_estimation['realized_vol'] = dataSPY_estimation['log_return'].rolling(window=21).std()

start_date = dates[0]  # Start date for API
end_date = dates[-1]  # End date for API

# Pre-fetch spreads in one call
spread1_list, spread2_list = treasury_rates_slope(start_date, end_date)
# Initialize storage
results = {'Date': [], 'Volatility': [], 'Volatility Spread': [], 'RSI': [], 'Dispersion': [], 'Spread1': [], 'Spread2': []}

# Calculate values for each date
for date in dates:
    results['Date'].append(date)  # Append the date first
    try:
        results['Volatility'].append(dataSPY_estimation.loc[date]['realized_vol'])
    except Exception as e:
        results['Volatility'].append(np.nan)  # Append NaN on failure
        print(f"Volatility error on {date}: {e}")
    try:
        # Volatility
        results['Volatility Spread'].append((VIX.loc[date]['Close'] - dataSPY_estimation.loc[date]['realized_vol'])/VIX.loc[date]['Close'])
    except Exception as e:
        results['Volatility Spread'].append(np.nan)  # Append NaN on failure
        print(f"Volatility Spread error on {date}: {e}")
    try:
        # RSI
        results['RSI'].append(RSI(dataSPY_estimation, 8, date.strftime('%Y-%m-%d')))
    except Exception as e:
        results['RSI'].append(np.nan)
        print(f"RSI error on {date}: {e}")
    
    try:
        # Dispersion
        results['Dispersion'].append(dispersion(VIX, date.strftime('%Y-%m-%d')))
    except Exception as e:
        results['Dispersion'].append(np.nan)
        print(f"Dispersion error on {date}: {e}")
    try:
        results['Spread1'].append(spread1_list.loc[date])
        results['Spread2'].append(spread2_list.loc[date])
    except Exception as e:
        results['Spread1'].append(np.nan)
        results['Spread2'].append(np.nan)
        print(f"Treasury rates error on {date}: {e}")
    
df = pd.DataFrame(results).set_index('Date')

# Calculate Correlation Matrix
correlation_matrix = df.corr()

print(correlation_matrix)

#%%

def z_score_vol_spread(date):
    
    z_score = (df.loc[date]['Volatility Spread']-df['Volatility Spread'].mean())/df['Volatility Spread'].std()
    return z_score

def z_score_dispersion(date):

    z_score = (df.loc[date]['Dispersion']-df['Dispersion'].mean())/df['Dispersion'].std()
    return z_score

def z_score_rate_spread1(date):
    
    z_score = (df.loc[date]['Spread1']-df['Spread1'].mean())/df['Spread1'].std()
    return z_score

def z_score_rate_spread2(date):
    
    z_score = (df.loc[date]['Spread2']-df['Spread2'].mean())/df['Spread2'].std()
    return z_score

print(z_score_vol_spread('2024-12-26'))
print(z_score_dispersion('2024-12-26'))
print(z_score_rate_spread1('2024-12-26'))
print(z_score_rate_spread2('2024-12-26'))

#%%
import matplotlib.pyplot as plt
import numpy as np

# Compute Z-Scores
df['VIX'] = VIX['Close'].reindex(df.index)

# Define rolling window size
window_size = 3000

# Initialize Z-Score columns with NaN (since the first 999 rows cannot have Z-scores)
df['Z_Vol_Spread'] = np.nan
df['Z_Dispersion'] = np.nan
df['Z_Close'] = np.nan

# Compute Z-scores using rolling window (past 1000 datapoints)
for i in range(window_size, len(df)):
    # Use only the past 'window_size' rows for calculations
    window = df.iloc[i - window_size:i]
    
    # Z-Score for Volatility Spread
    df.loc[df.index[i], 'Z_Vol_Spread'] = ((df.loc[df.index[i], 'Volatility Spread'] - window['Volatility Spread'].mean()) / window['Volatility Spread'].std())
    
    # Z-Score for Dispersion
    df.loc[df.index[i], 'Z_Dispersion'] = ((df.loc[df.index[i], 'Dispersion'] - window['Dispersion'].mean()) / window['Dispersion'].std())
    
    # Z-Score for Close (VIX)
    df.loc[df.index[i], 'Z_Close'] = ((df.loc[df.index[i], 'VIX'] - window['VIX'].mean()) / window['VIX'].std())

# Calculate Z-scores for spreads (fixed window since they aren't rolling)
df['Z_Spread1'] = (df['Spread1'] - df['Spread1'].mean()) / df['Spread1'].std()
df['Z_Spread2'] = (df['Spread2'] - df['Spread2'].mean()) / df['Spread2'].std()

# Number of subplots
n_subplots = 60
data_split = np.array_split(df, n_subplots)

# Set up figure with subplots
fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 150))  # Wider and taller for visibility

threshold = 2

y_min = VIX['Close'].min()
y_max = VIX['Close'].max()

# Loop through each subplot
for i, ax in enumerate(axes):
    data = data_split[i]
    
    # Plot VIX
    ax.plot(data.index, VIX['Close'].reindex(data.index), label='VIX', color='blue')
    
    ax.scatter(data.index[data['Z_Close'] > 2], 
               VIX['Close'].reindex(data.index)[data['Z_Close'] > 2], 
               color='black', label='Z_Close > 2')
    
    ax.scatter(data.index[data['Z_Close'] < -1.05], 
               VIX['Close'].reindex(data.index)[data['Z_Close'] < -1.05], 
               color='red', label='Z_Close < -1.05')
    
    # Volatility Spread
    ax.scatter(data.index[data['Z_Vol_Spread'] > threshold], 
               VIX['Close'].reindex(data.index)[data['Z_Vol_Spread'] > threshold], 
               color='black', label='Vol Spread > 2')
    
    ax.scatter(data.index[data['Z_Vol_Spread'] < -threshold], 
               VIX['Close'].reindex(data.index)[data['Z_Vol_Spread'] < -threshold], 
               color='darkred', label='Vol Spread < -2')
    
    # Dispersion
    
    ax.scatter(data.index[data['Z_Dispersion'] < -1.05], 
               VIX['Close'].reindex(data.index)[data['Z_Dispersion'] < -1.05], 
               color='magenta', label='Dispersion < -1.05')

    # Spread1
    ax.scatter(data.index[data['Z_Spread1'] > threshold], 
               VIX['Close'].reindex(data.index)[data['Z_Spread1'] > threshold], 
               color='orange', label='Spread1 > 2')
    
    ax.scatter(data.index[data['Z_Spread1'] < -threshold], 
               VIX['Close'].reindex(data.index)[data['Z_Spread1'] < -threshold], 
               color='slategray', label='Spread1 < -2')

    # Spread2
    ax.scatter(data.index[data['Z_Spread2'] > threshold], 
               VIX['Close'].reindex(data.index)[data['Z_Spread2'] > threshold], 
               color='purple', label='Spread2 > 2')
    
    # RSI
    ax.scatter(data.index[data['RSI'] > 90], 
               VIX['Close'].reindex(data.index)[data['RSI'] > 90], 
               color='aqua', label='RSI > 90')
    
    ax.scatter(data.index[data['RSI'] < 20], 
               VIX['Close'].reindex(data.index)[data['RSI'] < 20], 
               color='lightgreen', label='RSI < 29')
    
    # Formatting
    ax.set_title(f'Section {i+1}', fontsize=10)
    ax.grid(True)
    ax.set_xlim(data.index.min(), data.index.max())# Stretch each section to full x-axis
    ax.set_ylim(y_min, y_max)


# Global title
fig.suptitle('VIX with Z-Score Threshold Alerts (Split into 80 Sections)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust spacing to fit title
plt.show()

#%%

class CustomPandasData(bt.feeds.PandasData):
    lines = ('volatility', 'volatility_spread', 'rsi', 'dispersion', 'spread1', 'spread2', 'vix', 'z_vol_spread', 'z_dispersion', 'z_close', 'z_spread1', 'z_spread2')
    params = (
    ('volatility', 'Volatility'),
    ('volatility_spread', 'Volatility Spread'),
    ('rsi', 'RSI'),
    ('dispersion', 'Dispersion'),
    ('spread1', 'Spread1'),
    ('spread2', 'Spread2'),
    ('vix', 'VIX'),
    ('z_vol_spread', 'Z_Vol_Spread'),
    ('z_dispersion', 'Z_Dispersion'),
    ('z_close', 'Z_Close'),
    ('z_spread1', 'Z_Spread1'),
    ('z_spread2', 'Z_Spread2')
)

def get_data(stocks, start, end):
    # Download the data using yfinance
    stockData = yf.download(stocks, start=start, end=end)
    
    # Rename columns to match backtrader's expectations if necessary
    stockData.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Ensure the index is a datetime type (should already be by default)
    stockData.index = pd.to_datetime(stockData.index)
    
    return stockData

stockList = ['LVO.AS'] #VTS is just the ticker we want

endDate = datetime.datetime.now().replace(tzinfo=None) #Timezone out

startDate = endDate - datetime.timedelta(days=2773) #Transforms 

stockData = get_data(stockList[0], startDate, endDate)

#add the df columns to stockdata

final_data = pd.concat([stockData, df], axis=1)
final_data = final_data.dropna()

#If we just print the length of this data object right now it would be 0 since it loads as we go through the days
data = data = CustomPandasData(dataname=final_data, name='VXX')
cerebro = bt.Cerebro()
cerebro.adddata(data)

#%%
import sys

log_file = open("strategy_output.log", "w")
sys.stdout = log_file
sys.stderr = log_file


class VolatilityStrategy(bt.Strategy):
    
    def __init__(self):
        # Initialize variables and indicators
        self.order = None
        self.total_commission = 0.0
        self.times_traded = 0
        self.portfolio_values = []
        self.day0_1 = None
        self.day0_2 = None
        self.day0_3 = None              
        self.triggered1 = False
        self.triggered2 = False
        self.triggered3 = False
        self.current_size = 0
        self.dispersion_trigger_days = []     
        self.position_open_day = None
        
    def start(self):
        # Now that data feeds are fully loaded, set up the timer
        self.add_timer(
            when=bt.timer.SESSION_START,
            notify=True,
        )

    def next(self):
        self.Z_Close_Signal()
        self.Z_Vol_Spread_Signal()
        self.RSI_90()
        self.RSI_20()
        self.Dispersion_Signal()
        self.Z_Close_Low_Signal()
        self.Minimum_Holding_Period()
    
    def notify_order(self, order):
        
        stock_name = order.data._name
        
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f"Order for {order.data._name} not completed yet")

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                f'BUY EXECUTED for {stock_name}, Price: {order.executed.price:.2f}, '
                f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                f'Size: {order.executed.size:.0f}')
                
                self.total_commission += order.executed.comm
                self.times_traded += 1

            elif order.issell():
                self.log(
                f'SELL EXECUTED for {stock_name}, Price: {order.executed.price:.2f}, '
                f'Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                f'Size: {order.executed.size:.0f}')
                self.total_commission += order.executed.comm
                self.times_traded += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    
    def stop(self):
        # Final portfolio value
        final_value = self.broker.get_value()
        roi = (final_value / self.broker.startingcash - 1) * 100

        print('-' * 50)
        print('Strategy Performance:')
        print(f'Initial Cash: ${self.broker.startingcash:,.2f}')
        print(f'Final Value: ${final_value:,.2f}')
        print(f'Total Commission Paid: ${self.total_commission:,.2f}')
        print(f'ROI: {roi:.2f}%')
        print(f'Number of Trades: {self.times_traded}')
        print('-' * 50)
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def Z_Close_Signal(self):
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()
        if position < -0.6:
            return
        
        z_close = self.datas[0].z_close[0]

        # Day 0 Trigger: Z_Close > 2 and either no prior Day 0 or within 2 days of the last Day 0
        if z_close > 2 and (self.day0_1 is None or len(self) - self.day0_1 <= 2): #FIX the second part of the if statement
            self.day0_1 = len(self)  # Set or reset Day 0
            self.triggered1 = True # Mark trigger active
            self.log(f'Triggered Z_Close_Signal Day 0 at index {len(self)}')
            if position < 0:
                self.close()
            return  # No action until Day 2 or Day 3

        # Check if Day 2 is reached after trigger
        if self.triggered1 and self.day0_1 is not None:
            if len(self) - self.day0_1 >= 1 and len(self) - self.day0_1 <= 5:
                # Enter short position 40%
                self.close()
                size = 0.4 * self.broker.getvalue() / self.datas[0].close[0]  # Allocate 40% of portfolio
                self.sell(size=size)  # Short 40%  # Short position
                self.current_size = size  # Track position size
                self.position_open_day = len(self)  # Mark Day 0 for holding period
                self.day0_1 = None
                self.log(f'Z_Close_Signal: Short position opened at index {len(self)} with size {size}')
                self.triggered1 = False
                return

    def Z_Vol_Spread_Signal(self):
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()  #UNDERSTAND WHY IT DOESNT HIT
        if position > 0.5:
            return
        
        z_vol_spread = self.datas[0].z_vol_spread[0]
        
        if z_vol_spread < -2 and (self.day0_2 is None or len(self) - self.day0_2 <= 3):
            self.day0_2 = len(self)  # Set or reset Day 0
            self.triggered2 = True # Mark trigger active
            self.log(f'Triggered Z_Vol_Spread_Signal Day 0 at index {len(self)}')
            return  # No action until Day 2 or Day 3

        # Check if Day 2 is reached after trigger
        if self.triggered2 and self.day0_2 is not None:
            if len(self) - self.day0_2 == 1:
                if self.getposition().size <= 0:
                    self.close()
                    size = 0.1 * self.broker.getvalue() / self.datas[0].close[0]  # Allocate 10% of portfolio
                    self.buy(size=size)  # Long 10%  # Long position
                    self.current_size = size  # Track position size
                    self.position_open_day = len(self)  # Mark Day 0 for holding period
                    self.day0_2 = None
                    self.log(f'Z_Vol_Spread_Signal: Long position opened at index {len(self)} with size {size}')
                    return
                else:
                    size = 0.1 * self.broker.getvalue() / self.datas[0].close[0]  # add 10% of portfolio to existing long position
                    self.buy(size=size)
                    self.log(f'Z_Vol_Spread_Signal: Added long position at index {len(self)} with size {size}')
                    return
    
    def RSI_90(self):
        rsi = self.datas[0].rsi[0]
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()
        if position > 0.5:
            return
    
        # Initialize trigger days list if not present
        if not hasattr(self, 'rsi_90_trigger_days'):
            self.rsi_90_trigger_days = []
    
        # Track RSI > 90 occurrences
        if rsi > 90:
            self.log(f'RSI_90 Triggered: RSI={rsi:.2f} at index {len(self)}')
            self.rsi_90_trigger_days.append(len(self))
    
        # Keep only triggers from the last 3 days
        self.rsi_90_trigger_days = [d for d in self.rsi_90_trigger_days if len(self) - d <= 3]
    
        # If RSI > 90 occurred at least twice in the last 3 days, buy 0.6
        if len(self.rsi_90_trigger_days) >= 2:
            if self.getposition().size <= 0:  # If in a short or no position, close and buy 0.6
                self.close()
                size = 0.6 * self.broker.getvalue() / self.datas[0].close[0]  # Allocate 60% of portfolio
                self.buy(size=size)
                self.position_open_day = len(self)  # Mark Day 0 for holding period
                self.log(f'RSI_90: Long position opened with size {size}')
            elif self.getposition().size > 0:  # Already long, ensure total position size is 0.6
                self.close()  # Close existing position to reset to exactly 0.6
                size = 0.6 * self.broker.getvalue() / self.datas[0].close[0]  # Allocate 60% of portfolio
                self.buy(size=size)
                self.log(f'RSI_90: Adjusted long position to 0.6 with size {size}')


    def RSI_20(self):
        rsi = self.datas[0].rsi[0]
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()
        if position < -0.59:
            return
    
        # Trigger Day 0
        if rsi < 20:
            if self.day0_3 is None or len(self) - self.day0_3 > 1:  # Allow retrigger if previous trigger expired
                self.day0_3 = len(self)  # Set or reset Day 0
                self.triggered3 = True  # Mark trigger active
                self.log(f'RSI_20 Triggered at index {len(self)} - Waiting for Day 1 confirmation.')
            return  # No action until Day 1
    
        # Check if Day 2 is reached after trigger
        if self.triggered3 and self.day0_3 is not None:
            if len(self) - self.day0_3 == 1:
                self.log(f'RSI_20 Confirmed at index {len(self)} - Opening short position.')
                self.close()  # Close any existing positions
                size = 0.3 * self.broker.getvalue() / self.datas[0].close[0]  # Allocate 30% of portfolio
                self.sell(size=size)  # Short 30%  # Short position
                self.current_size = size  # Track position size
                self.position_open_day = len(self)  # Mark Day 0 for holding period
                self.day0_3 = None  # Reset Day 0 after execution
                self.triggered3 = False  # Reset trigger flag
                self.log(f'RSI_20: Short position opened with size {size}')

    def Dispersion_Signal(self):
        z_dispersion = self.datas[0].z_dispersion[0]
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()
        if position >= 0.55:
            return
        
        if z_dispersion < -1.05:
            self.log(f'Dispersion_Signal Triggered at index {len(self)} - Checking conditions.')
            self.dispersion_trigger_days.append(len(self))

            # Keep only last 5 days
            self.dispersion_trigger_days = [d for d in self.dispersion_trigger_days if len(self) - d <= 5]

            if len(self.dispersion_trigger_days) >= 2 and self.getposition().size > 0:
                size = 0.1* self.broker.getvalue() / self.datas[0].close[0]
                self.buy(size=size)
                self.log(f"Dispersion_Signal: Added Long Position. New size={size:.2f} at index {len(self)}")

    def Z_Close_Low_Signal(self):
        z_close = self.datas[0].z_close[0]
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()
        if position <= 0.6:
            return
        
        if z_close < -1.05 and (any(self.datas[0].rsi[-i] > 90 for i in range(1, 11)) or any(self.datas[0].z_vol_spread[-i] < -2 for i in range(1, 11))) and self.getposition().size > 0:
            size = 0.05 * self.broker.getvalue() / self.datas[0].close[0]
            self.buy(size=size)
            self.log(f'Z_Close_Low_Signal: Added long position with size {size}')

    def Minimum_Holding_Period(self):
        self.portfolio_values.append(self.broker.get_value())
        
        pct_return = 0
        
        
        if self.position_open_day is not None:
            holding_days = len(self) - self.position_open_day
            entry_price = self.datas[0].close[-holding_days]
            current_price = self.datas[0].close[0]
            
            if self.getposition().size > 0:  # Long position
                pct_return = (current_price - entry_price) / entry_price
            else:  # Short position
                pct_return = (entry_price - current_price) / entry_price
        
        
        position = self.getposition().size * self.datas[0].close[0] / self.broker.getvalue()
        self.log(f'current position = {position}, return = {pct_return}')

        
        if self.position_open_day is None:
            return
        
        if holding_days < 6:
            return
        
        # Close position if after day 14 and profit >= 10%
        
        if holding_days >= 10 and pct_return >= 0.09:
            self.close()
            self.position_open_day = None
            self.log(f'Minimum Holding Period: Position closed at index {len(self)} with return {pct_return:.2%}')
            return
        
        if holding_days < 11 and pct_return >= 0.04:
            self.close()
            self.position_open_day = None
            self.log(f'Minimum Holding Period: Position closed at index {len(self)} with return {pct_return:.2%}')
            
#%%

cerebro.addstrategy(VolatilityStrategy)

# Broker Information
broker_args = dict(coc=False)
cerebro.broker = bt.brokers.BackBroker(**broker_args)
cerebro.broker.set_coo(True)

cerebro.broker.set_cash(100000)

strategies = cerebro.run()
strategy = strategies[0]

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()

#%%

import numpy as np

# Calculate daily returns
portfolio_values = strategy.portfolio_values  # Replace with your strategy instance
daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]  # (Pn - Pn-1) / Pn-1
volatility = np.std(daily_returns)
print(f"Strategy Volatility (daily): {volatility:.6f}")
annualized_volatility = volatility * np.sqrt(252)
print(f"Annualized Volatility: {annualized_volatility:.6f}")

import pandas as pd
import matplotlib.pyplot as plt

end_date = datetime.datetime.now().replace(tzinfo=None)
# We have 239 values, so we need 239 dates.
# Go back 238 days from end_date. We'll assume consecutive days.
dates = pd.date_range(end=end_date, periods=len(portfolio_values), freq='B')  
# freq='B' stands for business days (weekdays), which is often close to trading days.

# Create a DataFrame with portfolio_values
df = pd.DataFrame({'PortfolioValue': portfolio_values}, index=dates)

# Plotting the portfolio value over time
plt.figure()
df['PortfolioValue'].plot()
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.show()

#%%
import statsmodels.api as sm

benchmark_sub = get_data('^GSPC', startDate, endDate)
benchmark = get_data('^VIX', startDate, endDate)

benchmark_daily_returns = benchmark['Adj Close'].pct_change().dropna().values

# Ensure daily_returns aligns with benchmark data
aligned_returns = min(len(daily_returns), len(benchmark_daily_returns))
strategy_returns = daily_returns[:aligned_returns]
benchmark_returns = benchmark_daily_returns[:aligned_returns]

# Calculate Beta and Alpha
X = sm.add_constant(benchmark_returns)  # Add intercept term
y = strategy_returns  # Dependent variable (strategy returns)

# Fit the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Print the summary table
print(model.summary())

# Max Drawdown Calculation
cumulative_returns = np.cumprod(1 + daily_returns)  # Convert daily returns to cumulative returns
cumulative_max = np.maximum.accumulate(cumulative_returns)  # Running max of cumulative returns
drawdowns = cumulative_returns / cumulative_max - 1  # Drawdowns as a percentage
max_drawdown = drawdowns.min()  # Minimum drawdown is the max drawdown

tolerance = 1e-9  # Small tolerance for floating-point precision
flat_periods = np.isclose(np.diff(cumulative_returns, prepend=cumulative_returns[0]), 0, atol=tolerance)

# Set drawdowns to 0 during flat periods
drawdowns[flat_periods] = 0

print(f"Max Drawdown: {max_drawdown:.4%}")

# Plot cumulative returns and drawdowns
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(cumulative_returns, label='Cumulative Returns')
plt.plot(cumulative_max, label='Cumulative Max', linestyle='--')
plt.plot(np.cumprod(1 + benchmark_returns), label='S&P 500')
plt.title('Cumulative Returns and Maximum Drawdown')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(drawdowns, label='Drawdowns', color='red')
plt.axhline(0, linestyle='--', color='black', linewidth=0.8)
plt.title('Drawdowns')
plt.legend()

plt.tight_layout()
plt.show()



