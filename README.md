
## Predicting the stock market



I will be working with data from the `S&P500 Index`. The `S&P500` is a stock market index. Some companies are publicly traded, which means that anyone can buy and sell their shares on the open market. A share entitles the owner to some control over the direction of the company, and to some percentage (or share) of the earnings of the company. When you buy or sell shares, it's common to say that you're trading a stock.

The price of a share is based mainly on supply and demand for a given stock. For example, Apple stock has a price of 120 dollars per share as of December 2015 -- http://www.nasdaq.com/symbol/aapl. A stock that is in less demand, like Ford Motor Company, has a lower price -- http://finance.yahoo.com/q?s=F. Stock price is also influenced by other factors, including the number of shares a company has issued.

Stocks are traded daily, and the price can rise or fall from the beginning of a trading day to the end based on demand. Stocks that are in more in demand, such as Apple, are traded more often than stocks of smaller companies.

Indexes aggregate the prices of multiple stocks together, and allow you to see how the market as a whole is performing. For example, the Dow Jones Industrial Average aggregates the stock prices of 30 large American companies together. The S&P500 Index aggregates the stock prices of 500 large companies. When an index fund goes up or down, you can say that the underlying market or sector it represents is also going up or down. For example, if the Dow Jones Industrial Average price goes down one day, you can say that American stocks overall went down (ie, most American stocks went down in price).

I'll be using historical data on the price of the `S&P500` Index to make predictions about future prices. Predicting whether an index will go up or down will help us forecast how the stock market as a whole will perform. Since stocks tend to correlate with how well the economy as a whole is performing, it can also help us make economic forecasts.

There are also thousands of traders who make money by buying and selling Exchange Traded Funds. ETFs allow you to buy and sell indexes like stocks. This means that you could "buy" the `S&P500` Index ETF when the price is low, and sell when it's high to make a profit. Creating a predictive model could allow traders to make money on the stock market.


In this project, I'll be working with a csv file containing index prices. Each row in the file contains a daily record of the price of the S&P500 Index from 1950 to 2015. The dataset is stored in `sphist.csv`.

The columns of the dataset are:

`Date` -- The date of the record.

`Open` -- The opening price of the day (when trading starts).

`High` -- The highest trade price during the day.

`Low` -- The lowest trade price during the day.

`Close` -- The closing price for the day (when trading is finished).

`Volume` -- The number of shares traded.

`Adj Close` -- The daily closing price, adjusted retroactively to include any corporate actions. Read more here.

I'll be using this dataset to develop a predictive model,  train the model with data from 1950-2012, and try to make predictions from 2013-2015.


```python
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

### Exploring and Cleaning the Data


```python
sphist = pd.read_csv("sphist.csv")
```


```python
sphist.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-12-07</td>
      <td>2090.419922</td>
      <td>2090.419922</td>
      <td>2066.780029</td>
      <td>2077.070068</td>
      <td>4.043820e+09</td>
      <td>2077.070068</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-04</td>
      <td>2051.239990</td>
      <td>2093.840088</td>
      <td>2051.239990</td>
      <td>2091.689941</td>
      <td>4.214910e+09</td>
      <td>2091.689941</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-03</td>
      <td>2080.709961</td>
      <td>2085.000000</td>
      <td>2042.349976</td>
      <td>2049.620117</td>
      <td>4.306490e+09</td>
      <td>2049.620117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-02</td>
      <td>2101.709961</td>
      <td>2104.270020</td>
      <td>2077.110107</td>
      <td>2079.510010</td>
      <td>3.950640e+09</td>
      <td>2079.510010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-01</td>
      <td>2082.929932</td>
      <td>2103.370117</td>
      <td>2082.929932</td>
      <td>2102.629883</td>
      <td>3.712120e+09</td>
      <td>2102.629883</td>
    </tr>
  </tbody>
</table>
</div>




```python
sphist['Date']= pd.to_datetime(sphist['Date'])
sphist_sorted= sphist.sort_values('Date', ascending=True)
sphist_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16589</th>
      <td>1950-01-03</td>
      <td>16.66</td>
      <td>16.66</td>
      <td>16.66</td>
      <td>16.66</td>
      <td>1260000.0</td>
      <td>16.66</td>
    </tr>
    <tr>
      <th>16588</th>
      <td>1950-01-04</td>
      <td>16.85</td>
      <td>16.85</td>
      <td>16.85</td>
      <td>16.85</td>
      <td>1890000.0</td>
      <td>16.85</td>
    </tr>
    <tr>
      <th>16587</th>
      <td>1950-01-05</td>
      <td>16.93</td>
      <td>16.93</td>
      <td>16.93</td>
      <td>16.93</td>
      <td>2550000.0</td>
      <td>16.93</td>
    </tr>
    <tr>
      <th>16586</th>
      <td>1950-01-06</td>
      <td>16.98</td>
      <td>16.98</td>
      <td>16.98</td>
      <td>16.98</td>
      <td>2010000.0</td>
      <td>16.98</td>
    </tr>
    <tr>
      <th>16585</th>
      <td>1950-01-09</td>
      <td>17.08</td>
      <td>17.08</td>
      <td>17.08</td>
      <td>17.08</td>
      <td>2520000.0</td>
      <td>17.08</td>
    </tr>
  </tbody>
</table>
</div>




```python
sphist_sorted.shape
```




    (16590, 7)



### Generating Indicators

Here are some indicators that are interesting to generate for each row:

- The average price from the past 5 days.
- The average price for the past 30 days.
- The average price for the past 365 days.
- The ratio between the average price for the past 5 days, and the average price for the past 365 days.
- The standard deviation of the price over the past 5 days.
- The standard deviation of the price over the past 365 days.
- The ratio between the standard deviation for the past 5 days, and the standard deviation for the past 365 days.


```python


sphist_sorted["day_5"] = sphist_sorted['Close'].shift(1).rolling(center=False, window=5).mean()
sphist_sorted["year_1"] = sphist_sorted['Close'].shift(1).rolling(center=False, window=365).mean()
sphist_sorted["day_year_ratio"] = sphist_sorted["day_5"] / sphist_sorted["year_1"]
sphist_sorted["day_5_std"] = sphist_sorted['Close'].shift(1).rolling(center=False, window=5).std()
sphist_sorted["year_1_std"] = sphist_sorted['Close'].shift(1).rolling(center=False, window=365).std()
sphist_sorted["day_year_std_ratio"] = sphist_sorted["day_5_std"] / sphist_sorted["year_1_std"]

```


```python
sphist_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>day_5</th>
      <th>year_1</th>
      <th>day_year_ratio</th>
      <th>day_5_std</th>
      <th>year_1_std</th>
      <th>day_year_std_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16589</th>
      <td>1950-01-03</td>
      <td>16.66</td>
      <td>16.66</td>
      <td>16.66</td>
      <td>16.66</td>
      <td>1260000.0</td>
      <td>16.66</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16588</th>
      <td>1950-01-04</td>
      <td>16.85</td>
      <td>16.85</td>
      <td>16.85</td>
      <td>16.85</td>
      <td>1890000.0</td>
      <td>16.85</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16587</th>
      <td>1950-01-05</td>
      <td>16.93</td>
      <td>16.93</td>
      <td>16.93</td>
      <td>16.93</td>
      <td>2550000.0</td>
      <td>16.93</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16586</th>
      <td>1950-01-06</td>
      <td>16.98</td>
      <td>16.98</td>
      <td>16.98</td>
      <td>16.98</td>
      <td>2010000.0</td>
      <td>16.98</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16585</th>
      <td>1950-01-09</td>
      <td>17.08</td>
      <td>17.08</td>
      <td>17.08</td>
      <td>17.08</td>
      <td>2520000.0</td>
      <td>17.08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
sphist_sorted.shape
```




    (16590, 13)



 Some of the indicators use `365` days of historical data, and the dataset starts on `1950-01-03`. Thus, any rows that fall before `1951-01-03` don't have enough historical data to compute all the indicators. There is need to remove these rows before you split the data.


```python
sphist_sorted=  sphist_sorted[sphist_sorted['Date']> datetime(year= 1951, month= 1,  day= 2)]

# Use the dropna method to remove any rows with NaN values. Pass in the axis=0 argument to drop rows.
sphist_sorted.dropna(axis = 0, inplace = True)
sphist_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>day_5</th>
      <th>year_1</th>
      <th>day_year_ratio</th>
      <th>day_5_std</th>
      <th>year_1_std</th>
      <th>day_year_std_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16224</th>
      <td>1951-06-19</td>
      <td>22.020000</td>
      <td>22.020000</td>
      <td>22.020000</td>
      <td>22.020000</td>
      <td>1100000.0</td>
      <td>22.020000</td>
      <td>21.800</td>
      <td>19.447726</td>
      <td>1.120954</td>
      <td>0.256223</td>
      <td>1.790253</td>
      <td>0.143121</td>
    </tr>
    <tr>
      <th>16223</th>
      <td>1951-06-20</td>
      <td>21.910000</td>
      <td>21.910000</td>
      <td>21.910000</td>
      <td>21.910000</td>
      <td>1120000.0</td>
      <td>21.910000</td>
      <td>21.900</td>
      <td>19.462411</td>
      <td>1.125246</td>
      <td>0.213659</td>
      <td>1.789307</td>
      <td>0.119409</td>
    </tr>
    <tr>
      <th>16222</th>
      <td>1951-06-21</td>
      <td>21.780001</td>
      <td>21.780001</td>
      <td>21.780001</td>
      <td>21.780001</td>
      <td>1100000.0</td>
      <td>21.780001</td>
      <td>21.972</td>
      <td>19.476274</td>
      <td>1.128142</td>
      <td>0.092574</td>
      <td>1.788613</td>
      <td>0.051758</td>
    </tr>
    <tr>
      <th>16221</th>
      <td>1951-06-22</td>
      <td>21.549999</td>
      <td>21.549999</td>
      <td>21.549999</td>
      <td>21.549999</td>
      <td>1340000.0</td>
      <td>21.549999</td>
      <td>21.960</td>
      <td>19.489562</td>
      <td>1.126757</td>
      <td>0.115108</td>
      <td>1.787659</td>
      <td>0.064390</td>
    </tr>
    <tr>
      <th>16220</th>
      <td>1951-06-25</td>
      <td>21.290001</td>
      <td>21.290001</td>
      <td>21.290001</td>
      <td>21.290001</td>
      <td>2440000.0</td>
      <td>21.290001</td>
      <td>21.862</td>
      <td>19.502082</td>
      <td>1.121008</td>
      <td>0.204132</td>
      <td>1.786038</td>
      <td>0.114293</td>
    </tr>
  </tbody>
</table>
</div>



## Generating and Training Data


```python
train= sphist_sorted[sphist_sorted['Date'] < datetime(year= 2013, month= 1,  day= 1)]
test= sphist_sorted[sphist_sorted['Date'] >= datetime(year= 2013, month= 1,  day= 1)]
```


```python
train.shape
```




    (15486, 13)




```python
test.shape
```




    (739, 13)



## Training a Linear Regression Model


```python
features=['day_5','year_1','day_year_ratio','day_5_std','year_1_std','day_year_std_ratio']
target=['Close']
```


```python
lr=LinearRegression()
lr.fit(train[features], train[target])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



## Making Predictions


```python
predictions=lr.predict(test[features])
```


```python
rmse=mean_squared_error(test["Close"], predictions) ** (1/2)
rmse
```




    22.151803990066828




```python

```
