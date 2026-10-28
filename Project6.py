# Time - Series Strategy with Walk - Forward Validation

import numpy as np
import pandas as pd

buy_cost, sell_cost = 0.0022, 0.0021
rf_monthly = 1.065**(1/12) - 1

# Step 1: Score every stock
def ma_score(prices,params):
       fast, slow = params
       return prices.rolling(fast).mean()/prices.rolling(slow).mean() - 1
       
def mr_score(prices,params):
       lookback, entry_z = params
       z = (prices - prices.rolling(lookback).mean())/prices.rolling(lookback).std()
       return -z-entry_z
       
# Step 2 : Choose stocks and give equal weights
def pick_weights(score_row):
       s = score_row.dropna()
       if len(s)<50:
              return pd.Series(0.0, index = score_row.index)
       weights = pd.Series(0.0, index = score_row.index)
       on = s.index[s>0]
       if len(on) > 0 :
              weights[on] = 1.0/len(on)
       return weights
       
def build_weights(prices, score_fn, params):
       scores = score_fn(prices, params)
       return pd.DataFrame([pick_weights(scores.loc[d]) for d in scores.index], index = scores.index, columns = scores.columns)
       
# Step 3 : Simulate month by month (Backtesting)
def backtest(weights, next_returns):
       held = pd.Series(0.0, index = weights.columns)
       out = []
       for date in weights.index[:-1]:
              target = weights.loc[date]
              if target.isna().all():
                     held = held * 0
                     continue
              target = target.fillna(0)
              trade = target - held
              bought = trade[trade>0].sum()
              sold = -trade[trade<0].sum()
              cost = bought * buy_cost + sold * sell_cost
              r = next_returns.loc[date].fillna(0)
              invested = target.sum()
              gross = (target * r).sum() + (1-invested)*rf_monthly 
              held = target * (1+r)/(1+gross)
              out.append((date,gross,cost,gross-cost,bought,invested))
       return pd.DataFrame(out, columns=["date","gross","cost","net","turnover","invested"]).set_index("date")
       
# Step 4 : Summary numbers
def sharpe(r):
       r = r.dropna()
       ex = r - rf_monthly
       return ex.mean()/ex.std() * np.sqrt(12)
       
def summary(r):
       years = len(r)/12
       nav = (1+r).cumprod()
       ex = r - rf_monthly
       return { "CAGR": nav.iloc[-1]**(1/years) - 1, 
                "Vol": ex.std()*np.sqrt(12),
                "Sharpe": sharpe(r),
                "MaxDD": (nav/nav.cummax() - 1).min(),
                "WinRate": (r>0).mean()
               }
               
# Step 5 : Walk Forward Validation
def walk_forward(prices, next_returns, score_fn, grid, train = 60, test = 12):
       weights = {a : build_weights(prices, score_fn, a) for a in grid}
       results = {a : backtest(weights[a],next_returns) for a in grid}
       months = results[grid[0]].index
       for a in grid:
              months = months.intersection(results[a].index)
       stitched = pd.DataFrame(np.nan , index = prices.index, columns = prices.columns)
       folds = []
       start = train
       
       while start < len(months):
              train_months = months[start-train:start]
              test_months = months[start:min(start+test,len(months))]
              best = max(grid, key = lambda a : sharpe(results[a].loc[train_months,"net"]))
              stitched.loc[test_months] = weights[best].loc[test_months].values
              folds.append({"test_start" : test_months[0].strftime("%Y-%m"),
                            "test_end" : test_months[-1].strftime("%Y-%m"),
                            "test_year" : test_months[0].year,
                            "chosen" : str(best),
                            "train_sharpe" : sharpe(results[best].loc[train_months,"net"]),
                            "test_months" : len(test_months)
                           }
                          )
              start += test
       oos = backtest(stitched, next_returns)
       oos.loc[months[train:]]
       for f in folds: 
              seg = oos.loc[oos.index.year == f["test_year"],"net"]
              f["test_return"] = (1+seg).prod()-1      
              f["test_sharpe"] = sharpe(seg) if len(seg)>2 else np.nan
       return oos, pd.DataFrame(folds), results   
       
# Step 6 : Strategy definitiond
strats = {
          "MA crossover" : (ma_score, [(fast,slow) for fast in (2,3,4) for slow in (6,9,12) if fast < slow]),
          "Mean reversion" : (mr_score, [(lookback,entry_z) for lookback in (6,9,12) for entry_z in (0.5,1.0,1.5)])
         }
          
# Step 7 : Run the complete experiment
def run(prices):
       next_returns = prices.pct_change().shift( -1)
       all_results = {}
       for name, (score_fn,grid) in strats.items():
              oos, folds, results = walk_forward(prices,next_returns,score_fn,grid)
              all_results[name] = (oos,folds,results)
              print(f"\n{name}")
              print("\nOut-of-sample performance")
              print(pd.Series(summary(oos["net"])).round(3).to_string())
              print(f"\nAverage monthly turnover:" f"{oos['turnover'].mean():.1%}")
              print(f"\nAverage annual transaction cost:" f"{oos['cost'].mean() * 12 :.2%}")
              print(f"\nAverage invested fraction:" f"{oos['invested'].mean():.0%}")
              print("\nWalk Forward folds:")
              print(folds.round(3).to_string(index=False))
              if len(folds) > 1:
                     print(f"\nMean train sharpe:" f"{folds['train_sharpe'].mean():.2f}")
                     print(f"\nMean test sharpe:" f"{folds['test_sharpe'].mean():.2f}")
                     print(f"\nTrain-test sharpe correlation:" f"{folds['train_sharpe'].corr(folds['test_sharpe']):.2f}")
              sensitivity = pd.Series({params : sharpe(results[params]["net"]) for params in grid})
              print("\nParameter sensitivity (full-sample sharpe) :")
              print(sensitivity.unstack().round(3).to_string())
       return all_results
       
# Step 8 : Load NIFTY 500 stock list
import os
import yfinance as yf

my_tickers = (pd.read_csv("ind_nifty500list.csv")["Symbol"].str.strip().tolist())

if os.path.exists("daily.csv"):
    daily = pd.read_csv("daily.csv",index_col=0, parse_dates=True)
else:
    daily = yf.download([ticker + ".NS"for ticker in my_tickers],start="2005-01-01",auto_adjust=True,progress=False)["Close"]
    daily.to_csv("daily.csv")

try:
    prices = daily.resample("ME").last()      # pandas 2.2 and newer
except ValueError:
    prices = daily.resample("M").last()       # older pandas (your Python 3.8 environment)
    
if daily.index[-1] < prices.index[-1]:
    prices = prices.iloc[:-1]

prices = prices.where(prices >= 10)

# Step 9 : Remove stocks with no usable data
prices = prices.dropna(axis=1,how="all")

# Step 10 : Display the final dataset
print(
    f"{prices.shape[1]} stocks, "
    f"{len(prices)} months, "
    f"{prices.index[0].date()} to "
    f"{prices.index[-1].date()}"
)
run(prices) 
