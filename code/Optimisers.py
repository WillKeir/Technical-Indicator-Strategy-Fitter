# Imports
import numpy as np
import pandas as pd
import Indicators as ic
import PerformanceMetrics as pm
import copy

from binance.client import Client as bnb_client
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


# BNB Client
client = bnb_client()

# Binance data retrieval function
def get_binance_px(symbol, freq, start_ts, end_ts):
    data = client.get_historical_klines(symbol,freq,start_ts,end_ts)
    columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
    'num_trades','taker_base_volume','taker_quote_volume','ignore']

    data = pd.DataFrame(data,columns = columns)
    
    # Convert from POSIX timestamp (number of millisecond since jan 1, 1970)
    data['open_time'] = data['open_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    data['close_time'] = data['close_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    return data 

# get_data function. Used to source data
def get_data(univ, freq, start_ts, end_ts):
    frames = []
    for x in univ:
        data = get_binance_px(x,freq,start_ts, end_ts)
        data = data.set_index('open_time')[['open', 'high', 'low', 'close', 'volume']]
        data.columns = pd.MultiIndex.from_product([[x], data.columns])
        frames.append(data)

    px = pd.concat(frames, axis=1).astype(float)
    px = px.reindex(pd.date_range(px.index[0],px.index[-1],freq=freq))
    px.index = pd.to_datetime(px.index, utc=True)

    px = px[univ[0]].dropna()

    return px


# Backtest Function (with filters)
def backtest(data, indicator_names, long_filters, short_filters, indicator_params, mode, output_metric, backtest_startdate='2018-01-01', backtest_enddate=None, optim=False):
    # Clean Data
    if not optim:
        data = pm.data_cleaning(data)

    # Aggregate indicator signals
    TPI_Signals = ic.Aggregator(data, indicator_names, indicator_params).combine_signals()
    
    if len(long_filters) > 0:
        long_filter_signals = ic.Aggregator(data, long_filters, indicator_params).combine_signals()
    else:
        long_filter_signals = pd.Series(0.0, index=data.index)


    if len(short_filters) > 0:
        short_filter_signals = ic.Aggregator(data, short_filters, indicator_params).combine_signals()
    else:
        short_filter_signals = pd.Series(0.0, index=data.index)

    # If TPI_Signals is long and long filter is long, go long
    # If TPI_Signals is short and short filter is short, go short
    # If no filters in a particular, don't change signal from TPI_Signals for that direction
    # Otherwise, set TPI to 0 (no change to current position)
    test_filter = ((TPI_Signals > 0) & ((long_filter_signals > 0) | (len(long_filters) == 0)) | ((TPI_Signals < 0) & ((short_filter_signals < 0) | (len(short_filters) == 0))))

    TPI = np.where(
        test_filter,
        TPI_Signals, 0)
    
    TPI = pd.Series(TPI, index=data.index)
                   
    # Process signals to generate trades and equity curve
    data_optim = pm.TPI(data, TPI, backtest_startdate, backtest_enddate)
    trades_optim = pm.trades(data_optim, mode)
    equity_optim = pm.equity(data_optim, trades_optim)

    metrics = pm.PerformanceMetrics(data_optim, equity_optim, trades_optim, mode)
    
    # Return results
    if output_metric == 'MetricsTable':
        return metrics.MetricsTable()
    elif output_metric == 'Sharpe':
        return metrics.sharpe_ratio(), None
    elif output_metric == 'Sortino':
        return metrics.sortino_ratio(), None
    elif output_metric == 'Profit Factor':
        return metrics.profit_factor(), None
    elif output_metric == 'Omega':
        return metrics.omega_ratio(), None
    elif output_metric == 'Win Rate':
        return metrics.percent_profitable(), None
    elif output_metric == 'Net Profit':
        return metrics.net_profit_pct(), None
    elif output_metric == 'Slap Score':
        ss = metrics.slapScore()
        netp = metrics.net_profit_pct()
        return ss, netp
    elif output_metric == 'tradesDF':
        return trades_optim
    elif output_metric == 'equityDF':
        return equity_optim
    else:
        return None

# Bayes Optimisation Function
def Engine_Bayes(data, iterations, indicator_names, long_filters, short_filters, mode, output_metric='Slap Score', backtest_startdate='2018-01-01', backtest_enddate=None):

    # Initialize variables for optimization
    best_params = {}
    best_score1 = -np.inf
    best_score2 = -np.inf

    # Complete Iterations
    for i in range(iterations):

        backtest_params = {ind: {} for ind in list(pd.unique((indicator_names + long_filters + short_filters)))}

        # Randomise indicator inputs
        for indicator in list(pd.unique((indicator_names + long_filters + short_filters))):
            indClass = ic.INDICATOR_CLASSES[indicator]
            for input in indClass.bounds.items():
                # check for int
                if isinstance(input[1][0], int):
                    value = np.random.randint(*indClass.bounds[input[0]])

                elif isinstance(input[1][0], float):
                    value = round(np.random.uniform(*indClass.bounds[input[0]]), 1)

                elif isinstance(input[1][0], str):
                    value = np.random.choice(list(indClass.bounds[input[0]]))
                    
                # Place inputs into dictionary
                backtest_params[indicator][input[0]] = value
                
            # Check for conflicting inputs. Adjuct inputs to remove conflicts
            backtest_params[indicator] = indClass.check_inputs_Bayes(backtest_params[indicator])

        # Test inputs against dataset
        score1, score2 = backtest(data, indicator_names, long_filters, short_filters, backtest_params, mode, output_metric, backtest_startdate, backtest_enddate, optim=True)

        # Record best performer
        # If using slap score as metric, use net profit as secondary metric
        if output_metric == 'Slap Score':
            if score1 == best_score1:
                if score2 > best_score2:
                    best_score2 = score2
                    best_params = copy.deepcopy(backtest_params)

            elif score1 > best_score1:
                best_score1 = score1
                best_score2 = score2
                best_params = copy.deepcopy(backtest_params)
                
        # If not using slap score, ignore above condition
        elif score1 > best_score1:
            best_score1 = score1
            best_params = copy.deepcopy(backtest_params)
        
        # Print to track progress
        if (i+1)%5 == 0:
            print("BAYESIAN: iteration " + str(i+1) + "/" + str(iterations) + " complete")
    
    # Produce Metrics of the best performer
    final_score = backtest(data, indicator_names,long_filters, short_filters, 
                                 best_params, mode, 
                                 output_metric='MetricsTable', 
                                 backtest_startdate=backtest_startdate, 
                                 backtest_enddate=backtest_enddate, 
                                 optim=True)

    return best_params, final_score


# Bayes Optimiser Function
def Optimiser_Bayes(data, iterations, indicator_names, long_filters, short_filters, mode, output_metric='Slap Score', backtest_startdate='2018-01-01', backtest_enddate=None):
     
    # Clean Data
    data = pm.data_cleaning(data)

    # Start Bayesian Optimisation (Round 1)
    print("\nSTARTING BAYESIAN OPTIMISATION\n")
    bayes_params, bayes_metrics = Engine_Bayes(data, iterations, indicator_names, long_filters, short_filters, mode, output_metric, backtest_startdate, backtest_enddate)
    
    # Print Results
    print("\nRESULTS:")
    print(f"\n{bayes_metrics}\n")
    
    for ind in list(pd.unique((indicator_names + long_filters + short_filters))):
        print(ind + ":", bayes_params[ind])

    return bayes_params, bayes_metrics


