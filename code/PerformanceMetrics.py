# Import Libraries
import numpy as np
import pandas as pd
import math

# Clean data for use
def data_cleaning(data):
    
    data = data.copy()

    data = data[data.index >= pd.to_datetime('2017-01-01', utc=True)]
    data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['hl2'] = (data['high'] + data['low']) / 2
    data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
    data['hlcc4'] = (data['high'] + data['low'] + 2 * data['close']) / 4
    data['diff'] = data['close'].pct_change()
    data = data.iloc[:-1]

    return data


# TPI Setup. Add trade signals to data. Only include data in the specified date range
def TPI(data, TPI_Signals, backtest_startdate, backtest_enddate = None):
    # Merge data
    data['TPI'] = TPI_Signals

    # Inclusion of backtest_enddate
    if backtest_enddate == None:
        data = data[data.index >= pd.to_datetime(backtest_startdate, utc=True)]
    
    else:
        data = data[(data.index >= pd.to_datetime(backtest_startdate, utc=True)) & (data.index < pd.to_datetime(backtest_enddate, utc=True))]
    
    return data

# Trades
def trades(data, mode='LS'):
    # mode can take values 
    ## 'L': Long only
    ## 'S': Short only
    ## 'LS': Long/Short Perpetual

    # Set-up
    trades = []
    entry_price = None
    exit_price = None
    entry_time = None
    exit_time = None
    returns = None
    current = None
    
    # Perform iteratively
    for i in range(len(data) - 1):
    
        # Find first trade entry
        if current == None or current == 'cash':
            # Check for first short trade
            if data['TPI'].iloc[i] > 0 and mode != 'S':
                current = 'long'
                entry_price = data['close'].iloc[i]
                entry_time = data.index[i+1]
    
            # Check for first long trade
            elif data['TPI'].iloc[i] < 0 and mode != 'L': 
                current = 'short'
                entry_price = data['close'].iloc[i]
                entry_time = data.index[i+1]
    
            else:
                continue
    
        # Closing a trade
        if current != 'cash' and current != None:
            # Detect close condition. Add data
            if (data['TPI'].iloc[i] < 0 and current == 'long') or (data['TPI'].iloc[i] > 0 and current == 'short'):
                exit_price = data['close'].iloc[i]
                exit_time = data.index[i+1]
        
                returns = (exit_price - entry_price) / entry_price

                # Invert returns for short trades
                if current == 'short':
                    returns = -returns
                
                trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'position': current,
                        'returns': returns
                    })
                
                entry_price = exit_price
                entry_time = exit_time
                exit_price = None
                exit_time = None
                returns = None
        
                # Flip position for 'LS' strategies. Move into cash for 'L' and 'S' strategies
                if current == 'long':
                    current = 'short' if mode == 'LS' else 'cash'
        
                elif current == 'short':
                    current = 'long' if mode == 'LS' else 'cash'

        
    # Append open trade
    if current != None and current != 'cash':
        trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'position': current,
                        'returns': returns
                        })
    
    # Additional Adjustments
    trades = pd.DataFrame(trades)
    trades['equity'] = np.cumprod(1 + trades['returns'])
    trades['diff'] = trades['equity'].diff()
    trades['diff'] = trades['diff'].fillna(trades['returns'])
    
    return trades

# Equity
def equity(data, trades):
    # Initialise Variables
    gross_profit = 1
    open_profit = 0
    position = None
    in_trade = False
    t_index = 0

    equity_curve = np.ones(len(data))
    gross_profit2 = np.ones(len(data))
    open_profit2 = np.ones(len(data))

    move = 0

    # Compute iteratively
    for i in range(len(data)):
        
        # Check for new trade
        if t_index < len(trades) and data.index[i] == trades['entry_time'].iloc[t_index]:
            
            # Close previous position if already in a trade  
            if in_trade:  
                gross_profit *= (data['close'].iloc[i] / trades['entry_price'].iloc[t_index - 1]) ** move  

            # Extract data and set variables
            in_trade = True
            position = trades['position'].iloc[t_index]
            entry_price = trades['entry_price'].iloc[t_index]
            move = 1 if position == 'long' else -1

        # Calculate open profit if in a trade
        if in_trade:
            open_profit = gross_profit * ((data['close'].iloc[i] / entry_price) - 1) * move

        # Check for trade exit
        if t_index < len(trades) and data.index[i] == trades['exit_time'].iloc[t_index]:

            # Update Variables
            gross_profit = trades['equity'].iloc[t_index]  # Update profit at exit
            open_profit = 0
            t_index += 1  # Move to next trade

            # Immediately check if another trade opens right away
            if t_index < len(trades) and trades['entry_time'].iloc[t_index] == data.index[i]:
                
                # Extract Data
                entry_price = trades['entry_price'].iloc[t_index]
                position = trades['position'].iloc[t_index]
                move = 1 if position == 'long' else -1
            
            # If no immediate new trade, change variable
            else:
                in_trade = False 

        # Store changes
        equity_curve[i] = gross_profit + open_profit
        gross_profit2[i] = gross_profit
        open_profit2[i] = open_profit

    # Store full data in one dataframe
    equity = pd.DataFrame(
        {
            'equity': equity_curve,
            'gross': gross_profit2,
            'open': open_profit2
        }, 
        index = data.index
    )

    return equity


# Performance Metrics Class
class PerformanceMetrics:
    def __init__(self, data_df, equity_df, trades_df, mode):
        self.data = data_df 
        self.equity = equity_df
        self.trades = trades_df        
        self.mode = mode
    
    # Equity Max DD
    def equity_maxDD(self):
        
        equity = self.equity['equity']
        running_max = equity.cummax()
        drawdown = (running_max - equity) / running_max
        max_dd = drawdown.max()

        return round(100 * max_dd, 2)

    # Intra-trade Max DD
    def intra_max_DD(self):

        max_DD = 0.0
        t_index = 0
        position = None
        current = None
        entry = None
        move = 0
        
        DD = np.zeros(len(self.data) - 1)
            
        for i in range(len(self.data) - 1):
            # Handle trade entries
            if t_index < len(self.trades) and self.trades['entry_time'].iloc[t_index] == self.data.index[i]:
                entry = self.trades['entry_price'].iloc[t_index]
                position = self.trades['position'].iloc[t_index]  # 'long' or 'short'
                move = 1 if position == 'long' else -1

            # Handle trade exits
            if t_index < len(self.trades) and self.trades['exit_time'].iloc[t_index] == self.data.index[i]:
                t_index += 1  # Move to the next trade
                
                # Check if a new trade immediately follows the exit (perpetual system)
                if t_index < len(self.trades) and self.trades['entry_time'].iloc[t_index] == self.data.index[i]:
                    entry = self.trades['entry_price'].iloc[t_index]
                    position = self.trades['position'].iloc[t_index]
                    move = 1 if position == 'long' else -1
                else:
                    position = 'cash'
                    move = 0
                    entry = None  # Reset entry for non-perpetual systems

            # Update current price based on position
            if position == 'long':
                current = self.data['low'].iloc[i]  # Worst-case for long
            elif position == 'short':
                current = self.data['high'].iloc[i]  # Worst-case for short

            # Calculate drawdown
            if entry is not None:
                max_DD = min(max_DD, ((current - entry) / entry) * move)
                DD[i] = max_DD
                
        return round(-100*max_DD, 2)

    # Sortino Ratio
    def sortino_ratio(self):    
        daily_equity = self.equity['equity'].pct_change()
        negative_devs = np.where(daily_equity < 0, daily_equity, 0)
        squared_devs = negative_devs ** 2
        mean_squared_dev = np.mean(squared_devs)
        downside_deviation = np.sqrt(mean_squared_dev)
        mean_returns = np.mean(daily_equity)
        
        return round(mean_returns / downside_deviation * np.sqrt(365, 2))

    # Sharpe Ratio
    def sharpe_ratio(self):
        daily_equity = self.equity['equity'].pct_change()
        stdev = np.std(daily_equity)
        mean_returns = np.mean(daily_equity)
        
        return round(mean_returns / stdev * np.sqrt(365), 2)

    # Profit Factor
    def profit_factor(self):
        profits = np.sum(self.trades['diff'][self.trades['diff'] > 0])
        losses = np.sum(self.trades['diff'][self.trades['diff'] < 0])

        if losses == 0:
            return np.nan
        
        profit_factor = profits / (-losses)

        return round(profit_factor, 2)

    # Percent Profitable
    def percent_profitable(self):
        count_wins = len(self.trades[self.trades['returns'] > 0])
        count_losses = len(self.trades[self.trades['returns'] < 0])
        
        if (count_wins + count_losses) > 0.0:
            return round(100*(count_wins / (count_wins + count_losses)), 2)
        else:
            return 0.0
    
    # Number of Trades
    def NofTrades(self):
        return len((self.trades[self.trades['exit_price'].notna()]))

    # Omega Ratio
    def omega_ratio(self):
        diff = self.equity['equity'].pct_change()
        profit = np.sum(diff[diff > 0])
        losses = np.sum(diff[diff < 0])
        
        return round(profit / - losses, 2)

    # Half Kelly
    def half_kelly(self):
        n_wins = len(self.trades[self.trades['returns'] > 0])
        n_trades = self.NofTrades()
        pf = self.profit_factor()
        half_kelly = round(100*(n_wins/n_trades - ((1 - (n_wins/n_trades)) / pf))/2, 2)
        
        return half_kelly

    # Net Profit
    def net_profit_pct(self):
        return round(100*(self.equity['gross'].iloc[-1] - 1), 2)
    
    def annualised_returns(self):
        rets = self.equity['equity'].pct_change().dropna()
        geomean = (math.prod((1+rets)) ** (1/len(rets)))

        return round(100*(geomean ** (365) - 1), 2)


    # Slap Score
    def slapScore(self):
        # Compute stats
        equity_maxDD_value = self.equity_maxDD()
        intra_max_DD_value = self.intra_max_DD()
        sortino_ratio_value = self.sortino_ratio()
        sharpe_ratio_value = self.sharpe_ratio()
        profit_factor_value = self.profit_factor()
        percent_profitable_value = self.percent_profitable()
        N_of_Trades_value = self.NofTrades()
        omega_ratio_value = self.omega_ratio()

        slapScoreMetrics = None
        slapScore = 0

        # Dictionary of metric bounds that determine Slap Score. Different scores depending on strategy type
        if self.mode == 'L': # Use different scoring for Long Only strategies
            slapScoreMetrics = {
                'drawdown': {
                    'low': 20,
                    "bound1": 25,
                    "bound2": 30,
                    "bound3": 35,
                    "bound4": 40,
                    'high': 45,
                },
                'sharpe': {
                    "high": 1.7,
                    "bound1": 1.6,
                    "bound2": 1.5,
                    "bound3": 1.3,
                    "bound4": 1.1,
                    "low": 0.9,
                },
                'sortino': {
                    "high": 2.7,
                    "bound1": 2.5,
                    "bound2": 2.3,
                    "bound3": 2.1,
                    "bound4": 1.9,
                    "low": 1.7,
                },
                'trades': {
                    "low": 20,
                    "bound1": 22,
                    "bound2": 25,
                    "bound3": 28,
                    "bound4": 33,
                    "high": 45,
                },
                'winrate': {
                    'high': 70,
                    "bound1": 65,
                    "bound2": 60,
                    "bound3": 55,
                    "bound4": 50,
                    'low': 45,
                },
                'profit_factor': {
                    "high": 5.0,
                    'bound1': 4.5,
                    'bound2': 4.0,
                    'bound3': 3.5,
                    'bound4': 3.0,
                    "low": 2.5,
                },
                'omega':{
                    "high": 1.45,
                    "bound1": 1.4,
                    "bound2": 1.35,
                    "bound3": 1.3,
                    "bound4": 1.25,
                    "low": 1.2,
                }
            }

        elif self.mode == 'S': # Use different scoring for Short Only strategies
            slapScoreMetrics = {
                'drawdown': {
                    'low': 10,
                    "bound1": 15,
                    "bound2": 20,
                    "bound3": 25,
                    "bound4": 30,
                    'high': 35,
                },
                'sharpe': {
                    "high": 1.0,
                    "bound1": 0.9,
                    "bound2": 0.8,
                    "bound3": 0.7,
                    "bound4": 0.6,
                    "low": 0.5,
                },
                'sortino': {
                    "high": 1.5,
                    "bound1": 1.4,
                    "bound2": 1.3,
                    "bound3": 1.2,
                    "bound4": 1.1,
                    "low": 1.0,
                },
                'trades': {
                    "low": 20,
                    "bound1": 22,
                    "bound2": 25,
                    "bound3": 28,
                    "bound4": 33,
                    "high": 45,
                },
                'winrate': {
                    'high': 70,
                    "bound1": 65,
                    "bound2": 60,
                    "bound3": 55,
                    "bound4": 50,
                    'low': 45,
                },
                'profit_factor': {
                    "high": 4.0,
                    'bound1': 3.5,
                    'bound2': 3.0,
                    'bound3': 2.5,
                    'bound4': 2.0,
                    "low": 1.5,
                },
                'omega':{
                    "high": 1.2,
                    "bound1": 1.15,
                    "bound2": 1.1,
                    "bound3": 1.05,
                    "bound4": 1.0,
                    "low": 0.9,
                }
            }

        else: # Use different scoring for Long/Short strategies
            slapScoreMetrics = {
                'drawdown': {
                    'low': 20,
                    "bound1": 25,
                    "bound2": 30,
                    "bound3": 35,
                    "bound4": 45,
                    'high': 50,
                },
                'sharpe': {
                    "high": 2.0,
                    "bound1": 1.8,
                    "bound2": 1.6,
                    "bound3": 1.4,
                    "bound4": 1.2,
                    "low": 1.0,
                },
                'sortino': {
                    "high": 3.1,
                    "bound1": 2.9,
                    "bound2": 2.7,
                    "bound3": 2.5,
                    "bound4": 2.3,
                    "low": 2.1,
                },
                'trades': {
                    "low": 30,
                    "bound1": 35,
                    "bound2": 40,
                    "bound3": 45,
                    "bound4": 55,
                    "high": 80,
                },
                'winrate': {
                    'high': 70,
                    "bound1": 65,
                    "bound2": 60,
                    "bound3": 55,
                    "bound4": 50,
                    'low': 45,
                },
                'profit_factor': {
                    "high": 4.5,
                    'bound1': 4.0,
                    'bound2': 3.5,
                    'bound3': 3.0,
                    'bound4': 2.5,
                    "low": 2.0,
                },
                'omega':{
                    "high": 1.35,
                    "bound1": 1.3,
                    "bound2": 1.25,
                    "bound3": 1.2,
                    "bound4": 1.15,
                    "low": 1.1,
                }
                
            }

        # Scoring
        # Equity DD
        slapScore += np.where(equity_maxDD_value < slapScoreMetrics['drawdown']["low"], 5, 
                            np.where(equity_maxDD_value >= slapScoreMetrics['drawdown']["low"] and equity_maxDD_value < slapScoreMetrics['drawdown']["bound1"], 4, 
                                    np.where(equity_maxDD_value >= slapScoreMetrics['drawdown']["bound1"] and equity_maxDD_value < slapScoreMetrics['drawdown']["bound2"], 3,
                                                np.where(equity_maxDD_value >= slapScoreMetrics['drawdown']["bound2"] and equity_maxDD_value < slapScoreMetrics['drawdown']["bound3"], 2, 
                                                        np.where(equity_maxDD_value >= slapScoreMetrics['drawdown']["bound3"] and equity_maxDD_value < slapScoreMetrics['drawdown']["bound4"], 1, 
                                                                np.where(equity_maxDD_value >= slapScoreMetrics['drawdown']["bound4"] and equity_maxDD_value < slapScoreMetrics['drawdown']["high"], 0, -1))))))
        
        # Intra-Trade DD
        slapScore += np.where(intra_max_DD_value < slapScoreMetrics['drawdown']["low"], 5, 
                            np.where(intra_max_DD_value >= slapScoreMetrics['drawdown']["low"] and intra_max_DD_value < slapScoreMetrics['drawdown']["bound1"], 4, 
                                    np.where(intra_max_DD_value >= slapScoreMetrics['drawdown']["bound1"] and intra_max_DD_value < slapScoreMetrics['drawdown']["bound2"], 3,
                                                np.where(intra_max_DD_value >= slapScoreMetrics['drawdown']["bound2"] and intra_max_DD_value < slapScoreMetrics['drawdown']["bound3"], 2, 
                                                        np.where(intra_max_DD_value >= slapScoreMetrics['drawdown']["bound3"] and intra_max_DD_value < slapScoreMetrics['drawdown']["bound4"], 1, 
                                                                np.where(intra_max_DD_value >= slapScoreMetrics['drawdown']["bound4"] and intra_max_DD_value < slapScoreMetrics['drawdown']["high"], 0, -1000)))))) # Changed -1 to -1000
        
        # Sortino Ratio
        slapScore += np.where(sortino_ratio_value >= slapScoreMetrics['sortino']['high'], 5, 
                            np.where(sortino_ratio_value < slapScoreMetrics['sortino']['high'] and sortino_ratio_value >= slapScoreMetrics['sortino']['bound1'], 4, 
                                    np.where(sortino_ratio_value < slapScoreMetrics['sortino']['bound1'] and sortino_ratio_value >= slapScoreMetrics['sortino']['bound2'], 3, 
                                                np.where(sortino_ratio_value < slapScoreMetrics['sortino']['bound2'] and sortino_ratio_value >= slapScoreMetrics['sortino']['bound3'], 2, 
                                                        np.where(sortino_ratio_value < slapScoreMetrics['sortino']['bound3'] and sortino_ratio_value >= slapScoreMetrics['sortino']['bound4'], 1, 
                                                                np.where(sortino_ratio_value < slapScoreMetrics['sortino']['bound4'] and sortino_ratio_value >= slapScoreMetrics['sortino']['low'], 0, -1))))))
        
        # Sharpe Ratio
        slapScore += np.where(sharpe_ratio_value >= slapScoreMetrics['sharpe']['high'], 5, 
                            np.where(sharpe_ratio_value < slapScoreMetrics['sharpe']['high'] and sharpe_ratio_value >= slapScoreMetrics['sharpe']['bound1'], 4, 
                                    np.where(sharpe_ratio_value < slapScoreMetrics['sharpe']['bound1'] and sharpe_ratio_value >= slapScoreMetrics['sharpe']['bound2'], 3, 
                                                np.where(sharpe_ratio_value < slapScoreMetrics['sharpe']['bound2'] and sharpe_ratio_value >= slapScoreMetrics['sharpe']['bound3'], 2, 
                                                        np.where(sharpe_ratio_value < slapScoreMetrics['sharpe']['bound3'] and sharpe_ratio_value >= slapScoreMetrics['sharpe']['bound4'], 1, 
                                                                np.where(sharpe_ratio_value < slapScoreMetrics['sharpe']['bound4'] and sharpe_ratio_value >= slapScoreMetrics['sharpe']['low'], 0, -1))))))
        
        # Profit Factor
        slapScore += np.where(profit_factor_value >= slapScoreMetrics['profit_factor']['high'], 5, 
                            np.where(profit_factor_value < slapScoreMetrics['profit_factor']['high'] and profit_factor_value >= slapScoreMetrics['profit_factor']['bound1'], 4, 
                                    np.where(profit_factor_value < slapScoreMetrics['profit_factor']['bound1'] and profit_factor_value >= slapScoreMetrics['profit_factor']['bound2'], 3, 
                                                np.where(profit_factor_value < slapScoreMetrics['profit_factor']['bound2'] and profit_factor_value >= slapScoreMetrics['profit_factor']['bound3'], 2, 
                                                        np.where(profit_factor_value < slapScoreMetrics['profit_factor']['bound3'] and profit_factor_value >= slapScoreMetrics['profit_factor']['bound4'], 1, 
                                                                np.where(profit_factor_value < slapScoreMetrics['profit_factor']['bound4'] and profit_factor_value >= slapScoreMetrics['profit_factor']['low'], 0, -1))))))
    
        # Percent Profitable
        slapScore += np.where(percent_profitable_value >= slapScoreMetrics['winrate']['high'], 5,
                            np.where(percent_profitable_value < slapScoreMetrics['winrate']['high'] and percent_profitable_value >= slapScoreMetrics['winrate']['bound1'], 4, 
                                    np.where(percent_profitable_value < slapScoreMetrics['winrate']['bound1'] and percent_profitable_value >= slapScoreMetrics['winrate']['bound2'], 3, 
                                                np.where(percent_profitable_value < slapScoreMetrics['winrate']['bound2'] and percent_profitable_value >= slapScoreMetrics['winrate']['bound3'], 2, 
                                                        np.where(percent_profitable_value < slapScoreMetrics['winrate']['bound3'] and percent_profitable_value >= slapScoreMetrics['winrate']['bound4'], 1, 
                                                                np.where(percent_profitable_value < slapScoreMetrics['winrate']['bound4'] and percent_profitable_value >= slapScoreMetrics['winrate']['low'], 0, -1))))))
        
        # Number of Trades
        slapScore += np.where(N_of_Trades_value < slapScoreMetrics['trades']['low'] or N_of_Trades_value > slapScoreMetrics['trades']['high'], -1,  
                            np.where(N_of_Trades_value >= slapScoreMetrics['trades']['low'] and N_of_Trades_value < slapScoreMetrics['trades']['bound1'], 5, 
                                    np.where(N_of_Trades_value >= slapScoreMetrics['trades']['bound1'] and N_of_Trades_value < slapScoreMetrics['trades']['bound2'], 4, 
                                                np.where(N_of_Trades_value >= slapScoreMetrics['trades']['bound2'] and N_of_Trades_value < slapScoreMetrics['trades']['bound3'], 3,
                                                        np.where(N_of_Trades_value >= slapScoreMetrics['trades']['bound3'] and N_of_Trades_value < slapScoreMetrics['trades']['bound4'], 2, 
                                                                np.where(N_of_Trades_value >= slapScoreMetrics['trades']['bound4'] and N_of_Trades_value < slapScoreMetrics['trades']['high'], 1, -1))))))
        
        # Omega Ratio
        slapScore += np.where(omega_ratio_value >= slapScoreMetrics["omega"]['high'], 5, 
                            np.where(omega_ratio_value < slapScoreMetrics['omega']['high'] and omega_ratio_value >= slapScoreMetrics['omega']['bound1'], 4, 
                                    np.where(omega_ratio_value < slapScoreMetrics['omega']['bound1'] and omega_ratio_value >= slapScoreMetrics['omega']['bound2'], 3, 
                                                np.where(omega_ratio_value < slapScoreMetrics['omega']['bound2'] and omega_ratio_value >= slapScoreMetrics['omega']['bound3'], 2, 
                                                        np.where(omega_ratio_value < slapScoreMetrics['omega']['bound3'] and omega_ratio_value >= slapScoreMetrics['omega']['bound4'], 1, 
                                                                np.where(omega_ratio_value < slapScoreMetrics['omega']['bound4'] and omega_ratio_value >= slapScoreMetrics['omega']['low'], 0, -1))))))
        
        return slapScore


    # Metrics Table
    def MetricsTable(self):

        # Performance Metrics
        equity_maxDD_value = self.equity_maxDD()
        intra_max_DD_value = self.intra_max_DD()
        sortino_ratio_value = self.sortino_ratio()
        sharpe_ratio_value = self.sharpe_ratio()
        profit_factor_value = self.profit_factor()
        percent_profitable_value = self.percent_profitable()
        N_of_Trades_value = self.NofTrades()
        omega_ratio_value = self.omega_ratio()
        half_kelly_value = self.half_kelly()
        net_profit_value = self.net_profit_pct()
        annualised_returns_value = self.annualised_returns()
        slapScore_value = self.slapScore()
        
        # Output Dataframe
        Metrics_Table = pd.DataFrame({
            'Metric': [
                'Equity Max DD (%)', 
                'Intra-Trade Max DD (%)', 
                'Sortino Ratio', 
                'Sharpe Ratio', 
                'Profit Factor', 
                'Percent Profitable (%)', 
                'Number of Trades', 
                'Omega Ratio', 
                'Half Kelly (%)', 
                'Net Profit (%)', 
                'Annualised Returns (%)', 
                'Slap Score'
                ],
            'Value': [
                equity_maxDD_value, 
                intra_max_DD_value,
                sortino_ratio_value, 
                sharpe_ratio_value, 
                profit_factor_value, 
                percent_profitable_value, 
                N_of_Trades_value, 
                omega_ratio_value, 
                half_kelly_value, 
                net_profit_value, 
                annualised_returns_value, 
                slapScore_value]
            }
        )

        return Metrics_Table


# Metrics Table (Portfolio)
# Pass equity curve of portfolio
def portMetricsTable(port):
    # Sharpe
    sharpe = round(port.mean() / port.std() * np.sqrt(365), 2)

    # Max Drawdown (Equity)
    cum_eq = (1 + port).cumprod()
    hwm = cum_eq.cummax()
    dd = (hwm - cum_eq) / hwm
    maxDD = round(dd.max()*100, 2)

    # Percent Positive Days
    pctWinDays = round(sum(port > 0)/len(port) * 100, 2)

    # Annualised Returns
    annRets = round((np.prod((1+port)) ** (365 / len(port)) - 1)*100, 2)

    # Return DataFrame
    metricsDF = pd.DataFrame({
        'Metric': [
            'Sharpe Ratio', 
            'Equity Max Drawdown (%)',
            'Winning Days (%)',
            'Annualised Returns (%)'
        ],
        'Values': [
            sharpe, 
            maxDD, 
            pctWinDays, 
            annRets
        ]
        }
    )

    return metricsDF



