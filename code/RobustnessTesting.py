import numpy as np
import pandas as pd
import pandas_ta as ta
import copy

import Indicators as ic
import Optimisers as op
import PerformanceMetrics as pm


def param_robustness(output, data, indicator_names, long_filters, short_filters, indicator_params, mode, backtest_startdate='2018-01-01', backtest_enddate=None):
    
    # Clean Data
    data = pm.data_cleaning(data)

    # Copy indicator_params
    backtest_params = copy.deepcopy(indicator_params)

    # List of Metrics
    metrics_list = [
                'Equity Max DD (%)',
                'Intra-Trade Max DD (%)', 
                'Sortino Ratio', 
                'Sharpe Ratio', 
                'Profit Factor', 
                'Percent Profitable (%)', 
                'Number of Trades', 
                'Omega Ratio',
                'Annualised Returns (%)'
            ]

    # Initialise Coefficient of Variation dictionary ()
    CofV = {ind: {} for ind in list(pd.unique((indicator_names + long_filters + short_filters)))}

    # Initialise parameter performance dictionary (stores metrics for each parameter change)
    param_robustness_data = {ind: {} for ind in list(pd.unique((indicator_names + long_filters + short_filters)))}

    # Iterate through each indicator
    for indicator in list(pd.unique((indicator_names + long_filters + short_filters))):
        indClass = ic.INDICATOR_CLASSES[indicator]
        
        for input in indClass.bounds.items():

            # Skip input if not in inputStep
            if input[0] not in indClass.inputStep:
                continue
            
            # Dataframe to store metrics
            input_metrics = pd.DataFrame(index=metrics_list)

            # Pull step length
            step = indClass.inputStep[input[0]]

            # Check steps
            steps_list = [i * step for k in range(8) for i in ([k, -k] if k != 0 else [0])]
            countIter=0

            for steps in steps_list: 
                # Update Parameter to test
                backtest_params[indicator][input[0]] = indicator_params[indicator][input[0]] + steps
                
                # Skip iteration if input is invalid
                skip_iter = indClass.check_inputs_FAFO(backtest_params[indicator])

                if not skip_iter:
                    # Backtest Inputs
                    metrics = op.backtest(data, indicator_names, long_filters, short_filters, backtest_params, mode, 'MetricsTable', backtest_startdate, backtest_enddate, optim=True) 

                    # Store Metrics of Backtest
                    input_metrics[str(backtest_params[indicator][input[0]])] = [
                        metrics.loc[metrics['Metric'] == m, 'Value'].values[0] for m in metrics_list
                    ]
                
                # Add to count
                countIter += 1

                # Cancel loop after 7 tests
                if countIter >= 7:
                    break

            # Sort columns into in ascending order (for ease of viewing)
            input_metrics = input_metrics[sorted(input_metrics.columns)]
            
            # Add metric changes to param_robustness_data
            param_robustness_data[indicator][input[0]] = input_metrics

            # Add CofV column
            input_metrics['CofV'] = round(input_metrics.std(1) / input_metrics.mean(1), 4)

            # Add Coefficient of Variation values to dictionary
            CofV[indicator][input[0]] = input_metrics['CofV']

            # Reset params to backtest after each indicator
            backtest_params = copy.deepcopy(indicator_params)

    # Turn param_robustness_data into a multi-index DataFrame
    dfs = []

    for outer_key, inner_dict in param_robustness_data.items():
        for inner_key, df in inner_dict.items():
            # Assign MultiIndex to columns of this DataFrame
            df.columns = pd.MultiIndex.from_product([[outer_key], [inner_key], df.columns],
                                                names=['Indicator', 'Input', 'OriginalCol'])
            dfs.append(df)

    # Combine 'dfs' into one DataFrame with MultiIndex columns
    param_robustness_data = pd.concat(dfs, axis=1)

    # Compute average CofV per metric
    df = [pd.DataFrame(CofV[i]) for i in CofV.keys()]
    CofV_metrics = pd.concat(df, axis=1).mean(1).to_frame(name='CofV')
    
    # Compute average CofV per input
    results = []

    for indicator in list(pd.unique((indicator_names + long_filters + short_filters))):
        means = round(pd.DataFrame(CofV[indicator]).mean(), 4)
        df = means.to_frame(name='CofV')
        df['Indicator'] = indicator
        df = df.set_index('Indicator', append=True).swaplevel()
        results.append(df)
    
    CofV_inputs = pd.concat(results)

    # Produce Output
    # 'raw': Dataframe of metrics at every parameter change
    if output == 'raw':
        return param_robustness_data
    
    # 'inputs': Coefficient of Variation for each input
    elif output == 'inputs':
        return CofV_inputs
    
    # 'inputs': Coefficient of Variation for each metric
    elif output == 'metrics':
        return CofV_metrics
    else:
        return None
    