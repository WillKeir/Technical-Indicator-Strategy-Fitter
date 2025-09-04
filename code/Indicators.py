# Import Libraries
import numpy as np
import pandas as pd
import pandas_ta as ta

# Define Indicator Class
class Indicator:
    def __init__(self, name):
        self.name = name
    
    @staticmethod
    def check_inputs_Bayes(indicator_params):
        return indicator_params
    
    def calculate(self, df):
        raise NotImplementedError("Subclasses must implement this method.")


# Indicators         
## RSI x Threshold
class RSI_X_Threshold(Indicator):
    
    # Constructor
    def __init__(self, **params):
        super().__init__("RSI_X_Threshold")
        self.params = params

    # Input Bounds
    bounds = {
        'source': ('close', 'hl2', 'hlc3', 'hlcc4'),
        'length': (5, 45),
        'lThresh': (40, 75),
        'sThresh': (30, 65),
    }
    
    # Input Step (for parameter robustness)
    inputStep = {
        'length': 1,
        'lThresh': 1,
        'sThresh': 1
    }

    # Check inputs during Bayesian Optimisation
    # Checks for clashes/inconsistencies in inputs before passing them to the backtest function
    @staticmethod
    def check_inputs_Bayes(indicator_params):
        if indicator_params['sThresh'] > indicator_params['lThresh']:
            low_bound = RSI_X_Threshold.bounds['sThresh'][0]
            high_bound = indicator_params['lThresh'] + 1    # + 1 as sThresh = lThresh is OK
            indicator_params['sThresh'] = np.random.randint(low_bound, high_bound)
        
        return indicator_params

    # Compute signal
    def calculate(self, df):
        source = self.params.get('source')
        length = self.params.get('length')
        lThresh = self.params.get('lThresh')
        sThresh = self.params.get('sThresh')

        data = df.copy()
        
        delta = data[source].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        rma_gain = ta.rma(pd.Series(gain), length)
        rma_loss = ta.rma(pd.Series(loss), length)

        rs = rma_gain / rma_loss
        rsi = 100 - (100 / (1 + rs))

        signal = np.where(rsi > lThresh, 1.0, np.where(rsi < sThresh, -1.0, 0.0))

        return pd.Series(signal, index=data.index)


# Mapping of indicator names to their respective classes. Used in optimisation
INDICATOR_CLASSES = {
    'RSI_X_Threshold': RSI_X_Threshold
}

# Aggregator Class
## Used to aggregate indicators together
class Aggregator:

    # Constructor
    def __init__(self, data, indicator_names, indicator_params):
        self.data = data
        self.indicator_names = indicator_names
        self.indicator_params = indicator_params

    # Combine Indicator Signals
    def combine_signals(self):
        TPI_Signals = 0
        n_indis = len(self.indicator_names)
        
        for name in self.indicator_names:
            if name in INDICATOR_CLASSES:
                cls = INDICATOR_CLASSES[name]
                params = self.indicator_params.get(name, {})
                instance = cls(**params)
                signal = instance.calculate(self.data)
                TPI_Signals += signal
            else:
                print(f"Warning: Indicator '{name}' not found.")
        
        return TPI_Signals / n_indis if n_indis > 0 else 0

