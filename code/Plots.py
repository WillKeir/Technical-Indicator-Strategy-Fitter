# Imports
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Plot Candlestick Chart with Trades
def candleChart(data, trades, mode):

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        increasing=dict(line=dict(color="grey"), fillcolor="lightgrey"),
        decreasing=dict(line=dict(color="darkgrey"), fillcolor="white")
    )])

    # Trade entry data
    long_entries = trades[trades['position'] == 'long']['entry_time']
    short_entries = trades[trades['position'] == 'short']['entry_time']
    long_prices_entries = trades[trades['position'] == 'long']['entry_price']
    short_prices_entries = trades[trades['position'] == 'short']['entry_price']

    # Trade exit data
    long_exits = trades[trades['position'] == 'long']['exit_time']
    short_exits = trades[trades['position'] == 'short']['exit_time']
    long_prices_exits = trades[trades['position'] == 'long']['exit_price']
    short_prices_exits = trades[trades['position'] == 'short']['exit_price']

    if mode == 'LS':
        # Long markers (triangle and small dot)
        fig.add_trace(go.Scatter(
            x=long_entries,
            y=0.9 * long_prices_entries,
            mode="markers",
            marker=dict(symbol="triangle-up", color="blue", size=8),
            name="Long Entry"
        ))
        fig.add_trace(go.Scatter(
            x=long_entries,
            y=long_prices_entries,
            mode="markers",
            marker=dict(symbol="circle", color="blue", size=3),
            name="Long Entry"
        ))

        # Short markers (triangle and small dot)
        fig.add_trace(go.Scatter(
            x=short_entries,
            y=1.11 * short_prices_entries,
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=8),
            name="Short Entry"
        ))
        fig.add_trace(go.Scatter(
            x=short_entries,
            y=short_prices_entries,
            mode="markers",
            marker=dict(symbol="circle", color="red", size=3),
            name="Short Entry"
        ))

        # Layout
        fig.update_layout(
            title="Candlestick Chart with Trade Entries",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis_type="log",
            width=1000,
            height=700
        )
    
    elif mode == 'L':
        # Long markers (triangle and small dot)
        fig.add_trace(go.Scatter(
            x=long_entries,
            y=0.9 * long_prices_entries,
            mode="markers",
            marker=dict(symbol="triangle-up", color="blue", size=8),
            name="Long Entry"
        ))
        fig.add_trace(go.Scatter(
            x=long_entries,
            y=long_prices_entries,
            mode="markers",
            marker=dict(symbol="circle", color="blue", size=3),
            name="Long Entry"
        ))

        # Trade close markers (triangle and small dot)
        fig.add_trace(go.Scatter(
            x=long_exits,
            y=1.11 * long_prices_exits,
            mode="markers",
            marker=dict(symbol="triangle-down", color="#c334eb", size=8),
            name="Long Exit"
        ))
        fig.add_trace(go.Scatter(
            x=long_exits,
            y=long_prices_exits,
            mode="markers",
            marker=dict(symbol="circle", color="#c334eb", size=3),
            name="Long Exit"
        ))

        # Layout
        fig.update_layout(
            title="Candlestick Chart with Trade Entries",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis_type="log",
            width=1000,
            height=700
        )

    elif mode == 'S':
        # Trade close markers (triangle and small dot)
        fig.add_trace(go.Scatter(
            x=short_exits,
            y=0.9 * short_prices_exits,
            mode="markers",
            marker=dict(symbol="triangle-up", color="#c334eb", size=8),
            name="Short Exit"
        ))
        fig.add_trace(go.Scatter(
            x=short_exits,
            y=short_prices_exits,
            mode="markers",
            marker=dict(symbol="circle", color="#c334eb", size=3),
            name="Short Exit"
        ))

        # Short markers (triangle and small dot)
        fig.add_trace(go.Scatter(
            x=short_entries,
            y=1.11 * short_prices_entries,
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=8),
            name="Short Entry"
        ))
        fig.add_trace(go.Scatter(
            x=short_entries,
            y=short_prices_entries,
            mode="markers",
            marker=dict(symbol="circle", color="red", size=3),
            name="Short Entry"
        ))

        # Layout
        fig.update_layout(
            title="Candlestick Chart with Trade Entries",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis_type="log",
            width=1000,
            height=700
        )

    fig.show()

def retsPlot(equity):

    x = equity.index
    y = equity['equity'].pct_change().cumsum()

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, y, label='Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Cumulative Sum of Returns')
    plt.grid(True)
    plt.show()


def equityPlot(equity):

    x = equity.index
    y = equity['equity']

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, y, label='Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.title('Equity Curve Over Time')
    plt.grid(True)
    plt.show()

