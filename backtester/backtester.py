import dask_cudf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def backtest(df, initial_capital: float=100000, transaction_cost_rate:float=0.1, slippage: float = 0.0):
    transaction_cost_rate /= 100
    holding_penalty = 1
    interval_minutes = 30
    df["Time"] = pd.to_timedelta(df["Time"])
    df["Day"] = (df["Time"].diff() < pd.Timedelta(0)).cumsum()
    df["Day_Index"] = df.groupby("Day").cumcount()
    df = df.dropna(subset=["Price"])
    
    # Create sequential index for continuous plotting
    df = df.reset_index(drop=True)
    df["PlotIndex"] = range(len(df))

    df["Adjusted_Signal"] = 0
    df["Position"] = 0

    Days=0
    
    Daily_PnL=[]
    Total_PnL=0

    Daily_Trade_Count=[]

    Penalty_Counts=0
    All_Trades = []
    Total_Trades = 0
    Winning_Trades = 0
    Losing_Trades = 0
    Winning_Trade_PnLs = []
    Losing_Trade_PnLs = []

    Interval_PnL = np.zeros(15)
    for day, day_df in df.groupby("Day"):
        current_position = 0
        Position=[]
        Adjusted_Signal = []
        Entry_Indices = []
        shifted_signals = day_df["Signal"].shift(1).fillna(day_df["Signal"].iloc[0]).tolist()
        shifted_signals[-1] = day_df["Signal"].iloc[-1]
        Factor=initial_capital/day_df["Price"].iloc[0]
        for i,sig in enumerate(day_df["Signal"]):
            if sig != current_position and sig != 0:
                Adjusted_Signal.append(sig)
                current_position += sig
                Entry_Indices.append((i, sig))
            else:
                Adjusted_Signal.append(0)
            
            if i == len(day_df) - 1 and current_position != 0:
                Adjusted_Signal[-1] += -current_position
                current_position = 0
                Penalty_Counts+=1

            Position.append(current_position)
        
        exit_idx = 0
        for entry_idx, entry_direction in Entry_Indices:
            entry_price = day_df["Price"].iloc[entry_idx]
            exit_idx = None

            # Find exit
            for j in range(entry_idx + 1, len(Adjusted_Signal)):
                if Adjusted_Signal[j] != 0 and np.sign(Adjusted_Signal[j]) == -np.sign(entry_direction):
                    exit_idx = j
                    break

            # If no exit found → forced exit at last bar
            if exit_idx is None:
                exit_idx = len(Adjusted_Signal) - 1

            exit_price = day_df["Price"].iloc[exit_idx]

            # HOLD TIME
            hold_time = exit_idx - entry_idx
            All_Trades.append(hold_time)

            # TRADE PnL
            if entry_direction > 0:  # LONG
                trade_pnl = exit_price - entry_price
            else:                    # SHORT
                trade_pnl = entry_price - exit_price

            # WIN / LOSS COUNTING
            if trade_pnl > 0:
                Winning_Trades += 1
                Winning_Trade_PnLs.append(trade_pnl * Factor)
            elif trade_pnl <= 0:
                Losing_Trades += 1
                Losing_Trade_PnLs.append(trade_pnl * Factor)

        # Count total trades for this day
        Total_Trades += len(Entry_Indices)

        day_df.loc[day_df.index, "Adjusted_Signal"] = Adjusted_Signal
        day_df.loc[day_df.index, "Position"] = Position

        Trade_Quantity = day_df["Position"].diff() * Factor
        Trade_Quantity.iloc[0] = day_df["Position"].iloc[0] * Factor

        Unit_Transaction_Cost = abs(day_df["Adjusted_Signal"] * day_df["Price"]) * transaction_cost_rate + abs(day_df["Adjusted_Signal"]) * slippage
        Transaction_Cost = Unit_Transaction_Cost * Factor
        df.loc[day_df.index, "Transaction_Cost"] = Transaction_Cost

        Unit_Cash = day_df["Price"].iloc[0] - (Adjusted_Signal * day_df["Price"] + Unit_Transaction_Cost).cumsum()
        # Unit_Cash = pd.Series(np.zeros(len(day_df)), index=day_df.index)
        # for i in range(0, len(day_df)):
        #     if i==0:
        #         Unit_Cash.iloc[i] = day_df["Price"].iloc[0] - (day_df["Adjusted_Signal"].iloc[i] * day_df["Price"].iloc[i]) - Unit_Transaction_Cost.iloc[i]
        #     else:
        #         Unit_Cash.iloc[i] = Unit_Cash.iloc[i-1] - (day_df["Adjusted_Signal"].iloc[i] * day_df["Price"].iloc[i]) - Unit_Transaction_Cost.iloc[i]
        Cash = Unit_Cash * Factor

        Unit_Holdings = day_df["Position"] * day_df["Price"]
        Holdings = Unit_Holdings * Factor
        NAV = Cash + Holdings
        PnL = NAV - initial_capital

        df.loc[day_df.index, "Adjusted_Signal"] = Adjusted_Signal
        df.loc[day_df.index, "Position"] = Position
        df.loc[day_df.index, "Trade_Quantity"] = Trade_Quantity
        df.loc[day_df.index, "Transaction_Cost"] = Transaction_Cost
        df.loc[day_df.index, "Unit_Cash"] = Unit_Cash
        df.loc[day_df.index, "Cash"] = Cash
        df.loc[day_df.index, "Holdings"] = Holdings
        df.loc[day_df.index, "NAV"] = NAV
        df.loc[day_df.index, "PnL"] = PnL

        day_df["PnL"]=PnL
        interval = pd.Timedelta(minutes=interval_minutes)
        start_time = day_df["Time"].iloc[0]
        end_time = day_df["Time"].iloc[-1]
        num_intervals = int(np.ceil((end_time - start_time) / interval))

        for i in range(num_intervals):
            t_start = start_time + i * interval
            t_end = start_time + (i + 1) * interval

            # Ensure we stay within data bounds
            df_interval = day_df[(day_df["Time"] >= t_start) & (day_df["Time"] < t_end)]
            if df_interval.empty:
                pnl_in_bin = 0
            else:
                pnl_start = df_interval["PnL"].iloc[0]
                pnl_end = df_interval["PnL"].iloc[-1]
                pnl_in_bin = pnl_end - pnl_start

            if i < len(Interval_PnL):
                Interval_PnL[i] += pnl_in_bin
            else:
                Interval_PnL = np.append(Interval_PnL, pnl_in_bin)

        Days+=1

        Daily_PnL.append(PnL.iloc[-1])
        Total_PnL+=PnL.iloc[-1]

        Daily_Trade_Count.append((day_df["Adjusted_Signal"]!=0).sum())

        print("Day",day,"-> Daily PnL:", PnL.iloc[-1], "| Total PnL:", Total_PnL)
    print("\n")

    if len(All_Trades) > 0:
        Avg_Hold_Period_Bars = np.mean(All_Trades)
        Median_Hold_Period_Bars = np.median(All_Trades)
        Min_Hold_Period_Bars = np.min(All_Trades)
        Max_Hold_Period_Bars = np.max(All_Trades)
        # Convert to seconds (assuming 1 bar = 1 second)
        Avg_Hold_Period_Seconds = Avg_Hold_Period_Bars
        Avg_Hold_Period_Minutes = Avg_Hold_Period_Seconds / 60
        
        # Calculate percentiles
        P25_Hold = np.percentile(All_Trades, 25)
        P75_Hold = np.percentile(All_Trades, 75)
        P90_Hold = np.percentile(All_Trades, 90)
        P95_Hold = np.percentile(All_Trades, 95)
    else:
        Avg_Hold_Period_Bars = 0
        Median_Hold_Period_Bars = 0
        Min_Hold_Period_Bars = 0
        Max_Hold_Period_Bars = 0
        Avg_Hold_Period_Seconds = 0
        Avg_Hold_Period_Minutes = 0
        P25_Hold = 0
        P75_Hold = 0
        P90_Hold = 0
        P95_Hold = 0

    Final_Capital = initial_capital + Total_PnL
    Total_Returns = (Final_Capital - initial_capital) * 100 / initial_capital
    Final_Returns = Total_Returns - Penalty_Counts * holding_penalty

    CAGR = 100*((Final_Capital / initial_capital) ** (252 / Days) - 1)

    # Daily_Returns = np.array(Daily_PnL) / initial_capital
    
    Daily_MDD = []
    for day, day_df in df.groupby("Day"):
        NAV = day_df["NAV"].values
        Running_Max = np.maximum.accumulate(NAV)
        DD = (NAV - Running_Max) / Running_Max
        df.loc[day_df.index, "Drawdown"] = DD
        Daily_MDD.append(DD.min())

    Equity = initial_capital+np.cumsum(Daily_PnL)
    Returns=np.diff(Equity)/(np.abs(Equity[:-1])+1e-9)
    Sharpe_Ratio=(np.mean(Returns) / np.std(Returns)) * np.sqrt(len(Returns))
    Total_TCost = np.cumsum(df["Transaction_Cost"])
    # PnL_cumsum = np.cumsum(Daily_PnL)
    # DD=np.minimum(PnL_cumsum,0.0)
    # Maximum_Drawdown = (DD.min() / initial_capital) * 100
    Maximum_Drawdown = 100 * (min(Daily_MDD))
    Annualized_Returns = 100 * (((Final_Capital-initial_capital) * 252) / (initial_capital * Days))
    Calmar_Ratio = Annualized_Returns / abs(Maximum_Drawdown)
    Best_Day_PnL = max(Daily_PnL)
    Worst_Day_PnL = min(Daily_PnL)
    Best_Day = Daily_PnL.index(Best_Day_PnL)
    Worst_Day = Daily_PnL.index(Worst_Day_PnL)
    Winning_Days = sum(1 for x in Daily_PnL if x > 0)
    Losing_Days = sum(1 for x in Daily_PnL if x < 0)
    Average_Winning_Day_PnL = np.mean([x for x in Daily_PnL if x > 0]) if Winning_Days > 0 else 0
    Average_Losing_Day_PnL = np.mean([x for x in Daily_PnL if x < 0]) if Losing_Days > 0 else 0
    Win_Rate = (Winning_Trades / Total_Trades * 100) if Total_Trades > 0 else 0
    Average_Winning_Trade = np.mean(Winning_Trade_PnLs) if len(Winning_Trade_PnLs) > 0 else 0
    Average_Losing_Trade = np.mean(Losing_Trade_PnLs) if len(Losing_Trade_PnLs) > 0 else 0

    Results = {
        "Initial Capital": initial_capital,
        "Final Capital": Final_Capital,
        "Total PnL": Total_PnL,
        "Total Transaction Cost": Total_TCost.iloc[-1],
        "Penalty Counts": Penalty_Counts,
        "Total Returns": Total_Returns,
        "CAGR": CAGR,
        "Annualized Returns": Annualized_Returns,
        "Sharpe Ratio": Sharpe_Ratio,
        "Calmar Ratio": Calmar_Ratio,
        "Maximum Drawdown": Maximum_Drawdown,
        "No. of Days": Days,
        "Winning Days": Winning_Days,
        "Losing Days": Losing_Days,
        "Best Day": Best_Day,
        "Worst Day": Worst_Day,
        "Best Day PnL": Best_Day_PnL,
        "Worst Day PnL": Worst_Day_PnL,
        "Average Winning Day PnL": Average_Winning_Day_PnL,
        "Average Losing Day PnL": Average_Losing_Day_PnL,
        "Total Trades": Total_Trades,
        "Winning Trades": Winning_Trades,
        "Losing Trades": Losing_Trades,
        "Win Rate (%)": Win_Rate,
        "Average Winning Trade": Average_Winning_Trade,
        "Average Losing Trade": Average_Losing_Trade,
        "Average Hold Period (seconds)": Avg_Hold_Period_Seconds,
    }

    print("\n=== Backtest Summary ===")
    print(f"Initial Capital               : {initial_capital:,.2f}")
    print(f"Final Capital                 : {Final_Capital:,.2f}")
    print(f"Total PnL                     : {Total_PnL:,.2f}")
    print(f"Total Transaction Cost         : {Total_TCost.iloc[-1]:,.2f}")
    print(f"Penalty Counts                : {Penalty_Counts}")
    print(f"Final Returns                 : {Total_Returns:,.2f}%")
    print(f"CAGR                          : {CAGR:,.4f}%")
    print(f"Annualized Returns            : {Annualized_Returns:,.4f}%")
    print(f"Sharpe Ratio                  : {Sharpe_Ratio:,.4f}")
    print(f"Calmar Ratio                  : {Calmar_Ratio:,.4f}")
    print(f"Maximum Drawdown              : {Maximum_Drawdown:,.4f}%")
    print(f"No. of Days                   : {Days}")
    print(f"Winning Days                  : {Winning_Days}")
    print(f"Losing Days                   : {Losing_Days}")
    print(f"Best Day                      : {Best_Day}")
    print(f"Worst Day                     : {Worst_Day}")
    print(f"Best Day PnL                  : {Best_Day_PnL:,.2f}")
    print(f"Worst Day PnL                 : {Worst_Day_PnL:,.2f}")
    # print(f"Average Winning Day PnL       : {Average_Winning_Day_PnL:,.2f}")
    # print(f"Average Losing Day PnL        : {Average_Losing_Day_PnL:,.2f}")
    print(f"Total Trades                  : {Total_Trades}")
    print(f"Winning Trades                : {Winning_Trades}")
    print(f"Losing Trades                 : {Losing_Trades}")
    print(f"Win Rate (%)                  : {Win_Rate:,.2f}%")
    print(f"Average Winning Trade         : {Average_Winning_Trade:,.4f}")
    print(f"Average Losing Trade          : {Average_Losing_Trade:,.4f}")
    print(f"Average Hold Period (seconds) : {Avg_Hold_Period_Seconds:,.2f}")
    print(f"Average Hold Period (minutes) : {Avg_Hold_Period_Minutes:,.2f}")
    print("========================\n")
    print(Interval_PnL)
    # --- save the same summary to a text file ---
    txt_filename = f"backtest_results.txt"
    report_lines = [
        "=== Backtest Summary ===",
        f"Initial Capital               : {initial_capital:,.2f}",
        f"Final Capital                 : {Final_Capital:,.2f}",
        f"Total PnL                     : {Total_PnL:,.2f}",
        f"Total Transaction Cost         : {Total_TCost.iloc[-1]:,.2f}",
        f"Penalty Counts                : {Penalty_Counts}",
        f"Final Returns                 : {Total_Returns:,.2f}%",
        f"CAGR                          : {CAGR:,.4f}%",
        f"Annualized Returns            : {Annualized_Returns:,.4f}%",
        f"Sharpe Ratio                  : {Sharpe_Ratio:,.4f}",
        f"Calmar Ratio                  : {Calmar_Ratio:,.4f}",
        f"Maximum Drawdown              : {Maximum_Drawdown:,.4f}%",
        f"No. of Days                   : {Days}",
        f"Winning Days                  : {Winning_Days}",
        f"Losing Days                   : {Losing_Days}",
        f"Best Day                      : {Best_Day}",
        f"Worst Day                     : {Worst_Day}",
        f"Best Day PnL                  : {Best_Day_PnL:,.2f}",
        f"Worst Day PnL                 : {Worst_Day_PnL:,.2f}",
        f"Average Winning Day PnL       : {Average_Winning_Day_PnL:,.2f}",
        f"Average Losing Day PnL        : {Average_Losing_Day_PnL:,.2f}",
        f"Total Trades                  : {Total_Trades}",
        f"Winning Trades                : {Winning_Trades}",
        f"Losing Trades                 : {Losing_Trades}",
        f"Win Rate (%)                  : {Win_Rate:,.2f}",
        f"Average Winning Trade         : {Average_Winning_Trade:,.4f}",
        f"Average Losing Trade          : {Average_Losing_Trade:,.4f}",
        f"Average Hold Period (seconds) : {Avg_Hold_Period_Seconds:,.2f}",
        f"Average Hold Period (minutes) : {Avg_Hold_Period_Minutes:,.2f}",
        f"Median Hold Period (seconds)  : {Median_Hold_Period_Bars:,.2f}",
        f"Min Hold Period (seconds)     : {Min_Hold_Period_Bars:,.2f}",
        f"Max Hold Period (seconds)     : {Max_Hold_Period_Bars:,.2f}",
        f"P25 Hold Period (seconds)     : {P25_Hold:,.2f}",
        f"P75 Hold Period (seconds)     : {P75_Hold:,.2f}",
        f"P90 Hold Period (seconds)     : {P90_Hold:,.2f}",
        f"P95 Hold Period (seconds)     : {P95_Hold:,.2f}",
        "========================",
        "",
    ]
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✓ Text summary saved to '{txt_filename}'")
    # --- end text file save ---

    # # Best10Days = np.argsort(Daily_PnL)[-1:][::-1]  # descending order
    # # Worst10Days = np.argsort(Daily_PnL)[:10]       # ascending order
    # # Worst5Days = np.argsort(Daily_PnL)[:1]

    # # days_to_plot = Worst5Days
    # # # match the feature names created in feature_analysis.py
    # # primary_cols = ["KFTrend", "STL", "SMA_30","SMA_15"]
    # # secondary_cols = [ "ATR_High", "STD_High", "KAMA_Slope_abs"]
    # # Interval_PnL = np.array(Interval_PnL)

    # # # if you have start_time and interval defined (pd.Timedelta)
    # # interval_minutes = 30
    # # interval = pd.Timedelta(minutes=interval_minutes)
    # # start_time = pd.to_timedelta("00:00:00")   # or whatever your session start is

    # # # create readable labels (e.g. '09:15-09:45')
    # # labels = []
    # # for i in range(len(Interval_PnL)):
    # #     t0 = (start_time + i*interval)
    # #     t1 = (start_time + (i+1)*interval)
    # #     labels.append(f"{str(t0).split()[-1]}-{str(t1).split()[-1]}")

    # # if len(days_to_plot) != 0:

    # #     plot_data=df[df["Day"]==days_to_plot[0]]
    # #     for day in days_to_plot[1:]:
    # #         plot_data=pd.concat([plot_data,df[df["Day"]==day]])
        
    # #     # Convert index to datetime if not already
    # #     #plot_data.index = pd.to_datetime(plot_data.index)
    # #     #plot_data = plot_data.sort_index()

    # #     Days=len(days_to_plot)
    # #     # Create subplot structure
    # #     fig = make_subplots(
    # #         rows=3, cols=1,
    # #         shared_xaxes=True,
    # #         vertical_spacing=0.08,
    # #         subplot_titles=(
    # #             'Price vs Strategy Portfolio Value',
    # #             'Price vs Signals',
    # #             'Trades per Interval'
    # #         ),
    # #         specs=[[{"secondary_y": True}],
    # #             [{"secondary_y": True}],
    # #             [{"secondary_y": False}]]
    # #     )
        
    # #     # Plot 1: Portfolio Value vs Price
    # #     fig.add_trace(
    # #         go.Scatter(
    # #             x=plot_data.index,
    # #             y=plot_data['NAV'],
    # #             name='Strategy Portfolio',
    # #             line=dict(color='#1f77b4', width=2),
    # #             hovertemplate='<b>Strategy</b><br>Value: $%{y:,.2f}<br>Time: %{x}<extra></extra>',
    # #             connectgaps=False
    # #         ),
    # #         row=1, col=1, secondary_y=True
    # #     )
        
    # #     fig.add_trace(
    # #         go.Scatter(
    # #             x=plot_data.index,
    # #             y=plot_data['Price'],
    # #             name='Price',
    # #             line=dict(color='#ff7f0e', width=2),
    # #             hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Time: %{x}<extra></extra>',
    # #             connectgaps=False
    # #         ),
    # #         row=1, col=1, secondary_y=False
    # #     )
        
    # #     # Plot 2: Price vs Signals
    # #     fig.add_trace(
    # #         go.Scatter(
    # #             x=plot_data.index,
    # #             y=plot_data['Price'],
    # #             name='Price',
    # #             line=dict(color='#1f77b4', width=2),
    # #             showlegend=False,
    # #             hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
    # #         ),
    # #         row=2, col=1, secondary_y=False
    # #     )
        
    # #     # fig.add_trace(
    # #     #     go.Scatter(
    # #     #         x=plot_data.index,
    # #     #         y=[max(p, 0) for p in plot_data['Position']],
    # #     #         name='Long Position',
    # #     #         mode='none',  # no line markers
    # #     #         fill='tozeroy',
    # #     #         fillcolor='rgba(0, 128, 0, 0.5)',
    # #     #         line_shape='hv',
    # #     #     ),
    # #     #     row=2, col=1, secondary_y=True
    # #     # )

    # #     # fig.add_trace(
    # #     #     go.Scatter(
    # #     #         x=plot_data.index,
    # #     #         y=[min(p, 0) for p in plot_data['Position']],
    # #     #         name='Short Position',
    # #     #         mode='none',
    # #     #         fill='tozeroy',
    # #     #         fillcolor='rgba(255, 0, 0, 0.5)',
    # #     #         line_shape='hv',
    # #     #     ),
    # #     #     row=2, col=1, secondary_y=True
    # #     # )

    # #     fig.add_trace(
    # #         go.Scatter(
    # #             x=plot_data.index,
    # #             y=plot_data['Position'],
    # #             name='Position',
    # #             mode='none',
    # #             fill='tozeroy',
    # #             fillcolor='rgba(0, 255, 0, 0.25)',
    # #             line_shape='hv',
    # #         ),
    # #         row=2, col=1, secondary_y=True
    # #     )
        
    # #     # Additional columns on primary y-axis
    # #     for col in primary_cols:
    # #         if col in plot_data.columns:
    # #             fig.add_trace(
    # #                 go.Scatter(
    # #                     x=plot_data.index,
    # #                     y=plot_data[col],
    # #                     name=f'{col} (Primary)',
    # #                     mode='lines',
    # #                     line=dict(width=1.5),
    # #                     visible=True,  # toggleable via legend
    # #                     hovertemplate=f'<b>{col}</b><br>Value: $%{{y:,.2f}}<br>Time: %{{x}}<extra></extra>'
    # #                 ),
    # #                 row=2, col=1, secondary_y=False
    # #             )

    # #     # Additional columns on secondary y-axis (scaled)
    # #     for col in secondary_cols:
    # #         if col in plot_data.columns:
    # #             # create scaled version for plotting on secondary axis (0-centered)
    # #             series = plot_data[col].replace([np.inf, -np.inf], np.nan).dropna()
    # #             if not series.empty:
    # #                 col_min, col_max = series.min(), series.max()
    # #                 if col_max != col_min:
    # #                     scaled = 2 * (plot_data[col] - col_min) / (col_max - col_min) - 1
    # #                 else:
    # #                     scaled = plot_data[col] * 0

    # #                 fig.add_trace(
    # #                     go.Scatter(
    # #                         x=plot_data.index,
    # #                         y=scaled,
    # #                         name=f'{col} (Scaled)',
    # #                         mode='lines',
    # #                         line=dict(width=1.5),
    # #                         visible=True,
    # #                         hovertemplate=f'<b>{col} (Scaled)</b><br>Scaled Value: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>'
    # #                     ),
    # #                     row=2, col=1, secondary_y=True
    # #                 )
    # #                 # also add raw series (toggleable)
    # #                 fig.add_trace(
    # #                     go.Scatter(
    # #                         x=plot_data.index,
    # #                         y=plot_data[col],
    # #                         name=f'{col} (Raw)',
    # #                         mode='lines',
    # #                         line=dict(width=1.0, dash='dot'),
    # #                         visible='legendonly',
    # #                         hovertemplate=f'<b>{col}</b><br>Value: %{{y:.6f}}<br>Time: %{{x}}<extra></extra>'
    # #                     ),
    # #                     row=2, col=1, secondary_y=True
    # #                 )
        
    # #     # Plot 3: Drawdown
    # #     # fig.add_trace(
    # #     #     go.Scatter(
    # #     #         x=plot_data.index,
    # #     #         y=plot_data['Drawdown'],
    # #     #         name='Strategy Drawdown',
    # #     #         fill='tozeroy',
    # #     #         line=dict(color='#d62728', width=1),
    # #     #         fillcolor='rgba(214, 39, 40, 0.3)',
    # #     #         hovertemplate='<b>Drawdown</b><br>Value: %{y:.2f}%<br>Time: %{x}<extra></extra>'
    # #     #     ),
    # #     #     row=3, col=1
    # #     # )
    # #     # Plot 3: Interval
    # #     fig.add_trace(
    # #         go.Bar(
    # #             x=list(range(len(Interval_PnL))),
    # #             y=Interval_PnL,
    # #             name="Trades per Interval",
    # #             marker_color="#636EFA",
    # #             opacity=0.9
    # #         ),
    # #         row=3, col=1
    # #     )

    # #     # Update x-axes (trim gaps and show HH:MM)
    # #     for row in range(1, 3):
    # #         fig.update_xaxes(
    # #             type='category',          # removes gaps for missing times
    # #             tickangle=0,
    # #             title_text="Time (HH:MM)",
    # #             row=row, col=1
    # #         )
        
    # #     # Update y-axes labels
    # #     fig.update_yaxes(title_text="Portfolio Value", row=1, col=1,secondary_y=True)
    # #     fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    # #     fig.update_yaxes(title_text="Signal", row=2, col=1, secondary_y=True)
    # #     fig.update_yaxes(title_text="Price", row=2, col=1, secondary_y=False)
    # #     fig.update_yaxes(title_text="Trades in Intervals", row=3, col=1)
                
    # #     # Update layout
    # #     fig.update_layout(
    # #         title={
    # #             'text': f'<b>Intraday Backtest Report</b>',
    # #             'x': 0.5,
    # #             'xanchor': 'center',
    # #             'font': {'size': 16}
    # #         },
    # #         height=1400,
    # #         width=1200,
    # #         template='plotly_white',
    # #         hovermode='x unified',
    # #         legend=dict(
    # #             orientation="h",
    # #             yanchor="bottom",
    # #             y=1.02,
    # #             xanchor="right",
    # #             x=1
    # #         )
    # #     )
        
    #     # Save and open HTML file
    #     html_filename = "2_backtest_report.html"
    #     fig.write_html(html_filename, auto_open=True)
        
        # print(f"✓ Interactive report saved to '{html_filename}' and opened in browser")
    print(f"✓ Backtest completed successfully!")
    
    return df

# Example usage
if __name__ == "__main__":
    df=pd.read_csv("/data/quant14/signals/temp_trading_signals_EBY.csv")
    # df=ddf.compute().to_pandas()
    out=backtest(df, 100,0.02)