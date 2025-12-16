import sys
from datetime import datetime
import pandas as pd

from alpha_research import BacktesterIIT, Side, Ticker

ebx_day = 0
ebx_last_ts = -1

eby_day = 0
eby_last_ts = -1

EBX_Path = "/data/quant14/signals/combined_signals_EBX2.csv"
EBY_Path = "/data/quant14/signals/combined_signals_EBY_copy.csv"

def load_signals(path, name):
    try:
        df = pd.read_csv(path, usecols=["Time", "Price", "Signal"])
        
        temp_time = pd.to_timedelta(df["Time"])
        
        df["Day"] = (temp_time.diff() < pd.Timedelta(0)).cumsum().fillna(0).astype(int)
        
        df.set_index(["Day", "Time"], inplace=True)
        
        print(f"{name} loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        print(f"No {name} signal file found: {e}")
        return None

ebx_df = load_signals(EBX_Path, "EBX")
eby_df = load_signals(EBY_Path, "EBY")

def Get_Signal(df, day, ts):
    if df is None:
        return 0
    try:
        return int(df.at[(day, ts), "Signal"])
    except KeyError:
        return 0
    except Exception:
        return 0

def my_broadcast_callback(state, ts):
    global ebx_day, ebx_last_ts
    global eby_day, eby_last_ts

    h, m, s = int(ts[0:2]), int(ts[3:5]), int(ts[6:8])
    curr_time = h * 3600 + m * 60 + s

    if curr_time <= ebx_last_ts:
        ebx_day += 1
    
    ebx_last_ts = curr_time
    ebx_signal = Get_Signal(ebx_df, ebx_day, ts)

    if curr_time <= eby_last_ts:
        eby_day += 1
        
    eby_last_ts = curr_time
    eby_signal = Get_Signal(eby_df, eby_day, ts)

    if ebx_signal != 0:
        backtest.place_order(
            ticker="EBX",
            qty=abs(ebx_signal)*100,
            side=Side.BUY if ebx_signal > 0 else Side.SELL
        )
    
    if eby_signal != 0:
        backtest.place_order(
            ticker="EBY",
            qty=abs(eby_signal)*100,
            side=Side.BUY if eby_signal > 0 else Side.SELL
        )

def on_timer(ts):
    print("On timer callback")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    backtest = BacktesterIIT(config_file)
    print(datetime.now().strftime("%H:%M:%S")) 
    backtest.run(broadcast_callback=my_broadcast_callback, timer_callback=on_timer)
    print(datetime.now().strftime("%H:%M:%S"))