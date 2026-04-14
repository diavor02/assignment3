import numpy as np
import pandas as pd
import torch

def create_sequences(demand, calendar, S=48, horizon=24):
    X_demand = []
    X_calendar = []

    T_total = len(demand)

    for t in range(T_total - (S + horizon)):
        # past demand
        d_seq = demand[t : t + S]                      # (S, Z)

        # calendar for past + future
        c_seq = calendar[t : t + S + horizon]          # (S+24, C_cal)

        X_demand.append(d_seq)
        X_calendar.append(c_seq)

    return np.array(X_demand), np.array(X_calendar)

df = pd.read_csv("demand_calendar.csv")

demand_cols = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]

calendar_cols = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos"
]

demand_array = df[demand_cols].values        # (T_total, Z)
calendar_array = df[calendar_cols].values    # (T_total, C_cal)

demand_seq, calendar_seq = create_sequences(
    demand_array,
    calendar_array,
    S=48,
    horizon=24
)

# to torch
demand_tensor   = torch.tensor(demand_seq, dtype=torch.float32)
calendar_tensor = torch.tensor(calendar_seq, dtype=torch.float32)
