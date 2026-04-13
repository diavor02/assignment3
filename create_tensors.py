import numpy as np

def create_weather_sequences(weather, S=48, horizon=24):
    X_weather = []

    for t in range(len(weather) - (S + horizon)):
        w_seq = weather[t : t + S + horizon]  # (S+24, H, W, C)
        X_weather.append(w_seq)

    return np.array(X_weather)

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
