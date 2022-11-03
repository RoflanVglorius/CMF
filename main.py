from simulator.load_data import  load_md_from_file
import pandas as pd
from simulator.simulator import Sim
from simulator.strategy import StoikovStrategy
from simulator.get_info import get_pnl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    md = load_md_from_file(path='md/btcusdt_Binance_LinearPerpetual/', nrows=10 ** 5)
    latency = pd.Timedelta(10, 'ms').delta
    md_latency = pd.Timedelta(10, 'ms').delta

    sim = Sim(md, latency, md_latency)

    delay = pd.Timedelta(0.1, 's').delta
    hold_time = pd.Timedelta(10, 's').delta
    gamma = 0.5
    sigma = 10
    horizon_const = 1
    intensity = 3

    strategy = StoikovStrategy(delay, gamma, sigma, horizon_const, intensity, hold_time)
    trades_list, md_list, updates_list, all_orders = strategy.run(sim)
    df = get_pnl(updates_list)
    PnL = df.total

    # save PNL plot
    sns.set_style('darkgrid')
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(0, len(PnL)), PnL)
    plt.title('PNL over time')
    plt.xlabel('Time')
    plt.ylabel('PNL')
    plt.savefig('PNL.jpg')
    plt.show()