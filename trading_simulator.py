from dataclasses import dataclass, field
from typing import Optional, Any
import collections
import queue
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@dataclass
class Order:  # Our own placed order
    order_id: int
    side: str
    size: float
    price: float
    timestamp: float


@dataclass
class AnonTrade:  # Market trade
    timestamp: float
    side: str
    size: float
    price: str


@dataclass
class OwnTrade:  # Execution of own placed order
    timestamp: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    timestamp: float
    asks: list[tuple[float, float]]  # tuple[price, size]
    bids: list[tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trades: Optional[list[AnonTrade]] = None


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class Strategy:
    def __init__(self, max_position: float, t_0: float, maker_fee: float = 0) -> None:
        self.max_position = max_position
        self.maker_fee = maker_fee
        self.order_cnt = 0
        self.active_orders = dict()
        self.cur_position = 0
        self.t_0 = t_0 * 1e6
        self.best_ask = self.best_bid = 0
        self.pnl = 0
        self.pnls = []
        self.timestamps = []

    def run(self, sim: "Sim", max_iter: int = 100000):
        iter = 0
        while iter <= max_iter:
            try:
                tick_result = sim.tick()
                timestamp, md_update = tick_result.priority, tick_result.item
                if isinstance(md_update, MdUpdate):
                    if md_update.trades is None:
                        self.best_bid = md_update.orderbook.bids[0][0]
                        self.best_ask = md_update.orderbook.asks[0][0]
                    else:
                        price = md_update.trades[0].price
                        if md_update.trades[0].side == 'ASK':
                            self.best_bid = price
                        else:
                            self.best_ask = price
                elif isinstance(md_update, OwnTrade):
                    if md_update.side == 'ASK':
                        self.best_bid = md_update.price
                        self.pnl += md_update.price * md_update.size
                    else:
                        self.best_ask = md_update.price
                        self.pnl -= md_update.price * md_update.size
                    if self.active_orders.pop(md_update.order_id, None) is None:
                        if md_update.side == 'ASK':
                            self.cur_position -= md_update.size
                        else:
                            self.cur_position += md_update.size
                orders_to_cancel = []
                for order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    if order.timestamp + self.t_0 < timestamp:
                        sim.cancel_order(order.order_id, timestamp)
                        orders_to_cancel.append(order.order_id)
                for order_id in orders_to_cancel:
                    order_to_cancel = self.active_orders.pop(order_id)
                    if order_to_cancel.side == 'ASK':
                        self.cur_position += order_to_cancel.size
                    else:
                        self.cur_position -= order_to_cancel.size
                side = random.choice(['ASK', 'BID'])
                if side == 'ASK' and self.cur_position > -self.max_position or side == 'BID' and \
                        self.cur_position == self.max_position:
                    self.place_ask(timestamp, sim)
                elif side == 'BID' and self.cur_position < self.max_position or side == 'ASK' and \
                        self.cur_position == -self.max_position:
                    self.place_bid(timestamp, sim)
                self.order_cnt += 1
                if len(self.timestamps) > 0 and timestamp == self.timestamps[-1]:
                    self.pnls[-1] = self.pnl
                else:
                    self.pnls.append(self.pnl)
                    self.timestamps.append(timestamp)
            except StopIteration:
                break
            iter += 1

    def place_ask(self, timestamp, sim: "Sim"):
        size = random.uniform(0.001, math.fabs(-self.max_position - self.cur_position))
        self.cur_position -= size
        self.active_orders[self.order_cnt] = Order(self.order_cnt, 'ASK', size, self.best_ask, timestamp)
        sim.place_order(Order(self.order_cnt, 'ASK', size, self.best_ask, timestamp))

    def place_bid(self, timestamp, sim: "Sim"):
        size = random.uniform(0.001, self.max_position - self.cur_position)
        self.cur_position += size
        self.active_orders[self.order_cnt] = Order(self.order_cnt, 'BID', size, self.best_bid, timestamp)
        sim.place_order(Order(self.order_cnt, 'BID', size, self.best_bid, timestamp))


class MdUpdateQueue:
    def __init__(self, trades, lobs, data_prefix=''):
        self.trades = trades
        self.lobs = lobs
        self.len = len(trades) + len(lobs)
        self.trade_ind = 0
        self.lob_ind = 0
        self.data_prefix = data_prefix

    def __len__(self):
        return self.len - self.lob_ind - self.trade_ind

    def pop(self, delete=True) -> tuple[float, MdUpdate]:
        if self.trade_ind < len(self.trades) and (
                self.lob_ind == len(self.lobs) or self.trades['exchange_ts'][self.trade_ind] <=
                self.lobs['exchange_ts'][self.lob_ind]):
            trade = self.trades.iloc[self.trade_ind]
            self.trade_ind += int(delete)
            return trade['exchange_ts'], MdUpdate(None, [
                AnonTrade(trade['exchange_ts'], trade['aggro_side'], trade['size'], trade['price'])])
        elif self.lob_ind < len(self.lobs) and (
                self.trade_ind == len(self.trades) or self.lobs['exchange_ts'][self.lob_ind] <=
                self.trades['exchange_ts'][self.trade_ind]):
            lob = self.lobs.iloc[self.lob_ind]
            self.lob_ind += int(delete)
            asks = []
            bids = []
            for i in range(10):
                asks.append((lob[self.data_prefix + '_ask_price_{}'.format(i)],
                             lob[self.data_prefix + '_ask_vol_{}'.format(i)]))
                bids.append((lob[self.data_prefix + '_bid_price_{}'.format(i)],
                             lob[self.data_prefix + '_bid_vol_{}'.format(i)]))

            return lob['exchange_ts'], MdUpdate(OrderbookSnapshotUpdate(lob['exchange_ts'], asks, bids), None)


def load_md_from_file(path_trades: str, path_lobs: str, prefix: str) -> MdUpdateQueue:
    trades = pd.read_csv(path_trades, skipinitialspace=True)
    lobs = pd.read_csv(path_lobs, skipinitialspace=True)
    return MdUpdateQueue(trades, lobs, prefix)


class Sim:
    def __init__(self, execution_latency: float, md_latency: float, path_trades: str, path_lobs: str,
                 prefix: str) -> None:
        self.md_latency = md_latency * 1e6
        self.execution_latency = execution_latency * 1e6
        self.md_queue = load_md_from_file(path_trades, path_lobs, prefix)
        self.strategy_updates_queue = queue.PriorityQueue()
        self.actions_queue = collections.deque()
        self.active_orders = dict()
        self.trade_cnt = 0
        self.best_bid = self.best_ask = 0

    def tick(self):
        while True:
            md_event_time = self.md_queue.pop(False)[0] if len(self.md_queue) > 0 else math.inf
            actions_event_time = self.actions_queue[0][0] if len(self.actions_queue) > 0 else math.inf
            strategy_event_time = self.strategy_updates_queue.queue[
                0].priority if self.strategy_updates_queue.qsize() > 0 else math.inf

            if md_event_time == math.inf and actions_event_time == math.inf:
                break

            if md_event_time <= strategy_event_time and md_event_time <= actions_event_time:
                md_update = self.md_queue.pop()
                timestamp = md_update[0]
                if md_update[1].trades is None:
                    self.best_bid = md_update[1].orderbook.bids[0][0]
                    self.best_ask = md_update[1].orderbook.asks[0][0]
                else:
                    price = md_update[1].trades[0].price
                    if md_update[1].trades[0].side == 'ASK':
                        self.best_bid = price
                    else:
                        self.best_ask = price
                self.strategy_updates_queue.put(PrioritizedItem(timestamp + self.md_latency, md_update[1]))
            elif actions_event_time <= md_event_time and actions_event_time <= strategy_event_time:
                self.prepare_orders(self.actions_queue.popleft())
            else:
                return self.strategy_updates_queue.get()
            timestamp = min(actions_event_time, md_event_time)
            self.execute_orders(timestamp)

    def prepare_orders(self, action):
        if isinstance(action[1], Order):
            self.active_orders[action[1].order_id] = action[1]
        else:
            self.active_orders.pop(action[1], None)

    def execute_orders(self, timestamp):
        orders_to_delete = []
        for order_id in self.active_orders:
            order = self.active_orders[order_id]
            if order.side == 'ASK' and order.price <= self.best_bid or order.side == 'BID' and order.price >= self.best_ask:
                self.strategy_updates_queue.put(
                    PrioritizedItem(timestamp + self.md_latency,
                                    OwnTrade(timestamp + self.md_latency, self.trade_cnt, order.order_id,
                                             order.side, order.size, order.price)))
                orders_to_delete.append(order.order_id)
                self.trade_cnt += 1
        for order_id in orders_to_delete:
            self.active_orders.pop(order_id)

    def place_order(self, order):
        order.timestamp += self.execution_latency
        self.actions_queue.append((order.timestamp, order))

    def cancel_order(self, order_id, timestamp):
        timestamp += self.execution_latency
        self.actions_queue.append((timestamp, order_id))


if __name__ == "__main__":
    strategy = Strategy(10, 100)
    sim = Sim(50, 50, 'md/btcusdt_Binance_LinearPerpetual/trades.csv', 'md/btcusdt_Binance_LinearPerpetual/lobs.csv',
              'btcusdt:Binance:LinearPerpetual')
    strategy.run(sim)

    # save PNL plot
    sns.set_style('darkgrid')
    plt.figure(figsize=(16, 9))
    plt.plot(strategy.timestamps, strategy.pnls)
    plt.title('PNL over time')
    plt.xlabel('Time')
    plt.ylabel('PNL')
    plt.savefig('PNL.jpg')
    plt.show()
