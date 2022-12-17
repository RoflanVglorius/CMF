from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from datetime import datetime

from .simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class StoikovStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''

    def __init__(self, delay: float, gamma: float = 0, horizon_const: float = 1, intensity: float = 1) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                gamma(float): risk parameter (when gamma = 0 our strategy is same to symmetric)
                sigma(float): volatility
                horizon_const(float): constant replacing (T - t)
                intensity(float): parameter k from paper (intensity of trades)
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.history = {'ask_price': [], 'bid_price': [], 'mid_price': [], 'indifference_price': [], 'inventory': [],
                        'pnl': [0.], 'time': []}
        self.delay = delay
        self.gamma = gamma
        self.horizon_const = horizon_const
        self.intensity = intensity
        self.sigmas = []
        self.means = []
        self.squared_means = []
        self.cur_pos = 0

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf
        indiff_bid = best_bid
        indiff_ask = best_ask
        indifference_price = 0
        spread = 0
        mean = 0
        mean_squared = 0
        counter = 0
        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                    self.history['pnl'].append(self.history['pnl'][-1])
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.side == 'ASK':
                        self.cur_pos -= update.size
                        self.history['pnl'].append(self.history['pnl'][-1] + update.size * update.price)
                    else:
                        self.cur_pos += update.size
                        self.history['pnl'].append(self.history['pnl'][-1] - update.size * update.price)
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else:
                    assert False, 'invalid type of update!'
                if counter < 1000:  # смотрим дисперсию за последние 1000 апдейтов
                    mean = (mean * counter + (best_ask + best_bid) / 2) / (counter + 1)
                    mean_squared = (mean_squared * counter + ((best_ask + best_bid) / 2) ** 2) / (counter + 1)
                if counter >= 1000:
                    mean = mean * 1000 - self.history['mid_price'][counter - 1000] + (best_ask + best_bid) / 2
                    mean /= 1000
                    mean_squared = mean_squared * 1000 - self.history['mid_price'][counter - 1000] ** 2 + (
                            (best_ask + best_bid) / 2) ** 2
                    mean_squared /= 1000
                counter += 1
                self.sigma = np.sqrt(np.abs(mean_squared - mean ** 2))
                self.sigmas.append(self.sigma)
                self.means.append(mean)
                self.squared_means.append(mean_squared)
                indifference_price = (best_bid + best_ask) / 2 - \
                                     self.cur_pos / 0.001 * self.gamma * self.sigma * self.horizon_const
                spread = self.gamma * self.sigma * self.horizon_const + 2 / self.gamma * np.log(
                    1 + self.gamma / self.intensity)
                indiff_bid = np.round(indifference_price - spread / 2, 1)
                indiff_ask = np.round(indifference_price + spread / 2, 1)
                self.history['ask_price'].append(indiff_ask)
                self.history['bid_price'].append(indiff_bid)
                self.history['indifference_price'].append(indifference_price)
                self.history['mid_price'].append((best_bid + best_ask) / 2)
                self.history['inventory'].append(self.cur_pos)
                self.history['time'].append(pd.to_datetime(update.receive_ts))
            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', indiff_bid)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', indiff_ask)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.delay:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders


class FutureMidPriceStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''

    def __init__(self, delay: float, future_data, commission: float = 0,
                 position_strategy: str = None, position_interval: tuple[float, float] = None,
                 price_strategy: str = None, market_strategy: str = None, gamma: float = None,
                 intensity: float = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
                position_strategy(str): strategy for position liquidation (weights, one_side)
                position_interval(tuple[float, float]): limits for position
                in one_side and cur_pos_weighted strategies [ask, bid]
        '''
        self.delay = delay
        self.future_data = future_data
        self.history = {'ask_price': [], 'bid_price': [], 'mid_price': [], 'inventory': [], 'pnl': [0.], 'time': [],
                        'indifference_price': []}
        self.position_strategy = position_strategy
        self.cur_pos = 0
        self.cur_fut_ind = 0
        self.commission = commission
        self.position_strategy = position_strategy
        self.position_interval = position_interval
        self.price_strategy = price_strategy
        self.market_strategy = market_strategy
        self.intensity = intensity
        self.gamma = gamma

    def look_at_future(self, current_time, delta):
        while self.cur_fut_ind < len(self.future_data) and \
                (self.future_data[self.cur_fut_ind].receive_ts - current_time) / 1e9 < delta:
            self.cur_fut_ind += 1
        if self.cur_fut_ind == len(self.future_data):
            return None
        return self.future_data[self.cur_fut_ind]

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                all_orders(List[Sorted]): list of all placed orders
        '''

        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf
        bid = best_bid
        ask = best_ask

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        counter = 0
        mean = 0
        mean_squared = 0
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    md_list.append(update)
                    self.history['pnl'].append(self.history['pnl'][-1])
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                    if update.side == 'ASK':
                        self.cur_pos -= update.size
                        self.history['pnl'].append(self.history['pnl'][-1] + update.size * update.price -
                                                   self.commission * update.size * update.price)
                    else:
                        self.cur_pos += update.size
                        self.history['pnl'].append(self.history['pnl'][-1] - update.size * update.price -
                                                   self.commission * update.size * update.price)
                else:
                    assert False, 'invalid type of update!'

                future_update = self.look_at_future(update.receive_ts, 1)
                if future_update is None:
                    future_update = update
                if isinstance(future_update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, future_update)
                if counter < 1000:  # смотрим дисперсию за последние 1000 апдейтов
                    mean = (mean * counter + (best_ask + best_bid) / 2) / (counter + 1)
                    mean_squared = (mean_squared * counter + ((best_ask + best_bid) / 2) ** 2) / (counter + 1)
                if counter >= 1000:
                    mean = mean * 1000 - self.history['mid_price'][counter - 1000] + (best_ask + best_bid) / 2
                    mean /= 1000
                    mean_squared = mean_squared * 1000 - self.history['mid_price'][counter - 1000] ** 2 + (
                            (best_ask + best_bid) / 2) ** 2
                    mean_squared /= 1000
                counter += 1
                self.sigma = np.sqrt(np.abs(mean_squared - mean ** 2))
                bid, ask = self.get_prices(best_bid, best_ask)
                self.history['ask_price'].append(ask)
                self.history['bid_price'].append(bid)
                self.history['mid_price'].append((best_bid + best_ask) / 2)
                self.history['inventory'].append(self.cur_pos)
                self.history['time'].append(pd.to_datetime(update.receive_ts))

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                bid_order, ask_order = self.place_order(sim, receive_ts, bid, ask)
                market_order = self.place_market_order(sim, receive_ts, best_bid, best_ask)
                if bid_order is not None:
                    ongoing_orders[bid_order.order_id] = bid_order
                if ask_order is not None:
                    ongoing_orders[ask_order.order_id] = ask_order
                if market_order is not None:
                    ongoing_orders[market_order.order_id] = market_order
                all_orders += [bid_order, ask_order, market_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.delay:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders

    def place_market_order(self, sim, receive_ts, best_bid, best_ask):
        if self.market_strategy is None:
            return None
        elif self.market_strategy == 'all_pos':
            if self.cur_pos < 0:
                return sim.place_order(receive_ts, np.abs(self.cur_pos), 'BID', best_bid)
            elif self.cur_pos >= 0:
                return sim.place_order(receive_ts, self.cur_pos, 'ASK', best_ask)
        elif self.market_strategy == 'min_trade':
            if self.cur_pos < 0:
                return sim.place_order(receive_ts, 0.001, 'BID', best_bid)
            elif self.cur_pos >= 0:
                return sim.place_order(receive_ts, 0.001, 'ASK', best_ask)

    def get_prices(self, best_bid, best_ask):
        if self.price_strategy is None:
            bid_price = best_bid
            ask_price = best_ask
        elif self.price_strategy == 'sale':
            bid_price = np.round(best_bid * (1 - np.sign(self.cur_pos) / 10000), 1)
            ask_price = np.round(best_ask * (1 - np.sign(self.cur_pos) / 10000), 1)
        elif self.price_strategy == 'stoikov':
            indifference_price = (best_bid + best_ask) / 2 - \
                                 self.cur_pos / 0.001 * self.gamma * self.sigma
            spread = self.gamma * self.sigma + 2 / self.gamma * np.log(
                1 + self.gamma / self.intensity)
            bid_price = np.round(indifference_price - spread / 2, 1)
            ask_price = np.round(indifference_price + spread / 2, 1)
        else:
            raise Exception("Invalid price strategy")
        return bid_price, ask_price

    def place_order(self, sim, receive_ts, bid_price, ask_price) -> tuple[Order, Order]:
        ask_size = bid_size = None
        if self.position_strategy is None:
            bid_size = 0.001
            ask_size = 0.001
        elif self.position_strategy == 'cur_pos_percentage':
            bid_size = np.round(max(0.001 - self.cur_pos / 100, 0), 3)
            ask_size = np.round(max(0.001 + self.cur_pos / 100, 0), 3)
        elif self.position_strategy == 'exp_weights':
            bid_size = np.round(0.001 * np.exp(-self.cur_pos / 0.01), 3)
            ask_size = np.round(0.001 * np.exp(self.cur_pos / 0.01), 3)
        elif self.position_strategy == 'сur_pos_weighted':
            if self.cur_pos > 0:
                k = min(1., np.abs(self.position_interval[1] / self.cur_pos))
                bid_size = np.round(0.001 * k, 3)
                ask_size = np.round(0.001 / k, 3)
            elif self.cur_pos < 0:
                k = min(1., np.abs(self.position_interval[0] / self.cur_pos))
                bid_size = np.round(0.001 / k, 3)
                ask_size = np.round(0.001 * k, 3)
            else:
                bid_size = 0.001
                ask_size = 0.001
        elif self.position_strategy == 'one_side':
            if self.cur_pos >= 0:
                bid_size = 0.001 if self.cur_pos < self.position_interval[1] else 0
                ask_size = 0.001
            if self.cur_pos < 0:
                bid_size = 0.001
                ask_size = 0.001 if self.cur_pos > self.position_interval[0] else 0
        else:
            raise Exception('Invalid position liquidation strategy')
        bid_order = sim.place_order(receive_ts, bid_size, 'BID', bid_price)
        ask_order = sim.place_order(receive_ts, ask_size, 'ASK', ask_price)
        return bid_order, ask_order


class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''

    def __init__(self, delay: float) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        self.history = {'ask_price': [], 'bid_price': [], 'mid_price': [], 'inventory': [], 'pnl': [0.], 'time': []}
        self.cur_pos = 0

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.history['pnl'].append(self.history['pnl'][-1])
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                    if update.side == 'ASK':
                        self.cur_pos -= update.size
                        self.history['pnl'].append(self.history['pnl'][-1] + update.size * update.price)
                    else:
                        self.cur_pos += update.size
                        self.history['pnl'].append(self.history['pnl'][-1] - update.size * update.price)
                else:
                    assert False, 'invalid type of update!'

                self.history['ask_price'].append(best_ask)
                self.history['bid_price'].append(best_bid)
                self.history['mid_price'].append((best_ask + best_bid) / 2)
                self.history['inventory'].append(self.cur_pos)
                self.history['time'].append(pd.to_datetime(update.receive_ts))

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', best_bid)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', best_ask)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.delay:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders
