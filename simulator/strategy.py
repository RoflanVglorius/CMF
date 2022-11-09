from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

from .simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class StoikovStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''

    def __init__(self, delay: float, gamma: float = 0, sigma: float = 1, horizon_const: float = 1, intensity: float = 1,
                 hold_time: Optional[float] = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                gamma(float): risk parameter (when gamma = 0 our strategy is same to symmetric)
                sigma(float): volatility
                horizon_const(float): constant replacing (T - t)
                intensity(float): parameter k from paper (intensity of trades)
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        self.gamma = gamma
        self.sigma = sigma
        self.horizon_const = horizon_const
        self.intensity = intensity
        if hold_time is None:
            hold_time = max(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time
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
        indifference_price = 0
        spread = 0
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
                    indifference_price = (best_bid + best_ask) / 2 - \
                                         self.cur_pos * self.gamma * self.sigma ** 2 * self.horizon_const
                    spread = self.gamma * self.sigma ** 2 * self.horizon_const + 2 / self.gamma * np.log(
                        1 + self.gamma / self.intensity)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.side == 'ask':
                        self.cur_pos -= update.size
                    else:
                        self.cur_pos += update.size
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else:
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', indifference_price - spread / 2)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', indifference_price + spread / 2)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders
