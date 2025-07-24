import ccxt
from config import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET
from typing import Literal, cast


class BinanceTrader:
    def __init__(self):
        self.exchange = ccxt.binanceusdm({
            'apiKey': BINANCE_TESTNET_API_KEY,
            'secret': BINANCE_TESTNET_API_SECRET,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.exchange.set_sandbox_mode(True)
        self.exchange.options['adjustForTimeDifference'] = True

    def get_position(self, symbol):
        positions = self.exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos['contracts']) > 0:
                return {
                    'side': pos['side'],
                    'amount': float(pos['contracts']),
                    'entry_price': float(pos['entryPrice'])
                }
        return None

    def close_all_positions(self, symbol):
        positions = self.exchange.fetch_positions([symbol])
        for position in positions:
            contracts = float(position['contracts'])
            if contracts > 0:
                close_side = cast(Literal['buy', 'sell'], 'sell' if position['side'].lower() == 'long' else 'buy')
                self.exchange.create_market_order(symbol, close_side, contracts, params={"reduceOnly": True})

    def create_order_with_sl_tp(
            self, symbol, side: Literal['buy', 'sell'], balance, risk_percent, atr,
            sl_multiplier, tp_multiplier, adaptive=True
    ):
        current_price = self.exchange.fetch_ticker(symbol)['last']

        position_size = self.calculate_position_size(symbol, risk_percent, balance, atr, current_price)

        tp_price = current_price + tp_multiplier * atr if side == 'buy' else current_price - tp_multiplier * atr
        sl_price = current_price - sl_multiplier * atr if side == 'buy' else current_price + sl_multiplier * atr

        opposite_side: Literal['buy', 'sell'] = cast(Literal['buy', 'sell'], 'sell' if side == 'buy' else 'buy')

        self.cancel_all_orders(symbol)

        self.exchange.create_market_order(symbol, side, position_size)

        self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=opposite_side,
            amount=position_size,
            params={'stopPrice': tp_price, 'closePosition': True}
        )

        self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=opposite_side,
            amount=position_size,
            params={'stopPrice': sl_price, 'closePosition': True}
        )

        print(f"✅ Позиция открыта {side.upper()}, TP: {tp_price:.2f}$, SL: {sl_price:.2f}$")

    def cancel_all_orders(self, symbol):
        self.exchange.cancel_all_orders(symbol)

    def calculate_position_size(self, symbol, balance, risk_percent, atr, current_price):
        position_size_usd = balance * risk_percent
        position_size = position_size_usd / (atr if atr > 0 else current_price * 0.01)

        min_order_size = self.get_min_order_amount(symbol, current_price)
        return max(round(position_size, 4), min_order_size)

    def get_min_order_amount(self, symbol, current_price):
        self.exchange.load_markets()
        market_info = self.exchange.market(symbol)
        min_notional = float(market_info['info']['filters'][5]['notional'])
        min_amount = min_notional / current_price
        return round(min_amount, 4)

    def update_trailing_stop(self, symbol, entry_price, atr, side: Literal['buy', 'sell'], current_price):
        trail_distance = atr * 1.5

        if side == 'buy':
            new_sl = current_price - trail_distance
            if new_sl > entry_price:
                self.cancel_all_orders(symbol)
                self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side='sell',
                    amount=self.get_position(symbol)['amount'],
                    params={'stopPrice': new_sl, 'closePosition': True}
                )

        elif side == 'sell':
            new_sl = current_price + trail_distance
            if new_sl < entry_price:
                self.cancel_all_orders(symbol)
                self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side='buy',
                    amount=self.get_position(symbol)['amount'],
                    params={'stopPrice': new_sl, 'closePosition': True}
                )
