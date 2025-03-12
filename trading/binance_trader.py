import csv
import os
from datetime import datetime, timedelta
import ccxt
import pandas as pd
from config import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET
from typing import Literal


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

    def get_trade_history(self, symbol, limit=20):
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            return [
                {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'amount': float(trade['amount']),
                    'price': float(trade['price']),
                    'profit': float(trade['info'].get('realizedPnl', '0') or 0)  # ← исправлено здесь
                }
                for trade in trades
            ]
        except Exception as e:
            print(f"⚠️ Ошибка получения истории сделок по {symbol}: {e}")
            return []

    def get_position(self, symbol='BTC/USDT'):
        positions = self.exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos['contracts']) > 0:
                return {
                    'side': pos['side'],
                    'amount': float(pos['contracts']),
                    'entry_price': float(pos['entryPrice'])
                }
        return None

    from typing import Literal

    def get_order_side(self, side: str) -> Literal['buy', 'sell']:
        if side.lower() == 'long':
            return 'sell'
        else:
            return 'buy'

    def close_position(self, symbol, side, amount):
        positions = self.exchange.fetch_positions([symbol])
        position = next((pos for pos in positions if pos['symbol'] == symbol and float(pos['contracts']) > 0), None)

        if position:
            entry_price = float(position['entryPrice'])
            order_side = self.get_order_side(side)

            exit_order = self.exchange.create_market_order(
                symbol=symbol,
                side=order_side,
                amount=amount,
                params={"reduceOnly": True}
            )

            exit_price = float(exit_order['average'])
            self.log_trade(symbol, side.upper(), entry_price, exit_price, amount)

    def get_close_side(self, position_side: str) -> Literal['buy', 'sell']:
        if position_side.lower() == 'long':
            return 'sell'
        else:
            return 'buy'

    def close_all_positions(self, symbol):
        positions = self.exchange.fetch_positions([symbol])
        for position in positions:
            contracts = float(position['contracts'])
            if contracts > 0:
                close_side = self.get_order_side(position['side'])

                exit_order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=close_side,
                    amount=contracts,
                    params={"reduceOnly": True}
                )

                exit_price = float(exit_order['average'])
                entry_price = float(position['entryPrice'])

                self.log_trade(
                    symbol=symbol,
                    side=position['side'].upper(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    amount=contracts
                )

        print(f"✅ Все позиции по {symbol} закрыты и записаны в метрики.")

    def cancel_all_orders(self, symbol):
        return self.exchange.cancel_all_orders(symbol)

    def create_order_with_sl_tp(
            self,
            symbol: str,
            side: Literal['buy', 'sell'],
            balance: float = None,
            risk_percent: float = None,
            atr: float = None,
            amount: float = None,
            sl_multiplier: float = 1.5,
            tp_multiplier: float = 3,
            take_profit_pct: float = 0.03,
            stop_loss_pct: float = 0.01,
            adaptive: bool = False
    ):
        try:
            current_price = self.exchange.fetch_ticker(symbol)['last']

            if adaptive:
                if balance is None or risk_percent is None or atr is None:
                    raise ValueError("Для адаптивного режима необходимы balance, risk_percent и atr.")

                amount = self.calculate_position_size(symbol, risk_percent, balance, atr)
                tp_price = (current_price + tp_multiplier * atr) if side == 'buy' else (
                            current_price - tp_multiplier * atr)
                sl_price = current_price - sl_multiplier * atr if side == 'buy' else current_price + sl_multiplier * atr
            else:
                if amount is None or amount <= 0:
                    raise ValueError("Необходимо явно указать amount при фиксированном режиме торговли.")

                tp_price = current_price * (1 + take_profit_pct if side == 'buy' else 1 - take_profit_pct)
                sl_price = current_price * (1 - stop_loss_pct if side == 'buy' else 1 + stop_loss_pct)

            opposite_side: Literal['buy', 'sell'] = 'sell' if side == 'buy' else 'buy'

            self.cancel_all_orders(symbol)

            # Открываем основной ордер
            self.exchange.create_market_order(symbol, side, amount)

            # TP (Take-Profit Market)
            self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=opposite_side,
                amount=amount,
                params={'stopPrice': tp_price, 'closePosition': True}
            )

            # SL (Stop Market)
            self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=opposite_side,
                amount=amount,
                params={'stopPrice': sl_price, 'closePosition': True}
            )

            print(f"✅ Позиция {side.upper()} открыта. TP: {tp_price:.2f}$, SL: {sl_price:.2f}$, объём: {amount}")

        except Exception as e:
            print(f"❌ Ошибка при открытии позиции с TP и SL: {e}")


    def get_pnl(self, symbol: str) -> float:
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) != 0:
                    entry_price = float(pos['entryPrice'])
                    current_price = float(pos['markPrice'])
                    side = pos['side']

                    if side.lower() == 'long':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100

                    return round(pnl_percent, 2)
            return 0.0
        except Exception as e:
            print(f"⚠️ Ошибка получения PNL для {symbol}: {e}")
            return 0.0

    def log_trade(self, symbol, side, entry_price, exit_price, amount):
        pnl_percent = ((exit_price - entry_price) / entry_price * 100) if side.lower() == 'buy' \
            else ((entry_price - exit_price) / entry_price * 100)

        file_exists = os.path.exists('bot_metrics.csv')

        with open('bot_metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'amount', 'pnl_percent'])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side, entry_price, exit_price, amount, round(pnl_percent, 2)
            ])

    def get_pnl_period(self, symbol, period='day'):
        if not os.path.exists('trades.csv'):
            return 0.0
        try:
            trades_df = pd.read_csv('trades.csv', parse_dates=['timestamp'])
            trades_df = trades_df[trades_df['symbol'] == symbol]

            now = datetime.now()

            if period == 'day':
                start_date = now - timedelta(days=1)
            elif period == 'week':
                start_date = now - timedelta(weeks=1)
            elif period == 'month':
                start_date = now - timedelta(days=30)
            elif period == 'year':
                start_date = now - timedelta(days=365)
            else:
                return 0.0

            period_trades = trades_df[trades_df['timestamp'] >= start_date]
            total_pnl = period_trades['pnl_percent'].sum()

            return round(total_pnl, 2)

        except Exception as e:
            print(f"⚠️ Ошибка при расчете PNL: {e}")
            return 0.0

    def get_min_order_amount(self, symbol, current_price):
        self.exchange.load_markets()
        market_info = self.exchange.market(symbol)
        min_notional = float(market_info['info']['filters'][5]['notional'])
        min_amount = min_notional / current_price
        return round(min_amount, 4)

    def calculate_position_size(self, symbol, risk_percent, balance, atr):
        current_price = self.exchange.fetch_ticker(symbol)['last']
        position_size = (balance * risk_percent) / atr

        min_amount = self.get_min_order_amount(symbol, current_price)
        if position_size < min_amount:
            position_size = min_amount  # можешь добавить логику пропуска сделки, если позиция слишком мала

        return round(position_size, 4)