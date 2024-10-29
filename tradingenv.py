import gym
from gym import spaces
import backtrader as bt
import pandas as pd
import numpy as np
import random
import re
from binance.client import Client

class TradingEnv(gym.Env):
    """
    A trading environment for OpenAI Gym that simulates spot trading with Binance fees.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000, lookback_window_size=50, trading_interval='6h', use_bnb_discount=False):
        super(TradingEnv, self).__init__()

        self.data = data  # The historical price data (DataFrame)
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.trading_interval = trading_interval
        self.use_bnb_discount = use_bnb_discount

        # Define action space: Continuous action between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: OHLCV data for the lookback window
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback_window_size, data.shape[1]), dtype=np.float32
        )

        self.reset()

    def _get_interval_hours(self):
        match = re.match(r'(\d+)([Hh])', self.trading_interval)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            if unit.lower() == 'h':
                return num
        raise ValueError('Invalid trading_interval format')

    def reset(self, training=True):
        if training:
            days = random.randint(15, 45)
            self.initial_balance = random.randint(5, 100000)
        else:
            days = 365
            self.initial_balance = 100


        interval_hours = self._get_interval_hours()
        steps_per_day = 24 // interval_hours
        episode_length = days * steps_per_day
        max_start = len(self.data) - self.lookback_window_size - episode_length
        self.start_index = random.randint(self.lookback_window_size,  max_start)

        if not training:
            self.start_index = max_start - days

        self.end_index = self.start_index + episode_length

        self.current_step = self.start_index

        self.balance = float(self.initial_balance)
        self.crypto_held = 0.0
        self.total_asset = self.balance

        self.done = False
        self.trades = []

        return self._next_observation()

    def _next_observation(self):
        obs = self.data['Close'].iloc[
            self.current_step - self.lookback_window_size : self.current_step
        ].values
        return obs

    def step(self, action):
        self._take_action(action)

        self.current_step += 1
        reward = 0  # Reward can be implemented later

        if self.current_step >= self.end_index:
            self.done = True
            reward = (self.total_asset - self.initial_balance) / self.initial_balance

        obs = self._next_observation()

        return obs, reward, self.done, {}

    def _take_action(self, action):
        #action = action[0]  # Extract the action value
        current_price = self.data.iloc[self.current_step]['Close']

        # Binance fee structure
        fee_rate = 0.001  # Default 0.1% trading fee
        if self.use_bnb_discount:
            fee_rate = 0.00075  # 0.075% fee with BNB discount

        if action > 0:
            # Buy proportion of available balance
            amount_to_spend = self.balance * action
            amount_to_buy = amount_to_spend / current_price
            # Apply fee
            fee = amount_to_spend * fee_rate
            self.balance -= (amount_to_spend + fee)  # Deduct both cost and fee
            self.crypto_held += amount_to_buy
            self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price, 'amount': amount_to_buy})
        elif action < 0:
            # Sell proportion of holdings
            amount_to_sell = self.crypto_held * (-action)
            amount_to_receive = amount_to_sell * current_price
            # Apply fee
            fee = amount_to_receive * fee_rate
            self.crypto_held -= amount_to_sell
            self.balance += (amount_to_receive - fee)  # Add proceeds minus fee
            self.trades.append({'step': self.current_step, 'type': 'sell', 'price': current_price, 'amount': amount_to_sell})

        # Update total assets
        self.total_asset = self.balance + self.crypto_held * current_price

    def render(self, mode='human'):
        current_date = self.data.index[self.current_step]
        profit = self.total_asset - self.initial_balance
        print(f'Date: {current_date}')
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Crypto held: {self.crypto_held:.6f}')
        print(f'Total asset: {self.total_asset:.2f}')
        print(f'Profit: {profit:.2f}\n')

def load_data(symbol='ETHUSDT', interval='6h'):
    """
    Download and preprocess historical ETH/USDT data from Binance.
    """
    client = Client()  # No need for API key for public data

    # Convert interval to Binance format
    binance_interval = interval.lower()

    # Download data starting from a specific date (e.g., January 1, 2020)
    klines = client.get_historical_klines(symbol, binance_interval, "1 Jan 2020")

    # Convert to DataFrame
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
               'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
               'Taker_buy_quote_asset_volume', 'Ignore']
    df = pd.DataFrame(klines, columns=columns)

    # Convert numeric columns to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)

    # Keep only the relevant columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return df
