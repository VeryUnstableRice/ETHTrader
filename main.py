import torch
from torch.distributions import Categorical
import eth_trader
from tradingenv import load_data, TradingEnv

if __name__ == "__main__":
    # Load data automatically
    data = load_data(symbol='ETHUSDT', interval='6h')

    # Initialize the trading environment
    env = TradingEnv(
        data,
        initial_balance=10,      # Easily changeable initial balance
        lookback_window_size=50,    # Number of past timesteps in the observation
        trading_interval='6h',      # Easily changeable trading interval
        use_bnb_discount=True       # Apply BNB fee discount (True/False)
    )

    # Reset the environment to start a new episode
    obs = env.reset()
    balance = env.balance
    networth = env.total_asset
    crypto_held = env.crypto_held
    done = False

    trader = eth_trader.TradingTransformer()

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        balance_tensor = torch.tensor(balance, dtype=torch.int).unsqueeze(0)
        networth_tensor = torch.tensor(balance, dtype=torch.int).unsqueeze(0)
        crypto_held_tensor = torch.tensor(crypto_held, dtype=torch.int).unsqueeze(0)

        action, value = trader(obs_tensor, balance_tensor, networth_tensor, crypto_held_tensor)

        action_distribution = Categorical(action)
        action_index = action_distribution.sample()  # Sample an action index
        action = (action_index.float() / 100) * 2 - 1  # Normalize to [-1, 1]

        obs, reward, done, info = env.step(action)

        balance = env.balance
        networth = env.total_asset

        env.render()