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
    done = False

    while not done:
        # Random agent: choose actions randomly between -1 and 1
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()