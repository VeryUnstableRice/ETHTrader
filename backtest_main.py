import torch
from torch.distributions import Categorical
import eth_trader
from ppo_agent import PPOAgent
from tradingenv import load_data, TradingEnv
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(log_dir='./tensorboard_backtest')

if __name__ == "__main__":
    # Load data automatically
    data = load_data(symbol='ETHUSDT', interval='6h')
    # Initialize the trading environment
    env = TradingEnv(
        data,
        initial_balance=100,
        lookback_window_size=50,
        trading_interval='6h',
        use_bnb_discount=True
    )
    action_count = 100
    obs = env.reset(False)
    balance = env.balance
    networth = env.total_asset
    crypto_held = env.crypto_held
    done = False
    trader = eth_trader.TradingTransformer(n_actions=action_count).to(device)
    agent = PPOAgent(policy=trader, device=device, writer=writer, deployed=True)

    # Load the saved model
    checkpoint = torch.load('./model.ckp', map_location=device)
    trader.load_state_dict(checkpoint['model_statedict'])
    episode_num = checkpoint.get('episode_num', 0)

    print(f"Total number of parameters: {sum(p.numel() for p in trader.parameters())}")

    episode_reward = 0  # Initialize episode reward

    while not done:
        action, log_prob, value, action_index = agent.select_action(obs, balance, networth, crypto_held)
        obs_next, reward, done, info = env.step(action)
        balance_next = env.balance
        networth_next = env.total_asset
        crypto_held_next = env.crypto_held
        obs = obs_next
        episode_reward += reward
        balance = balance_next
        networth = networth_next
        crypto_held = crypto_held_next

        # Output what it does
        print(f"Step: {env.current_step}, Profit: {((networth-env.initial_balance)/env.initial_balance)*100:.2f}, Action taken: {action:.4f}, Net Worth: {networth:.2f}, Balance: {balance:.2f}, Crypto Held: {crypto_held:.6f}")

    # After the episode is done
    print("Episode finished.")
    agent.log_episode_performance(episode_num, episode_reward, networth, env.initial_balance)

    # Print trades
    print("Trades made during the episode:")
    for trade in env.trades:
        print(trade)

writer.close()
