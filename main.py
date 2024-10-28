import torch
from torch.distributions import Categorical
import eth_trader
from ppo_agent import PPOAgent
from tradingenv import load_data, TradingEnv
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(log_dir='./tensorboard_logs')

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
    obs = env.reset()
    balance = env.balance
    networth = env.total_asset
    crypto_held = env.crypto_held
    done = False
    trader = eth_trader.TradingTransformer(n_actions=action_count).to(device)
    agent = PPOAgent(policy=trader, device=device, writer=writer)

    print(f"Total number of parameters: {sum(p.numel() for p in trader.parameters())}")

    update_timestep = 2000
    timestep = 0
    episode_num = 0
    episode_reward = 0  # Initialize episode reward
    while True:
        timestep += 1
        action, log_prob, value, action_index = agent.select_action(obs, balance, networth, crypto_held)
        obs_next, reward, done, info = env.step(action)
        balance_next = env.balance
        networth_next = env.total_asset
        crypto_held_next = env.crypto_held
        agent.store_transition(obs, balance, networth, crypto_held, action_index, log_prob, reward, done, value)
        obs = obs_next
        episode_reward += reward
        balance = balance_next
        networth = networth_next
        crypto_held = crypto_held_next

        if timestep % update_timestep == 0:
            agent.update(obs, balance, networth, crypto_held)
            timestep = 0

        if done:
            agent.update(obs, balance, networth, crypto_held)
            obs = env.reset()

            if episode_num % 500 == 0:
                torch.save({'model_statedict' : trader.state_dict(), 'episode_num':episode_num}, './model.ckp')

            episode_num += 1
            agent.log_episode_performance(episode_num, episode_reward, networth)
            balance = env.balance
            networth = env.total_asset
            crypto_held = env.crypto_held
            done = False
            timestep = 0
            agent.reset_window_tokens()
            episode_reward = 0

writer.close()