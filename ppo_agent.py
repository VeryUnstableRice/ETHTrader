import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, policy, device, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=10, batch_size=64):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.obs_buffer = []
        self.balance_buffer = []
        self.networth_buffer = []
        self.crypto_held_buffer = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.device = device

    def select_action(self, obs, balance, networth, crypto_held):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        balance_tensor = torch.tensor(balance, dtype=torch.float32, device=self.device).unsqueeze(0)
        networth_tensor = torch.tensor(networth, dtype=torch.float32, device=self.device).unsqueeze(0)
        crypto_held_tensor = torch.tensor(crypto_held, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_probs, value = self.policy(obs_tensor, balance_tensor, crypto_held_tensor, networth_tensor)
        action_distribution = Categorical(action_probs)
        action_index = action_distribution.sample()
        action = (action_index.float() / 100) * 2 - 1  # Normalize to [-1, 1]
        log_prob = action_distribution.log_prob(action_index)

        # Return 'action' as a tensor instead of a float
        return action, log_prob.item(), value.item(), action_index.item()

    def store_transition(self, obs, balance, networth, crypto_held, action_index, log_prob, reward, done, value):
        self.obs_buffer.append(obs)
        self.balance_buffer.append(balance)
        self.networth_buffer.append(networth)
        self.crypto_held_buffer.append(crypto_held)
        self.actions.append(action_index)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, next_value):
        returns = []
        advantages = []
        gae = 0
        prev_value = next_value
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * prev_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * 0.95 * gae * (1 - self.dones[i])
            advantages.insert(0, gae)
            prev_value = self.values[i]
            returns.insert(0, gae + self.values[i])
        return returns, advantages

    def update(self, next_obs, next_balance, next_networth, next_crypto_held):
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_balance_tensor = torch.tensor(next_balance, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_networth_tensor = torch.tensor(next_networth, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_crypto_held_tensor = torch.tensor(next_crypto_held, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, next_value = self.policy(next_obs_tensor, next_balance_tensor, next_crypto_held_tensor, next_networth_tensor)
        next_value = next_value.item()
        returns, advantages = self.compute_returns_and_advantages(next_value)
        obs_tensor = torch.tensor(self.obs_buffer, dtype=torch.float32, device=self.device)
        balance_tensor = torch.tensor(self.balance_buffer, dtype=torch.float32, device=self.device).unsqueeze(-1)
        networth_tensor = torch.tensor(self.networth_buffer, dtype=torch.float32, device=self.device).unsqueeze(-1)
        crypto_held_tensor = torch.tensor(self.crypto_held_buffer, dtype=torch.float32, device=self.device).unsqueeze(-1)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        log_probs_old = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(
            obs_tensor, balance_tensor, networth_tensor, crypto_held_tensor,
            actions_tensor, log_probs_old, returns, advantages
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.update_epochs):
            for batch in dataloader:
                batch_obs, batch_balance, batch_networth, batch_crypto_held, batch_actions, batch_log_probs_old, batch_returns, batch_advantages = batch
                action_probs, values = self.policy(batch_obs, batch_balance, batch_crypto_held, batch_networth)
                action_distribution = Categorical(action_probs)
                batch_log_probs = action_distribution.log_prob(batch_actions)
                entropy = action_distribution.entropy().mean()
                ratios = torch.exp(batch_log_probs - batch_log_probs_old)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                loss = -torch.min(surr1, surr2).mean() + 0.5 * value_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.obs_buffer = []
        self.balance_buffer = []
        self.networth_buffer = []
        self.crypto_held_buffer = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def log_episode_performance(self, episode_num, total_reward, networth):
        print(f"Episode {episode_num}: Total Reward: {total_reward:.2f}, Net Worth: {networth:.2f}")