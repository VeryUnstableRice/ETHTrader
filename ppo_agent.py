import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import math

class PPOAgent:
    def __init__(self, policy, device, writer, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=10, batch_size=64):
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
        self.window_tokens_buffer = []
        self.window_tokens = []  # Initialize the sliding window
        self.device = device
        self.writer = writer
        self.n_actions = self.policy.n_actions  # Get n_actions from the policy

    def reset_window_tokens(self):
        self.window_tokens = []

    def select_action(self, obs, balance, networth, crypto_held):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        balance_tensor = torch.tensor(balance, dtype=torch.float32, device=self.device).unsqueeze(0)
        networth_tensor = torch.tensor(networth, dtype=torch.float32, device=self.device).unsqueeze(0)
        crypto_held_tensor = torch.tensor(crypto_held, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Prepare window_tokens_tensor
        if len(self.window_tokens) > 0:
            window_tokens_tensor = torch.tensor(self.window_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            window_tokens_tensor = None  # Handle this in the policy

        num_iterations = 0
        while True:
            # Get logits and value from the policy
            logits, value = self.policy(obs_tensor, balance_tensor, crypto_held_tensor, networth_tensor, window_tokens_tensor)
            action_probs = F.softmax(logits, dim=-1)
            total_prob_under_n_actions = action_probs[:, :self.n_actions].sum().item()
            stop_probability = min(1.0, (2 ** num_iterations) / 100000)
            delta = math.log((stop_probability / (total_prob_under_n_actions + 1e-8)) + 1e-8)
            adjusted_logits = logits.clone()
            adjusted_logits[:, :self.n_actions] += delta
            action_probs_adjusted = F.softmax(adjusted_logits, dim=-1)
            action_distribution = Categorical(action_probs_adjusted)
            action_index = action_distribution.sample()
            action_index = action_index.item()
            # Log probability under the original logits
            action_log_probs_unadjusted = F.log_softmax(logits, dim=-1)
            log_prob = action_log_probs_unadjusted[0, action_index]

            if action_index < self.n_actions:
                action = (float(action_index) / (self.n_actions - 1)) * 2 - 1  # Normalize to [-1, 1]
                return action, log_prob.item(), value.item(), action_index
            else:
                action_token = action_index - self.n_actions
                # Add the new token to window_tokens
                self.window_tokens.append(action_token)
                if len(self.window_tokens) > 32:
                    self.window_tokens.pop(0)  # Remove the oldest token
                # Update window_tokens_tensor
                window_tokens_tensor = torch.tensor(self.window_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                num_iterations += 1
                #if num_iterations >= 40:
                    #print("Warning: Max iterations reached in select_action")
                #    action = ((float(action_index) % self.n_actions) / (self.n_actions - 1)) * 2 - 1  # Default action
                #    return action, log_prob.item(), value.item(), action_index

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
        self.window_tokens_buffer.append(self.window_tokens.copy())

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
        # Prepare window_tokens_tensor for the next state
        if len(self.window_tokens) > 0:
            next_window_tokens_tensor = torch.tensor(self.window_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            next_window_tokens_tensor = None  # Handle this in the policy
        _, next_value = self.policy(next_obs_tensor, next_balance_tensor, next_crypto_held_tensor, next_networth_tensor, next_window_tokens_tensor)
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
        # Process window_tokens_buffer
        max_window_length = 32
        window_tokens_padded = [
            wt + [0] * (max_window_length - len(wt)) if len(wt) < max_window_length else wt
            for wt in self.window_tokens_buffer
        ]
        window_tokens_tensor = torch.tensor(window_tokens_padded, dtype=torch.long, device=self.device)

        dataset = torch.utils.data.TensorDataset(
            obs_tensor, balance_tensor, networth_tensor, crypto_held_tensor,
            actions_tensor, log_probs_old, returns, advantages, window_tokens_tensor
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.update_epochs):
            for batch in dataloader:
                batch_obs, batch_balance, batch_networth, batch_crypto_held, \
                batch_actions, batch_log_probs_old, batch_returns, batch_advantages, \
                batch_window_tokens = batch
                # Pass batch_window_tokens to the policy
                logits, values = self.policy(batch_obs, batch_balance, batch_crypto_held, batch_networth, batch_window_tokens)
                action_distribution = Categorical(logits=logits)
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
        self.window_tokens_buffer = []

    def log_episode_performance(self, episode_num, total_reward, networth):
        print(f"Episode {episode_num}: Total Reward: {total_reward:.2f}, Net Worth: {networth:.2f}")
        # Log to TensorBoard
        self.writer.add_scalar("Total Reward", total_reward, episode_num)
        self.writer.add_scalar("Net Worth", networth, episode_num)
