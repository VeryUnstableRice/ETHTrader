import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

# Define the ActorCritic network
class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(ActorCritic, self).__init__()
        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 64),  # Input layer to hidden layer with 64 units
            nn.ReLU(),                # Activation function
        )
        # Policy head for action probabilities
        self.policy = nn.Linear(64, n_actions)  # Outputs logits for each action
        # Value head for state value estimation
        self.value = nn.Linear(64, 1)           # Outputs a single value

    def forward(self, x):
        x = self.fc(x)              # Pass input through shared layers
        logits = self.policy(x)     # Get action logits from policy head
        value = self.value(x)       # Get state value from value head
        return logits, value        # Return both logits and value

# Function to collect rollouts from the environment
def collect_rollout(env, model, n_steps):
    obs = env.reset()  # Reset the environment to initial state
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Convert observation to tensor
    # Initialize lists to store rollout data
    obs_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    log_probs_list = []
    values_list = []

    for _ in range(n_steps):
        logits, value = model(obs)             # Get action logits and state value from the model
        probs = F.softmax(logits, dim=-1)      # Convert logits to probabilities
        dist = torch.distributions.Categorical(probs)  # Create a categorical distribution
        action = dist.sample()                  # Sample an action from the distribution
        log_prob = dist.log_prob(action)        # Get log probability of the action

        # Detach tensors before storing to prevent backpropagation through them
        obs_list.append(obs)
        actions_list.append(action)
        values_list.append(value.detach())
        log_probs_list.append(log_prob.detach())

        action_np = action.cpu().numpy()[0]     # Convert action tensor to numpy
        obs_next, reward, done, _ = env.step(action_np)  # Take action in the environment
        rewards_list.append(torch.tensor([reward], dtype=torch.float32).to(device))  # Store reward
        dones_list.append(torch.tensor([1 - done], dtype=torch.float32).to(device))  # Store done flag

        if done:
            obs_next = env.reset()  # Reset environment if done

        # Prepare next observation
        obs = torch.FloatTensor(obs_next).unsqueeze(0).to(device)

    # Get the value for the last observation to compute returns
    with torch.no_grad():
        _, next_value = model(obs)
    values_list.append(next_value.detach())

    return obs_list, actions_list, rewards_list, dones_list, log_probs_list, values_list

# Main training loop
if __name__ == "__main__":
    env_name = "CartPole-v1"          # Define the environment name
    env = gym.make(env_name)           # Create the environment

    obs_size = env.observation_space.shape[0]  # Dimension of observation space
    n_actions = env.action_space.n             # Number of possible actions

    # Hyperparameters
    gamma = 0.99          # Discount factor for rewards
    clip_epsilon = 0.2    # Clipping parameter for PPO
    c1 = 0.5              # Coefficient for value loss
    c2 = 0.01             # Coefficient for entropy bonus
    lr = 1e-3             # Learning rate
    n_steps = 2048        # Number of steps per rollout
    epochs = 10           # Number of training epochs per update
    batch_size = 64      # Mini-batch size for training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    model = ActorCritic(obs_size, n_actions).to(device)  # Initialize the ActorCritic model
    optimizer = optim.Adam(model.parameters(), lr=lr)    # Initialize the optimizer

    num_updates = 1000  # Total number of training updates

    for update in range(num_updates):
        # Collect rollout data from the environment
        (obs_list,
         actions_list,
         rewards_list,
         dones_list,
         log_probs_list,
         values_list) = collect_rollout(env, model, n_steps)

        # Convert lists of tensors to batched tensors
        obs_batch = torch.cat(obs_list)                    # Shape: [n_steps, obs_size]
        actions_batch = torch.cat(actions_list)            # Shape: [n_steps]
        rewards_batch = torch.cat(rewards_list)            # Shape: [n_steps, 1]
        dones_batch = torch.cat(dones_list)                # Shape: [n_steps, 1]
        old_log_probs_batch = torch.cat(log_probs_list)    # Shape: [n_steps]
        values_batch = torch.cat(values_list[:-1]).squeeze(-1)  # Shape: [n_steps]
        next_value = values_list[-1]                        # Value for the last state

        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages = []
        returns = []
        gae = torch.zeros(1).to(device)  # Initialize GAE
        with torch.no_grad():
            for t in reversed(range(len(rewards_batch))):
                # Compute temporal difference error
                delta = rewards_batch[t] + gamma * values_list[t + 1] * dones_batch[t] - values_list[t]
                # Update GAE
                gae = delta + gamma * 0.95 * dones_batch[t] * gae  # lambda=0.95
                advantages.insert(0, gae)                  # Insert at the beginning
                returns.insert(0, gae + values_list[t])    # Compute return

        advantages = torch.cat(advantages).detach()        # Concatenate advantages
        returns = torch.cat(returns).detach().squeeze(-1) # Concatenate returns

        # Normalize advantages for better training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy and value network for multiple epochs
        for epoch in range(epochs):
            idx = np.arange(len(obs_batch))   # Create indices for shuffling
            np.random.shuffle(idx)             # Shuffle the indices for randomness

            # Iterate over mini-batches
            for start in range(0, len(obs_batch), batch_size):
                end = start + batch_size
                minibatch_idx = idx[start:end]  # Get indices for the current mini-batch

                # Select mini-batch data
                minibatch_obs = obs_batch[minibatch_idx]
                minibatch_actions = actions_batch[minibatch_idx]
                minibatch_old_log_probs = old_log_probs_batch[minibatch_idx]
                minibatch_advantages = advantages[minibatch_idx]
                minibatch_returns = returns[minibatch_idx]
                minibatch_values = values_batch[minibatch_idx]

                # Forward pass: get logits and value estimates
                logits, value = model(minibatch_obs)
                probs = F.softmax(logits, dim=-1)           # Compute action probabilities
                dist = torch.distributions.Categorical(probs) # Create distribution
                new_log_probs = dist.log_prob(minibatch_actions)  # Log probs of actions

                # Calculate probability ratio for PPO
                ratio = (new_log_probs - minibatch_old_log_probs).exp()
                # Calculate surrogate losses
                surr1 = ratio * minibatch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * minibatch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()  # PPO clipped loss

                # Calculate value function loss
                value_loss = F.mse_loss(value.squeeze(-1), minibatch_returns)

                # Calculate entropy bonus for exploration
                entropy = dist.entropy().mean()

                # Total loss combines policy loss, value loss, and entropy bonus
                loss = policy_loss + c1 * value_loss - c2 * entropy

                # Backpropagation
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()        # Compute gradients
                optimizer.step()       # Update model parameters

        # Optionally, print training statistics every 10 updates
        if update % 10 == 0:
            test_rewards = []  # List to store rewards from test episodes
            for _ in range(5):  # Evaluate over 5 episodes
                obs = env.reset()  # Reset environment
                done = False
                total_reward = 0
                while not done:
                    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Convert observation to tensor
                    with torch.no_grad():
                        logits, _ = model(obs)             # Get action logits from the model
                        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
                    action = probs.argmax(dim=-1).item()      # Select the action with highest probability
                    obs, reward, done, _ = env.step(action)    # Take action in the environment
                    total_reward += reward                     # Accumulate reward
                test_rewards.append(total_reward)              # Store total reward for the episode
            # Print the current update, loss, and average test reward
            print(f"Update {update}, Loss {loss.item()}, Test reward: {np.mean(test_rewards)}")
