import re
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to the agent.log file
log_file_path = "D:/DRL based Resource Allocation in/New RA/colosseum-oran-commag-dataset-main/agent.log"

# Read the log file
with open(log_file_path, 'r') as f:
    log_data = f.read()

# Regular expression pattern to match the data lines
pattern = r'\[\[(.*?)\]\]'

# Find all matches of the pattern
matches = re.findall(pattern, log_data, re.DOTALL)

# Initialize lists to store extracted values
tx_brate_values = []
episode_rewards = []

# Process each match
for match in matches:
    # Split the match into individual values
    values = match.split()
    
    # Extract tx_brate and reward values
    try:
        tx_brate = float(values[1])
        reward = float(values[-1])
        tx_brate_values.append(tx_brate)
        episode_rewards.append(reward)
    except ValueError:
        pass

# Aggregate values per 1000 timeslots
timeslots_per_group = 1000
num_groups = len(tx_brate_values) // timeslots_per_group

# Create lists to store aggregated values
grouped_tx_brate = []
grouped_rewards = []

# Aggregate values
for i in range(num_groups):
    start_idx = i * timeslots_per_group
    end_idx = (i + 1) * timeslots_per_group
    avg_tx_brate = np.mean(tx_brate_values[start_idx:end_idx])
    avg_reward = np.mean(episode_rewards[start_idx:end_idx])
    grouped_tx_brate.append(avg_tx_brate)
    grouped_rewards.append(avg_reward)
# Create x-axis values (group index)
group_indices = list(range(1, num_groups + 1))

# Create a single figure with subplots
plt.figure(figsize=(12, 10))

# Subplot for tx_brate values
plt.subplot(2, 1, 1)
plt.plot(group_indices, grouped_tx_brate, marker='o', label='tx_brate downlink [Mbps]')
plt.xlabel('Group Index (Each Group = 1000 Timeslots)')
plt.ylabel('Average Data Sum Rate in [Mbps]')
plt.title('Average Data Sum Rate and Episode Reward per 1000 Timeslots Group')
plt.legend()
plt.grid(True)

# Subplot for episode rewards
plt.subplot(2, 1, 2)
plt.plot(group_indices, grouped_rewards, marker='o', color='orange', label='Episode Reward')
plt.xlabel('Group Index (Each Group = 1000 Timeslots)')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlapping labels and titles
plt.tight_layout()

# Show the combined plot
plt.show()
