import re
import matplotlib.pyplot as plt
import numpy as np

# Initialize dictionaries to store data for each slice
slice_data = {0: {'name': 'eMMB', 'rewards': []},
              1: {'name': 'MTC', 'rewards': []},
              2: {'name': 'URLLC', 'rewards': []}}

# Regular expression patterns for extracting data
slice_pattern = re.compile(r'Slice (\d+):')
reward_pattern = re.compile(r'Reward is: ([\d.]+)')

# File path
filename = (r"D:\DRL based Resource Allocation in\drl_ra_adv_attack\ResourceAllocatorSaboteur\agent.log")

# Read the file and iterate over each line
with open(filename, 'r') as file:
    current_slice = None
    for line in file:
        # Extract slice number
        slice_match = slice_pattern.search(line)
        if slice_match:
            current_slice = int(slice_match.group(1))
        
        # Extract reward
        reward_match = reward_pattern.search(line)
        if reward_match and current_slice is not None:
            reward = float(reward_match.group(1))
            slice_data[current_slice]['rewards'].append(reward)

# Calculate average rewards for each slice
for slice_number, data in slice_data.items():
    rewards = data['rewards']
    if rewards:
        data['average_reward'] = np.mean(rewards)
    else:
        data['average_reward'] = 0.0

# Extract data for plotting
slice_names = [data['name'] for _, data in slice_data.items()]
average_rewards = [data['average_reward'] for _, data in slice_data.items()]

# Plotting using Matplotlib as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(slice_names, average_rewards, color=['blue', 'green', 'orange'])
plt.xlabel('Slice')
plt.ylabel('Average Reward')
plt.title('Average Reward for Each Slice')
for i, reward in enumerate(average_rewards):
    plt.text(i, reward + 0.01, f'{reward:.2f}', ha='center', va='bottom')
plt.ylim(0, max(average_rewards) + 0.2)
plt.grid(True)
plt.show()
