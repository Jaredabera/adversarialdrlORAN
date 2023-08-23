import re
import matplotlib.pyplot as plt

slice_rewards = {
    "Slice 0": [], 
    "Slice 1": [],
    "Slice 2": []
}

with open('agent.log', 'r') as f:
  for line in f:
    if "Slice 0" in line:
      match = re.search(r'Reward is: ([0-9\.]+)', line)
      if match:
        reward = float(match.group(1))
        slice_rewards["Slice 0"].append(reward)

    if "Slice 1" in line:
      match = re.search(r'Reward is: ([0-9\.]+)', line)
      if match:
        reward = float(match.group(1))  
        slice_rewards["Slice 1"].append(reward)

    if "Slice 2" in line:
      match = re.search(r'Reward is: ([0-9\.]+)', line)
      if match:
        reward = float(match.group(1))
        slice_rewards["Slice 2"].append(reward)
        
plt.figure()

for slice_name, rewards in slice_rewards.items():
  plt.subplot(1, 3, 1)
  plt.hist(rewards, bins=20, label=slice_name)

plt.legend()
plt.show()
