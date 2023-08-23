import re
import matplotlib.pyplot as plt

rewards = []

with open('agent.log', 'r') as f:
  for line in f:
    match = re.search(r'Reward is: ([0-9\.]+)', line)
    if match:
      reward = float(match.group(1)) 
      rewards.append(reward)

print("Reward values:")
print(rewards) 

plt.hist(rewards, bins=20)
plt.title("Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.show()
