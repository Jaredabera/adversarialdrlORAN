import h5py
import tensorflow as tf
import matplotlib.pyplot as plt


filename = (r"D:\DRL based Resource Allocation in\New RA\colosseum-oran-commag-dataset-main\agent.log")

with open(filename,'r') as f:
    data = f.readlines()
slice_0 = []
slice_1 = []
slice_2 = []

for i in data:
    if 'Slice 0:' in i:
        index = i.index('Reward is: ')
        slice_0.append(float(i[index:].split(' ')[-1].strip()))
        # print(float(i[index:].split(' ')[-1].strip()))
    if 'Slice 1:' in i:
        index = i.index('Reward is: ')
        slice_1.append(float(i[index:].split(' ')[-1].strip()))
    if 'Slice 2:' in i:
        index = i.index('Reward is: ')
        slice_2.append(float(i[index:].split(' ')[-1].strip()))
        


# Create a bar plot for the rewards
plt.figure(figsize=(10, 6))
# plt.subplot(1, 3, 1)
plt.plot([i for i in range(len(slice_0))], slice_0, color='blue',label='eMMB')

#plt.title('eMMB')
plt.ylabel('Cumulative Reward')
plt.xlabel('Episode')

# plt.subplot(1, 3, 2)
plt.plot([i for i in range(len(slice_1))], slice_1, color='red',label='MTC')
# plt.xlabel('time ')

# plt.title('MTC')
# plt.ylabel('Cumulative Reward')
# plt.xlabel('time')

#plt.subplot(1, 3, 3)
plt.plot([i for i in range(len(slice_2))], slice_2, color='green',label='URLLC')
#plt.xlabel('time')

#plt.title('URLLC')
# plt.ylabel('Cumulative Reward')
# plt.xlabel('time')

# plt.subplot(1, 2, 2)
#plt.bar(slice_names, [adversarial_reward, adversarial_reward, adversarial_reward], color='red')
# plt.title('Adversarial Rewards')

plt.tight_layout()

plt.legend()

plt.show()