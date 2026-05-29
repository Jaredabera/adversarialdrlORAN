from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import policy_saver

import absl
import time
import os
import glob

import pandas as pd
import numpy as np

import logging


# Generating dataset from csv file. Returns a Pandas DataFrame
def entire_dataset_from_single_file(filename,
                                    col_names,
                                    selected_col_names,
                                    remove_zero_req_prb_entries=True,
                                    scale_dl_buffer=True,
                                    replace_zero_with_one=False,
                                    add_prb_ratio=True):
    dataset = pd.read_csv(filename, names=col_names, usecols=selected_col_names, header=0)

    if remove_zero_req_prb_entries:
        dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)

    if scale_dl_buffer and any(["dl_buffer [bytes]" in m for m in
                                selected_col_names]):
        # scale the dl_buffer
        dataset['dl_buffer [bytes]'] = dataset['dl_buffer [bytes]'] / 100000

    if add_prb_ratio:
        dict_add = pd.DataFrame.from_dict({"ratio_granted_req": np.clip(np.nan_to_num(
            dataset["sum_granted_prbs"] / dataset["sum_requested_prbs"]), a_min=0, a_max=1)
        })
        if replace_zero_with_one:
            dict_add['ratio_granted_req'].loc[dataset['sum_requested_prbs'] <= 0] = 1.0
        return dataset.join(dict_add)
    else:
        return dataset

# return all csv files inside a single DataFrame
def entire_dataset_from_folder(main_folder,
                               wildcard,
                               col_names,
                               selected_col_names,
                               scale_dl_buffer=True,
                               remove_zero_req_prb_entries=True,
                               replace_zero_with_one=False,
                               add_prb_ratio=True):
    dataset = []
    for filename in glob.glob(main_folder + wildcard):
        db_tmp = entire_dataset_from_single_file(filename, col_names=col_names,
                                                 selected_col_names=selected_col_names,
                                                 scale_dl_buffer=scale_dl_buffer,
                                                 remove_zero_req_prb_entries=remove_zero_req_prb_entries,
                                                 replace_zero_with_one=replace_zero_with_one,
                                                 add_prb_ratio=add_prb_ratio)
        dataset.append(db_tmp)

    return pd.concat(dataset, axis=0, ignore_index=True)


# take n entries from the DataFrame at random
def extract_n_entries_from_dataset(dataset=None,
                                   slice_id=None,
                                   n_entries=10,
                                   metrics_export=None):
    if slice_id is not None:
        d_temp = dataset.loc[dataset['slice_id'] == int(slice_id)]
    else:
        d_temp = dataset

    d_temp = d_temp.sample(n=n_entries).reset_index(drop=True)
    if metrics_export is not None:
        d_temp = d_temp[metrics_export]

    return d_temp

# This function is used here to emulate a DU reporting real-time data. Replace this function with your DU
# FOR TESTING PURPOSES ONLY
def get_data_from_DUs(dataset=None,
                      n_entries=1000,
                      n_col=4,
                      slice_id=None,
                      metrics_export=None):

    if dataset is None:  # generate random data in case you do not have a dataset
        values = np.random.random(size=(n_entries, n_col))
        slice_id = np.random.randint(low=0, high=3, size=(n_entries, 1))
        data = np.concatenate((slice_id, values), axis=1)
    else:
        data = extract_n_entries_from_dataset(dataset=dataset,
                                              slice_id=slice_id,
                                              n_entries=n_entries,
                                              metrics_export=metrics_export)

    return data

# Return lists for metrics, rewards, prbs assigned to each slice.
# Ideally, the list is such that len(list) = num_slices
def split_data(slice_profiles=None,
               data_to_spit=None,
               metric_list=None,
               metric_dict=None,
               n_entries_per_slice=None):
    metrics = []
    prbs = []
    rewards = []

    # ordering here follows slice_profiles
    for i in slice_profiles:

        slice_data = data_to_spit[data_to_spit[:, metric_dict['slice_id']] == slice_profiles[i]['slice_id'], :]

        if slice_data.size > 0:
            # repmat on rows to reach needed dimension in case you do not have enough reporting data
            while slice_data.shape[0] < n_entries_per_slice:
                slice_data = np.vstack((slice_data, np.zeros((1, slice_data.shape[1]))))

            slice_prb = slice_data[:, metric_dict['slice_prb']]
            slice_metrics = slice_data[:, [metric_dict[x] for x in metric_list]]
            slice_reward = slice_data[:, metric_dict[slice_profiles[i]['reward_metric']]]

            if n_entries_per_slice is not None:
                slice_prb = slice_prb[0:n_entries_per_slice]
                slice_metrics = slice_metrics[0:n_entries_per_slice, :]
                slice_reward = slice_reward[0:n_entries_per_slice]
        else:
            slice_metrics = []
            slice_prb = []
            slice_reward = []

        metrics.append(slice_metrics)
        prbs.append(slice_prb)
        rewards.append(slice_reward)

    return metrics, prbs, rewards

# Used to generate the input to the DRL agent. It returns a TimeStep that contains (step_type, reward, discount, observations)
def generate_timestep_for_policy(obs_tmp=None):
    step_type = tf.convert_to_tensor(
        [0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor(
        [0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor(
        [1], dtype=tf.float32, name='discount')
    observations = tf.convert_to_tensor(
        [obs_tmp], dtype=tf.float32, name='observations')
    return ts.TimeStep(step_type, reward, discount, observations)


def apply_ppo_policy_infiltration_attack(obs_tmp, agent, epsilon=0.05, alpha=0.01, num_iterations=50):
    """
    PPO-specific Policy Infiltration Attack (PIA)
    
    Targets PPO's key vulnerabilities:
    1. Exploits clipped probability ratio mechanism (PPO's main defense)
    2. Manipulates actor-critic advantage computation
    3. Crafts perturbations to maximize divergence from trained policy
    4. Violates trust region constraints through gradient-based optimization
    
    Args:
        obs_tmp: Original observation (numpy array)
        agent: PPO policy agent
        epsilon: Maximum perturbation magnitude
        alpha: Step size for gradient ascent
        num_iterations: Number of attack iterations
    
    Returns:
        adversarial_action: Action from perturbed observation
        attack_success: Boolean indicating if attack exceeded trust region
        divergence: KL divergence between original and attacked policy distributions
    """
    
    # Initialize perturbation
    perturbation = tf.Variable(tf.zeros_like(obs_tmp, dtype=tf.float32), trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    
    # Store metrics
    attack_metrics = {
        'max_divergence': 0.0,
        'max_prob_ratio': 0.0,
        'iterations_to_break_trust_region': -1,
        'clip_threshold': 0.2  # PPO clipping parameter (typically 0.1-0.2)
    }
    
    # Get original policy distribution
    original_timestep = generate_timestep_for_policy(obs_tmp)
    
    try:
        original_dist = agent.actor_net(original_timestep.observation)
        if hasattr(original_dist, 'mean'):
            original_action_mean = original_dist.mean()
            original_action_std = original_dist.stddev()
        else:
            original_action_mean = original_dist
            original_action_std = tf.ones_like(original_dist) * 0.1
    except:
        logging.warning("Could not extract original policy distribution")
        original_action_mean = None
        original_action_std = None
    
    # Iterative gradient-based attack
    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            
            # Create adversarial observation
            adversarial_obs = obs_tmp + perturbation
            
            # Clip to valid observation space
            adversarial_obs = tf.clip_by_value(adversarial_obs, -10.0, 10.0)
            
            # Create timestep for perturbed observation
            adversarial_timestep = generate_timestep_for_policy(adversarial_obs)
            
            # Get attacked policy distribution
            try:
                attacked_dist = agent.actor_net(adversarial_timestep.observation)
                if hasattr(attacked_dist, 'mean'):
                    attacked_action_mean = attacked_dist.mean()
                    attacked_action_std = attacked_dist.stddev()
                else:
                    attacked_action_mean = attacked_dist
                    attacked_action_std = tf.ones_like(attacked_dist) * 0.1
            except:
                logging.warning(f"Could not extract attacked policy distribution at iteration {iteration}")
                continue
            
            # ATTACK OBJECTIVE 1: Maximize divergence from original policy
            # KL divergence between original and attacked distributions
            if original_action_mean is not None:
                # KL(original || attacked) = 0.5 * sum(((mu1-mu2)^2 + sig1^2 + sig2^2 - 2*sig1*sig2) / sig2^2)
                mean_diff = original_action_mean - attacked_action_mean
                var_ratio = (original_action_std ** 2 + attacked_action_std ** 2) / (2 * (attacked_action_std ** 2) + 1e-8)
                kl_divergence = 0.5 * tf.reduce_mean(
                    (mean_diff ** 2) / (attacked_action_std ** 2 + 1e-8) + var_ratio - 1.0
                )
            else:
                kl_divergence = 0.0
            
            # ATTACK OBJECTIVE 2: Exploit probability ratio clipping
            # Try to maximize the ratio that would exceed PPO's clipping threshold
            action_distance = tf.reduce_mean(tf.abs(attacked_action_mean - original_action_mean))
            
            # ATTACK OBJECTIVE 3: Maximize advantage function exploitation
            # Craft perturbations that lead to actions with higher predicted value
            try:
                attacked_value = agent.value_net(adversarial_timestep.observation)
            except:
                attacked_value = tf.constant(0.0)
            
            # Composite attack loss: maximize divergence and advantage mismatch
            attack_loss = -(kl_divergence + action_distance + tf.reduce_mean(attacked_value) * 0.1)
            
        # Compute gradients w.r.t. perturbation
        gradients = tape.gradient(attack_loss, perturbation)
        
        if gradients is not None:
            # Apply gradient ascent to maximize attack loss
            optimizer.apply_gradients([(gradients, perturbation)])
            
            # Project perturbation to epsilon-ball
            perturbation_norm = tf.norm(perturbation)
            if perturbation_norm > epsilon:
                perturbation.assign(perturbation * (epsilon / (perturbation_norm + 1e-8)))
            
            # Track metrics
            if kl_divergence > attack_metrics['max_divergence']:
                attack_metrics['max_divergence'] = float(kl_divergence.numpy())
            
            prob_ratio = tf.exp(-kl_divergence)  # Approximation of probability ratio
            if float(prob_ratio.numpy()) > attack_metrics['max_prob_ratio']:
                attack_metrics['max_prob_ratio'] = float(prob_ratio.numpy())
            
            # Check if trust region is violated (ratio > 1 + clip_threshold)
            if float(prob_ratio.numpy()) > (1.0 + attack_metrics['clip_threshold']) and attack_metrics['iterations_to_break_trust_region'] == -1:
                attack_metrics['iterations_to_break_trust_region'] = iteration
        
        if (iteration + 1) % 10 == 0:
            logging.debug(f"PIA Iteration {iteration + 1}/{num_iterations} - KL Div: {kl_divergence:.6f}, Prob Ratio: {float(prob_ratio.numpy()):.6f}")
    
    # Generate final adversarial action
    final_adversarial_obs = obs_tmp + perturbation
    final_adversarial_obs = tf.clip_by_value(final_adversarial_obs, -10.0, 10.0)
    final_timestep = generate_timestep_for_policy(final_adversarial_obs)
    
    try:
        adversarial_action = agent.action(final_timestep)
    except:
        # Fallback to policy network
        adversarial_action = agent.actor_net(final_timestep.observation)
    
    # Determine attack success
    attack_success = attack_metrics['iterations_to_break_trust_region'] != -1
    
    logging.info(f"PIA Attack Metrics:")
    logging.info(f"  Max KL Divergence: {attack_metrics['max_divergence']:.6f}")
    logging.info(f"  Max Prob Ratio: {attack_metrics['max_prob_ratio']:.6f}")
    logging.info(f"  Trust Region Broken: {attack_success} (at iteration {attack_metrics['iterations_to_break_trust_region']})")
    logging.info(f"  Perturbation Norm: {float(tf.norm(perturbation).numpy()):.6f}")
    
    return adversarial_action, attack_success, attack_metrics


if __name__ == '__main__':

    # Column names in the srsLTE CSV dataset
    all_metrics_list = ["Timestamp",
                        "num_ues",
                        "IMSI",
                        "RNTI",
                        "empty_1",
                        "slicing_enabled",
                        "slice_id",
                        "slice_prb",
                        "power_multiplier",
                        "scheduling_policy",
                        "empty_2",
                        "dl_mcs",
                        "dl_n_samples",
                        "dl_buffer [bytes]",
                        "tx_brate downlink [Mbps]",
                        "tx_pkts downlink",
                        "tx_errors downlink (%)",
                        "dl_cqi",
                        "empty_3",
                        "ul_mcs",
                        "ul_n_samples",
                        "ul_buffer [bytes]",
                        "rx_brate uplink [Mbps]",
                        "rx_pkts uplink",
                        "rx_errors uplink (%)",
                        "ul_rssi",
                        "ul_sinr",
                        "phr",
                        "empty_4",
                        "sum_requested_prbs",
                        "sum_granted_prbs",
                        "empty_5",
                        "dl_pmi",
                        "dl_ri",
                        "ul_n",
                        "ul_turbo_iters"]

    # Column names we need to extract from the dataset
    metric_list_to_extract = ["slice_id",
                              "dl_buffer [bytes]",
                              "tx_brate downlink [Mbps]",
                              "sum_requested_prbs",
                              "sum_granted_prbs"]

    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='./agent.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    use_gpu_in_env = True
    mtc_policy_filename = './ml_models/mtc_policy'
    urllc_policy_filename = './ml_models/urllc_policy'
    embb_policy_filename = './ml_models/embb_policy'
    autoencoder_filename = './ml_models/encoder.h5'

    # Location of the dataset we want to use (valid in offline testing ONLY)
    main_folder = './slice_traffic/rome_static_close/tr10'
    wildcard_match = '/*/*/slices_bs*/*_metrics.csv'

    # get dataset for testing purposes only.
    # This is used as this code does not run with hardware components.
    # Not needed if getting data from real DUs
    dataset = entire_dataset_from_folder(main_folder=main_folder,
                                         wildcard=wildcard_match,
                                         col_names=all_metrics_list,
                                         selected_col_names=metric_list_to_extract)

    # Input size to the autoencoder for dimentionality reduction
    n_entries_for_autoencoder = 10

    # set logging level + enable TF2 behavior
    absl.logging.set_verbosity(absl.logging.INFO)
    # select which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if use_gpu_in_env is False:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        print("Num GPUs Available outside environments: ", len(gpu_devices))

    # load policy, these are the folders where saved_model.pb is stored
    drl_agents = [tf.saved_model.load(embb_policy_filename),
                  tf.saved_model.load(mtc_policy_filename),
                  tf.saved_model.load(urllc_policy_filename)]

    absl.logging.info('Agents loaded')

    autoencoder = tf.keras.models.load_model(autoencoder_filename)
    
    absl.logging.info('Autoencoder loaded')

    slice_profiles = {'embb': {'slice_id': 0, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'mtc': {'slice_id': 1, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'urllc': {'slice_id': 2, 'reward_metric': "ratio_granted_req"}}

    metric_dict = {"dl_buffer [bytes]": 1,
                   "tx_brate downlink [Mbps]": 2,
                   "ratio_granted_req": 3,
                   "slice_id": 0,
                   "slice_prb": 4}

    metric_list_for_agents = ["dl_buffer [bytes]",
                   "tx_brate downlink [Mbps]",
                   "ratio_granted_req"]

    default_policy = 0
    previous_policy = dict()
    for _, val in slice_profiles.items():
        previous_policy[val['slice_id']] = default_policy

    previous_metrics = ''
    
    # Track PIA attack success metrics
    pia_metrics = {
        'total_attacks': 0,
        'successful_attacks': 0,
        'avg_kl_divergence': 0.0,
        'avg_prob_ratio': 0.0,
        'success_rate': 0.0
    }

    while True:
        policies = list()
        clean_actions = list()
        attacked_actions = list()

        # This is where data comes from the DUs.
        # As an example, we extract data from the static dataset.
        # Users may want to interface it with their own DUs
        data = get_data_from_DUs(dataset=dataset,
                                 n_entries=1000,
                                 metrics_export=metric_list_to_extract).to_numpy()

        data_tmp, prbs, rewards = split_data(slice_profiles=slice_profiles,
                                             data_to_spit=data,
                                             metric_dict=metric_dict,
                                             metric_list=metric_list_for_agents,
                                             n_entries_per_slice=n_entries_for_autoencoder)
        
        # PIA Attack Parameters
        epsilon_pia = 0.05  # Maximum perturbation magnitude (PPO trust region)
        alpha_pia = 0.01    # Gradient step size
        num_iterations_pia = 50  # Attack iterations

        for i, profile in enumerate(slice_profiles.keys()):
            if len(data_tmp[i]) > 0:
                for row in data_tmp[i]:
                    row[0] /= 100000

                logging.info('='*80)
                logging.info(f'Testing PIA on Slice {i} ({profile})')
                logging.info('='*80)
                
                logging.info('Data received from DU (dl_buffer [bytes], tx_brate downlink [Mbps], ratio_granted_req): ')
                logging.info(np.expand_dims(data_tmp[i], axis=0))

                obs_tmp = autoencoder.predict(np.expand_dims(data_tmp[i], axis=0)).astype('float32')
                obs_tmp = np.append(obs_tmp, prbs[i][0]).astype('float32')

                reward_mean = np.mean(rewards[i]).astype('float32')
                
                # Get clean action
                time_step = generate_timestep_for_policy(obs_tmp)
                clean_action = drl_agents[i].action(time_step)
                clean_actions.append(clean_action[0][0][0].numpy())
                
                # Apply PPO-specific Policy Infiltration Attack
                try:
                    adversarial_action, attack_success, attack_metrics = apply_ppo_policy_infiltration_attack(
                        obs_tmp, 
                        drl_agents[i], 
                        epsilon=epsilon_pia,
                        alpha=alpha_pia,
                        num_iterations=num_iterations_pia
                    )
                    
                    attacked_actions.append(adversarial_action[0][0][0].numpy())
                    
                    # Update global PIA metrics
                    pia_metrics['total_attacks'] += 1
                    if attack_success:
                        pia_metrics['successful_attacks'] += 1
                    pia_metrics['avg_kl_divergence'] += attack_metrics['max_divergence']
                    pia_metrics['avg_prob_ratio'] += attack_metrics['max_prob_ratio']
                    pia_metrics['success_rate'] = pia_metrics['successful_attacks'] / max(pia_metrics['total_attacks'], 1)
                    
                    logging.info(f"PIA Attack Success: {attack_success}")
                    logging.info(f"Clean Action: {clean_action[0][0][0].numpy()}")
                    logging.info(f"Attacked Action: {adversarial_action[0][0][0].numpy()}")
                    logging.info(f"Action Difference: {abs(float(adversarial_action[0][0][0].numpy()) - float(clean_action[0][0][0].numpy())):.6f}")
                    
                except Exception as e:
                    logging.error(f"PIA Attack failed on slice {i}: {str(e)}")
                    attacked_actions.append(clean_action[0][0][0].numpy())
                
                # Use attacked action if successful, else use clean action
                if len(attacked_actions) > len(clean_actions) - 1:
                    policies.append(attacked_actions[-1])
                else:
                    policies.append(clean_actions[-1])
                
                previous_policy[i] = policies[-1]

                logging.info(f'Slice {i}: Clean Action = {clean_action[0][0][0].numpy():.6f}, Reward = {reward_mean:.6f}')
            else:
                # append previous policy
                policies.append(previous_policy[i])
                logging.info(f'Using previous action {previous_policy[i]:.6f} for slice profile {i}')

        # build message to send policies to the DU
        msg = ','.join([f"{x:.6f}" for x in policies])
        logging.info('='*80)
        logging.info('Sending this message to the DU: ' + msg)
        logging.info(f"Global PIA Success Rate: {pia_metrics['success_rate']*100:.2f}% ({pia_metrics['successful_attacks']}/{pia_metrics['total_attacks']})")
        logging.info('='*80)

        time.sleep(10)
