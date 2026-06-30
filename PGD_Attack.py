#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.trajectories import time_step as ts

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
    # read only the selected_col_names from CSV (these must exist in the CSV)
    dataset = pd.read_csv(filename, names=col_names, usecols=selected_col_names, header=0)

    if remove_zero_req_prb_entries:
        dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)

    if scale_dl_buffer and any(["dl_buffer [bytes]" in m for m in selected_col_names]):
        # scale the dl_buffer
        dataset['dl_buffer [bytes]'] = dataset['dl_buffer [bytes]'] / 100000

    if add_prb_ratio:
        dict_add = pd.DataFrame.from_dict({
            "ratio_granted_req": np.clip(np.nan_to_num(
                dataset["sum_granted_prbs"] / dataset["sum_requested_prbs"]), a_min=0, a_max=1)
        })
        if replace_zero_with_one:
            dict_add.loc[dataset['sum_requested_prbs'] <= 0, 'ratio_granted_req'] = 1.0
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
        # select rows where slice_id column (index given by metric_dict) equals this slice_id
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
    step_type = tf.convert_to_tensor([0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor([0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor([1], dtype=tf.float32, name='discount')
    observations = tf.convert_to_tensor([obs_tmp], dtype=tf.float32, name='observations')
    return ts.TimeStep(step_type, reward, discount, observations)


# Black-box PGD-like attack (randomized search inside L_inf ball)
def apply_pgd_attack(data, eval_fn,
                     epsilon=0.01, num_iterations=50, alpha=0.005, restarts=3):
    """
    Black-box, gradient-free PGD-like search that tries to maximize eval_fn(perturbed_data).
    - data: numpy array shape (n_steps, n_features)
    - eval_fn: function(perturbed_data) -> float score (higher is 'better' for attacker)
    - epsilon: L_inf bound around data (per-feature)
    - num_iterations: inner loop iterations per restart
    - alpha: step size per iteration (L_inf)
    - restarts: number of random restarts
    Returns best_perturbed_data found (same shape as data).
    """
    best_data = data.copy()
    best_score = -np.inf

    # small sanity clip
    data = data.astype(np.float32)
    for r in range(restarts):
        # random start inside epsilon ball
        x = data + np.random.uniform(-epsilon, epsilon, size=data.shape).astype(np.float32)
        x = np.clip(x, data - epsilon, data + epsilon)

        for it in range(num_iterations):
            # propose small random step (directional)
            step = alpha * np.sign(np.random.normal(size=data.shape)).astype(np.float32)
            x = x + step
            x = np.clip(x, data - epsilon, data + epsilon)

            try:
                score = float(eval_fn(x))
            except Exception:
                score = -np.inf

            if score > best_score:
                best_score = score
                best_data = x.copy()

    return best_data


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

    # Columns we will ask pandas to read from CSV; these must exist in the CSV file.
    # We include slice_prb here because code later expects a slice_prb column.
    selected_cols_for_read = ["slice_id",
                              "dl_buffer [bytes]",
                              "tx_brate downlink [Mbps]",
                              "slice_prb",
                              "sum_requested_prbs",
                              "sum_granted_prbs"]

    # Columns we will export at runtime to the agent: include ratio_granted_req (added by entire_dataset_from_single_file)
    metrics_export = ["slice_id",
                      "dl_buffer [bytes]",
                      "tx_brate downlink [Mbps]",
                      "slice_prb",
                      "ratio_granted_req"]

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
                                         selected_col_names=selected_cols_for_read)

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

    # load policy, these are the folder where saved_model.pb is stored
    drl_agents = [tf.saved_model.load(embb_policy_filename),
                  tf.saved_model.load(mtc_policy_filename),
                  tf.saved_model.load(urllc_policy_filename)]

    absl.logging.info('Agents loaded')

    autoencoder = tf.keras.models.load_model(autoencoder_filename)

    absl.logging.info('Autoencoder loaded')

    slice_profiles = {'embb': {'slice_id': 0, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'mtc': {'slice_id': 1, 'reward_metric': "tx_brate downlink [Mbps]"},
                      'urllc': {'slice_id': 2, 'reward_metric': "ratio_granted_req"}}

    # IMPORTANT: metric_dict references indices in the runtime data array produced by metrics_export
    # metrics_export order: ["slice_id","dl_buffer...","tx_brate...","slice_prb","ratio_granted_req"]
    metric_dict = {"slice_id": 0,
                   "dl_buffer [bytes]": 1,
                   "tx_brate downlink [Mbps]": 2,
                   "slice_prb": 3,
                   "ratio_granted_req": 4}

    metric_list_for_agents = ["dl_buffer [bytes]",
                              "tx_brate downlink [Mbps]",
                              "ratio_granted_req"]

    default_policy = 0
    previous_policy = dict()
    for _, val in slice_profiles.items():
        previous_policy[val['slice_id']] = default_policy

    previous_metrics = ''
    # Initialize rewards_dict outside the loop
    rewards_dict = {profile: [] for profile in slice_profiles.keys()}

    while True:
        policies = list()

        # This is where data comes from the DUs.
        data = get_data_from_DUs(dataset=dataset,
                                 n_entries=1000,
                                 metrics_export=metrics_export)
        # ensure numpy array
        if hasattr(data, "to_numpy"):
            data = data.to_numpy()
        else:
            data = np.asarray(data)

        data_tmp, prbs, rewards = split_data(slice_profiles=slice_profiles,
                                             data_to_spit=data,
                                             metric_dict=metric_dict,
                                             metric_list=metric_list_for_agents,
                                             n_entries_per_slice=n_entries_for_autoencoder)

        # PGD attack hyper-parameters (tune to your scenario)
        epsilon = 0.01  # Max L_inf perturbation per-feature (in raw units)
        max_iterations = 50  # iterations per restart (reduced for speed)
        alpha = 0.005  # step size for each random step
        restarts = 3

        for i, profile in enumerate(slice_profiles.keys()):
            # scale dl_buffer entries (if present) safely
            if len(data_tmp[i]) > 0:
                data_tmp[i] = np.array(data_tmp[i], dtype=np.float32)
                # dl_buffer is at index 0 in each slice-metric row (because metric_list_for_agents first entry is dl_buffer)
                # but our data_tmp rows correspond to metrics_export order: [slice_id, dl_buffer, tx_brate, slice_prb, ratio]
                # when split_data selected slice_metrics it used metric_list_for_agents, so data_tmp[i] columns are
                # [dl_buffer, tx_brate, ratio_granted_req] in that order. We earlier scaled dl_buffer at the raw-data
                # stage; keep consistent scaling before attack:
                # data_tmp[i] here is shape (n_entries, 3) and column 0 is dl_buffer in our slice_metrics
                data_tmp[i][:, 0] /= 100000.0

                # compute clean action to compare against (used by black-box attack's eval function)
                # Build the autoencoder input and TimeStep for the clean data
                try:
                    clean_obs = autoencoder.predict(np.expand_dims(data_tmp[i], axis=0)).astype('float32')
                except Exception:
                    clean_obs = np.zeros((1, 1), dtype='float32')

                prb_val = prbs[i][0] if (len(prbs[i]) > 0) else 0.0
                clean_obs_tmp = np.append(clean_obs, prb_val).astype('float32')
                time_step_clean = generate_timestep_for_policy(clean_obs_tmp)
                try:
                    clean_action = drl_agents[i].action(time_step_clean)[0][0][0].numpy()
                except Exception:
                    # if policy call fails, fallback to default policy
                    clean_action = previous_policy[i]

                # define evaluation function that returns how different the action is from clean_action
                def eval_fn(perturbed_arr):
                    # perturbed_arr has the same shape as data_tmp[i]
                    try:
                        obs_p = autoencoder.predict(np.expand_dims(perturbed_arr, axis=0)).astype('float32')
                        obs_tmp_p = np.append(obs_p, prb_val).astype('float32')
                        tsp = generate_timestep_for_policy(obs_tmp_p)
                        act_p = drl_agents[i].action(tsp)[0][0][0].numpy()
                        return float(abs(act_p - clean_action))
                    except Exception:
                        return -np.inf

                # apply the black-box PGD-like attack (gradient-free)
                perturbed_data = apply_pgd_attack(data_tmp[i],
                                                  eval_fn=eval_fn,
                                                  epsilon=epsilon,
                                                  num_iterations=max_iterations,
                                                  alpha=alpha,
                                                  restarts=restarts)

                logging.info('Testing iteration ' + str(i))
                logging.info('Data received from DU (dl_buffer [bytes], tx_brate downlink [Mbps], ratio_granted_req): ')
                logging.info(np.expand_dims(perturbed_data, axis=0))

                # create observation for agent using perturbed data
                obs_tmp = autoencoder.predict(np.expand_dims(perturbed_data, axis=0)).astype('float32')
                obs_tmp = np.append(obs_tmp, prb_val).astype('float32')

                reward_mean = np.mean(rewards[i]).astype('float32') if len(rewards[i]) > 0 else 0.0
                rewards_dict[profile].append(float(reward_mean))
                time_step = generate_timestep_for_policy(obs_tmp)
                try:
                    action = drl_agents[i].action(time_step)
                    chosen_action = action[0][0][0].numpy()
                except Exception:
                    chosen_action = previous_policy[i]

                # append policies to send and store policy
                policies.append(chosen_action)
                previous_policy[i] = chosen_action

                logging.info('Slice ' + str(i) + ': Action is ' + str(chosen_action) + ' Reward is: ' + str(reward_mean))
            else:
                # append previous policy for this slice
                policies.append(previous_policy[i])
                logging.info('Using previous action ' + str(previous_policy[i]) + ' for slice profile ' + str(i))

        print()
        # build message to send policies to the DU
        msg = ','.join([str(x) for x in policies])
        logging.info('Sending this message to the DU: ' + msg)
        # with open('rewards.json', 'w') as rewards_file:
        #     json.dump(rewards_dict, rewards_file)

        time.sleep(10)

        # Users may want to plug in their own functions to send the DRL policies
        # to the DU based on the specific DU implementation in use
        # send_action_to_DU(DU_address, msg)
