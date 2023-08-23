# adversarial_drl_ORAN
The code in this repository has three main modules: a) Base_line_Agent for original model    b) Surrogate FGSM attack model and  c) Resource Allocation Saboteur- Policy Infiltrator model
The code defines several functions for loading and preprocessing data from CSV files into Pandas DataFrames:

entire_dataset_from_single_file() loads a single CSV file into a DataFrame. It allows selecting specific columns, scaling the 'dl_buffer' column, removing rows with 0 requested PRBs, and adding a 'ratio_granted_req' column.

entire_dataset_from_folder() calls entire_dataset_from_single_file() on all CSV files in a folder and concatenates the results into one DataFrame.

extract_n_entries_from_dataset() randomly samples rows from a DataFrame.

get_data_from_DUs() either extracts random samples from a provided DataFrame, or generates random fake data if no DataFrame is passed.

The key steps are:

Load CSV data into DataFrames using entire_dataset_from_folder(). This preprocesses the data.

Extract samples from the DataFrame using extract_n_entries_from_dataset() or get_data_from_DUs(). This emulates getting real-time data.

The samples contain columns like 'dl_buffer', 'requested_prbs', etc. These are used as input features for the agents.

The agents (defined elsewhere in the code) take these features as input and output resource allocation actions.

The actions are executed in the environment, producing updated data samples.
