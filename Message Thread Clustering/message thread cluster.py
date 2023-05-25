import os
import pandas as pd
import torch
from transformers import BertTokenizer, logging
import warnings


def message_cluster_analysis(model_path, input_folder, output_folder, min_matches):
    """
    This function takes in path to the pretrained model, input and output directories,
    performs a clustering analysis on the messages in each csv file in the input directory
    and saves the result into the output directory.

    :param model_path: str, path to the pretrained model
    :param input_folder: str, path to the input directory containing csv files
    :param output_folder: str, path to the output directory to save the results
    :param min_matches: int, minimum number of messages a cluster must have to be included in the final output
    """

    # Helper function to format message count
    def format_count(row):
        return f"{row['message_number']} of {row['total_count']}"

    # Set transformers logging level to ERROR to suppress unwanted messages
    logging.set_verbosity_error()

    # Suppress python warnings
    warnings.filterwarnings('ignore')

    # Load the pretrained model
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            # Load CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure 'message' column exists
            assert 'message' in df.columns, f"{filename} does not contain a 'message' column."

            # Create a DataFrame to store clustered messages
            cluster_df = pd.DataFrame(columns=['cluster_id', 'message'])

            # Initialize cluster_id
            cluster_id = 1

            # Loop through each pair of messages in the DataFrame
            for i in range(1, len(df['message'])):
                text_1 = df['message'][i - 1]
                text_2 = df['message'][i]

                # Ignore duplicate messages
                if text_1 == text_2:
                    continue

                # Check if texts are string
                if not isinstance(text_1, str) or not isinstance(text_2, str):
                    continue

                # Tokenize the texts
                inputs = tokenizer(text_1, text_2, return_tensors='pt', truncation='longest_first', padding=True)

                # Generate prediction
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits).item()

                # Check prediction
                if prediction == 0:
                    # If messages are not in context, increment the cluster_id
                    cluster_id += 1

                # Add messages to the respective cluster in the DataFrame
                cluster_df.loc[len(cluster_df)] = [cluster_id, text_2]

            # Format the 'message_count' column
            cluster_df['total_count'] = cluster_df.groupby('cluster_id')['message'].transform('count')
            cluster_df['message_number'] = cluster_df.groupby('cluster_id').cumcount() + 1
            cluster_df['message_count'] = cluster_df.apply(format_count, axis=1)

            # Filter the clusters based on min_matches
            cluster_df = cluster_df[cluster_df['total_count'] >= min_matches]

            # Create a directory for each input file in the output folder
            file_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(file_output_folder, exist_ok=True)

            # Group the DataFrame by 'cluster_id' and iterate over the groups
            for cluster_id, group in cluster_df.groupby('cluster_id'):
                # Format the cluster_id for filename
                formatted_cluster_id = str(cluster_id).zfill(4)  # This pads the cluster_id with leading zeros

                # Create an output filename for the cluster
                output_filename = f"cluster_{formatted_cluster_id}.csv"

                # Create a full output file path
                output_file_path = os.path.join(file_output_folder, output_filename)

                # Write the group to a CSV file
                group.to_csv(output_file_path, index=False)

            # Print console feedback
            print(f"[+] Completed clustering for file: {filename}")

    print(f"[+] Completed Clustering Job")


# Use the function
message_cluster_analysis('mode.pt', "../clustering_keyword_search", "../output_clusters", 5)
