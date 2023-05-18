# Import necessary libraries
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import os
from keybert import KeyBERT
from sklearn.metrics import pairwise_distances_chunked

# Detect and use GPU when available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def generate_keywords(text, n_keywords=5):
    """
    Extracts keywords from a text.

    :param text: str, input text
    :param n_keywords: int, number of keywords to extract (default: 5)
    :return: list of str, extracted keywords
    """
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=n_keywords, stop_words='english')
    return [keyword for keyword, _ in keywords]


def cluster_messages_in_directory(csv_directory_path: str, model_name_or_path: str,
                                  output_dir: str = 'clustering_output', batch_size: int = 32,
                                  max_length: int = 128):
    """
    Clusters messages in CSV files located in the specified directory, using DBSCAN algorithm with BERT embeddings.
    The resulting clusters are saved in separate CSV files in the specified output directory.
    :param csv_directory_path: Path to the directory containing the CSV files
    :param model_name_or_path: Name or path of the BERT-based model to use for generating sentence embeddings
    :param output_dir: Path to the directory where the resulting clusters will be saved
    :param batch_size: Number of sentences to process in a single batch
    :param max_length: Maximum length of input sentences for the tokenizer
    """

    def mean_pooling(model_output, attention_mask):
        """
        Performs mean pooling on the output of a BERT-based model.
        :param model_output: Output of a BERT-based model
        :param attention_mask: Attention mask used for input to the BERT-based model
        :return: Mean-pooled sentence embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Load model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)

    # Hyperparameter tuning
    eps_values = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Iterate over all CSV files in the directory
    for csv_file_name in os.listdir(csv_directory_path):
        if csv_file_name.endswith('.csv'):
            csv_file_path = os.path.join(csv_directory_path, csv_file_name)

            # Read messages from file
            df = pd.read_csv(csv_file_path, low_memory=False)
            messages = list(df['message'])
            sentences = [str(x) for x in messages]

            # Process sentences in batches
            num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
            sentence_embeddings = []

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, len(sentences))
                batch_sentences = sentences[start_index:end_index]
                encoded_batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt',
                                          max_length=max_length)

                # Add this line to move the input tensor to the same device as the model
                encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}

                with torch.no_grad():
                    model_output_batch = model(**encoded_batch)

                sentence_embeddings_batch = mean_pooling(model_output_batch, encoded_batch['attention_mask'])
                sentence_embeddings.append(sentence_embeddings_batch)

            sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

            # Convert torch tensor to numpy array
            X = sentence_embeddings.cpu().numpy()

            best_num_clusters = 0
            best_parameters = (0, 0)

            # Perform hyperparameter tuning
            for eps in eps_values:
                for min_samples in min_samples_values:
                    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                    if hasattr(clustering, 'labels_'):
                        labels = clustering.labels_
                        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        if num_clusters > best_num_clusters:
                            best_num_clusters = num_clusters
                            best_parameters = (eps, min_samples)

            # Compute pairwise distances with cosine similarity
            distance_matrix = list(pairwise_distances_chunked(X, metric='cosine', n_jobs=-1))

            # Perform clustering using DBSCAN with the best parameters and a precomputed distance matrix
            clustering = DBSCAN(eps=best_parameters[0], min_samples=best_parameters[1], metric='precomputed').fit(
                distance_matrix)

            # Get clustering labels
            if hasattr(clustering, 'labels_'):
                labels = clustering.labels_

                # Add the labels to the dataframe
                df['cluster'] = labels

                # Extract the clusters
                clusters = df.groupby('cluster')
                noise_points = df[df['cluster'] == -1]

                # Create the output directory if it doesn't exist
                csv_output_dir = os.path.join(output_dir, csv_file_name[:-4])
                if not os.path.exists(csv_output_dir):
                    os.makedirs(csv_output_dir)

                # Save the noise points (unclassified messages) to a separate CSV file
                noise_file_name = f"{csv_output_dir}/unclassified.csv"
                noise_points.to_csv(noise_file_name, index=False)
                print(f"Unclassified messages saved to {noise_file_name}")

                # Initialize the dictionary to store cluster keywords for each directory
                cluster_keywords_dict = {}

                # Extract and save the clusters (including noise points) to separate CSV files
                for cluster_id, cluster in clusters:
                    file_name = f"{csv_output_dir}/cluster_{cluster_id}.csv"
                    cluster_messages = ' '.join(cluster['message'].astype(str))
                    cluster_keywords = generate_keywords(cluster_messages)
                    cluster['keywords'] = ', '.join(cluster_keywords)
                    cluster.to_csv(file_name, index=False)
                    print(f"Cluster {cluster_id} saved to {file_name}")

                    # Update the cluster_keywords_dict with the directory as the key and the keywords as the value
                    cluster_directory = f"{csv_output_dir}/cluster_{cluster_id}"
                    cluster_keywords_dict[cluster_directory] = cluster_keywords

                # Check if the cluster_keywords.py file exists, if not create one
                if not os.path.exists("../../Downloads/OneDrive-2023-05-12(1)/cluster_keywords.py"):
                    with open("../../Downloads/OneDrive-2023-05-12(1)/cluster_keywords.py", "w", encoding='utf-8') as f:
                        f.write("cluster_keywords_dict = {\n")
                        f.write("}\n")

                # Check if the file is empty
                file_empty = False
                with open("../../Downloads/OneDrive-2023-05-12(1)/cluster_keywords.py", "r", encoding='utf-8') as f:
                    content = f.read()
                    if content == "cluster_keywords_dict = {\n}\n":
                        file_empty = True

                # Write the cluster_keywords_dict to the Python file called cluster_keywords.py
                with open("../../Downloads/OneDrive-2023-05-12(1)/cluster_keywords.py", "a", encoding='utf-8') as f:
                    if file_empty:
                        f.write("cluster_keywords_dict = {\n")

                    for directory, keywords in cluster_keywords_dict.items():
                        f.write(f"    \"{directory}\": {keywords},\n")

                    if file_empty:
                        f.write("}\n")

                print("Cluster keywords dictionary written to cluster_keywords.py")


# Use a smaller model for better performance, such as 'distilbert-base-uncased'
cluster_messages_in_directory('/data/orojoa/Telegram_Project/Telegram_Dataset/Message_CSV', 'efederici/sentence-bert-base', '/data/orojoa/Telegram_Project/Script_Output/clustering_output', batch_size=32,
                              max_length=128)
