import os
import pandas as pd
from keyword_dict import pdf_keywords_without_scores

keyword_dict = pdf_keywords_without_scores


def search_clusters(keywords, base_dir='csv', output_dir='clustering_keyword_search'):
    """
    Searches the output of the clustering for matching rows for each set of keywords.
    The matching rows are saved into separate CSV files in the specified output directory.
    :param keywords: Dictionary of keyword sets
    :param base_dir: Path to the directory containing the clustered CSV files
    :param output_dir: Path to the directory where the resulting CSV files will be saved
    """
    results = {}

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all directories in the base directory
    for directory in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, directory)

        # Check if the path is a directory
        if os.path.isdir(dir_path):
            # Iterate over all files in the directory
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)

                # Check if the path is a file and has a .csv extension
                if os.path.isfile(file_path) and file.endswith('.csv'):
                    # Read the clustered data
                    cluster_df = pd.read_csv(file_path)

                    # Check each keyword set
                    for category, keyword_list in keywords.items():
                        # Split keywords with multiple words into individual words for partial matching
                        split_keywords = [word for keyword in keyword_list for word in keyword.split()]

                        matched_rows = cluster_df['message'].astype(str).str.contains('|'.join(split_keywords),
                                                                                      case=False, na=False)

                        if matched_rows.any():
                            if category not in results:
                                results[category] = pd.DataFrame(columns=cluster_df.columns)

                            # Add matched rows to the results
                            matched_cluster = cluster_df[matched_rows].copy(deep=True)
                            matched_cluster['source_file'] = file_path
                            results[category] = pd.concat([results[category], matched_cluster], ignore_index=True)

    # Save the results into separate CSV files
    for category, df in results.items():
        output_file = f"{output_dir}/{category.replace(' ', '_').replace(':', '')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Matching rows for '{category}' saved to {output_file}")


search_clusters(keyword_dict)
