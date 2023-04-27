import os
import pandas as pd
from googletrans import Translator

# Set up the directories
input_dir = "csv"
output_dir = "csv_english"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up the translator
translator = Translator()


# Function to translate the 'message' column
def translate_message(row):
    message = row["message"]

    # Skip empty fields
    if not message == "":
        return message

    try:
        translated_text = translator.translate(message, dest="en").text
        return translated_text
    except Exception as e:
        print(f"Error translating message: {e}")
        return message


# Iterate through files in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        # Read the CSV file
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path, low_memory=False)

        # Check if 'message' column exists
        if "message" in df.columns:
            # Translate the 'message' column
            df["message"] = df.apply(translate_message, axis=1)

            # Save the translated CSV file to the output directory
            output_file_path = os.path.join(output_dir, file)
            df.to_csv(output_file_path, index=False)
            print(f"Translated file saved to {output_file_path}")
        else:
            print(f"'message' column not found in {file_path}")
