import pandas as pd
from googletrans import Translator

# Read the original CSV file
df = pd.read_csv('messages.csv')

# Create a translator object
translator = Translator()

# Define a function to translate non-English messages to English
def translate_message(message):
    if pd.notnull(message):
        try:
            translation = translator.translate(message)
            return translation.text
        except:
            return message
    else:
        return message

# Translate the messages and store them in a new column
df['translated_message'] = df['message'].apply(translate_message)

# Save the translated data to a new CSV file
df.to_csv('translated_messages.csv', index=False)
