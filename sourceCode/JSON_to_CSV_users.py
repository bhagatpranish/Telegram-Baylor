import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Setting configuration values
api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
api_hash = str(api_hash)
phone = config['Telegram']['phone']
username = config['Telegram']['username']
entity = config['Telegram']['entity']

directory = "Channel-"+entity


with open(directory+'/channel_users.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv(directory+'/channel_users.csv', encoding='utf-8', index=False)
