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


with open(directory+'/channel_messages.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv(directory+'/channel_messages.csv', encoding='utf-8', index=False)

print("\nchannel_messages.json successfully converted to channel_messages.csv")

# with open(directory+'/channel_users.json', encoding='utf-8') as inputfile:
#     df = pd.read_json(inputfile)

# df.to_csv(directory+'/channel_users.csv', encoding='utf-8', index=False)

# print("\nchannel_users.json successfully converted to channel_users.csv\n")
# with open('./Datapoints - Messages and Multimedia 2' + 'channel_messages.json', encoding='utf-8') as inputfile:
#     df = pd.read_json(inputfile)

# df.to_csv('./Datapoints - Messages and Multimedia 2/ChannelUsers.csv', encoding='utf-8', index=False)
