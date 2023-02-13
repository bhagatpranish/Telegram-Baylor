import configparser

from telethon.sync import TelegramClient, events
from tqdm import tqdm
import os
import shutil
from telethon.tl.types import (
    PeerChannel
)


# Reading Configs
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
destinationPath = directory
images_folder = entity


if entity.isdigit():
    channelId = PeerChannel(int(entity))
else:
    channelId = entity


# preset the folder and tagId with the channelId.
folderId = entity
tagId = entity


client = TelegramClient("session_name", api_id, api_hash)
client.start()

with TelegramClient("name", api_id, api_hash) as client:
    messages = client.get_messages(channelId, limit=50)
    for msg in tqdm(messages):
        msg.download_media(file=os.path.join(folderId, tagId))

print("\nAll media downloaded successfully")
files_to_move = [images_folder]

for file in files_to_move:
    destination = directory
    source = file
    shutil.move(source, destination)


print("\nMedia moved successfully")
