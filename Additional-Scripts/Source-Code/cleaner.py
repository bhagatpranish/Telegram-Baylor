import pandas as pd
import re
import ast


# Read the csv file
df = pd.read_csv('channel_messages.csv')


# Replace the json format in the "peer_id" column with just the value of "channel_id"
def process_peer_id(x):
    match = re.search(r'(\d+)', x)
    if match:
        return int(match.group(1))
    return x


df["peer_id"] = df["peer_id"].apply(process_peer_id)

df['from_id"'] = ''
for i, row in df.iterrows():
    if pd.notnull(row['from_id']):
        x = ast.literal_eval(row['from_id'])
        df.at[i, 'from_id'] = x.get('user_id', '')

# Extract the dictionary in the "media" column into separate columns
df['media_type'] = ''
df['photo_id'] = ''
df['access_hash'] = ''
df['file_reference'] = ''
df['date'] = ''
df['sizes'] = ''
df['dc_id'] = ''
df['has_stickers'] = ''
df['video_sizes'] = ''
df['ttl_seconds'] = ''

for i, row in df.iterrows():
    if pd.notnull(row['media']):
        x = ast.literal_eval(row['media'])
        df.at[i, 'media_type'] = x.get('_', '')
        try:
            photo = x['photo']
            df.at[i, 'photo_id'] = photo.get('id', '')
            df.at[i, 'access_hash'] = photo.get('access_hash', '')
            df.at[i, 'file_reference'] = photo.get('file_reference', '')
            df.at[i, 'date'] = photo.get('date', '')
            df.at[i, 'sizes'] = photo.get('sizes', '')
            df.at[i, 'dc_id'] = photo.get('dc_id', '')
            df.at[i, 'has_stickers'] = photo.get('has_stickers', '')
            df.at[i, 'video_sizes'] = photo.get('video_sizes', '')
        except KeyError:
            df.at[i, 'photo_id'] = ''
            df.at[i, 'access_hash'] = ''
            df.at[i, 'file_reference'] = ''
            df.at[i, 'date'] = ''
            df.at[i, 'sizes'] = ''
            df.at[i, 'dc_id'] = ''
            df.at[i, 'has_stickers'] = ''
            df.at[i, 'video_sizes'] = ''
        df.at[i, 'ttl_seconds'] = x.get('ttl_seconds', '')
    else:
        df.at[i, 'media_type'] = ''
        df.at[i, 'photo_id'] = ''
        df.at[i, 'access_hash'] = ''
        df.at[i, 'file_reference'] = ''
        df.at[i, 'date'] = ''
        df.at[i, 'sizes'] = ''
        df.at[i, 'dc_id'] = ''
        df.at[i, 'has_stickers'] = ''
        df.at[i, 'video_sizes'] = ''
        df.at[i, 'ttl_seconds'] = ''


df.drop(columns=['media'], inplace=True)\

# entities
df['Message_Entity'] = ''
df['offset'] = ''
df['length'] = ''
for i, row in df.iterrows():
    if pd.notnull(row['entities']):
        x = ast.literal_eval(row['entities'])
        for j in range(len(x)):
            try:
                df.at[i, f'Message_Entity_[{j}]'] = x[j].get('_', '')
                df.at[i, f'offset_[{j}]'] = x[j].get('offset', '')
                df.at[i, f'length_[{j}]'] = x[j].get('length', '')
            except KeyError:
                df.at[i, f'Message_Entity_[{j}]'] = ''
                df.at[i, f'offset_[{j}]'] = ''
                df.at[i, f'length_[{j}]'] = ''
    else:
        df.at[i, 'Message_Entity'] = ''
        df.at[i, 'offset'] = ''
        df.at[i, 'length'] = ''

# reply_to
{'_': 'MessageReplyHeader', 'reply_to_msg_id': 3238,
    'reply_to_peer_id': None, 'reply_to_top_id': None}

df['MessageReplyHeader'] = ''
df['reply_to_msg_id'] = ''
df['reply_to_peer_id'] = ''
df['reply_to_top_id'] = ''

for i, row in df.iterrows():
    if pd.notnull(row['reply_to']):
        x = ast.literal_eval(row['reply_to'])
        try:
            df.at[i, 'MessageReplyHeader'] = x.get('_', '')
            df.at[i, 'reply_to_msg_id'] = x.get('reply_to_msg_id', '')
            df.at[i, 'reply_to_peer_id'] = x.get('reply_to_peer_id', '')
            df.at[i, 'reply_to_top_id'] = x.get('reply_to_top_id', '')
        except KeyError:
            df.at[i, 'MessageReplyHeader'] = ''
            df.at[i, 'reply_to_msg_id'] = ''
            df.at[i, 'reply_to_peer_id'] = ''
            df.at[i, 'reply_to_top_id'] = ''
    else:
        df.at[i, 'MessageReplyHeader'] = ''
        df.at[i, 'reply_to_msg_id'] = ''
        df.at[i, 'reply_to_peer_id'] = ''
        df.at[i, 'reply_to_top_id'] = ''


# replies
df['_replies'] = ''
df['replies_count'] = ''
df['replies_pts'] = ''
df['comments'] = ''
df['recent_repliers'] = ''
df['channel_id'] = ''
df['max_id'] = ''
df['read_max_id'] = ''


for i, row in df.iterrows():
    if pd.notnull(row['replies']):
        x = ast.literal_eval(row['replies'])
        try:
            df.at[i, '_replies'] = x.get('_', '')
            df.at[i, 'replies_count'] = x.get('replies', '')
            df.at[i, 'replies_pts'] = x.get('replies_pts', '')
            df.at[i, 'comments'] = x.get('comments', '')
            df.at[i, 'recent_repliers'] = x.get('recent_repliers', '')
            df.at[i, 'channel_id'] = x.get('channel_id', '')
            df.at[i, 'max_id'] = x.get('max_id', '')
            df.at[i, 'read_max_id'] = x.get('read_max_id', '')
        except KeyError:
            df.at[i, '_replies'] = ''
            df.at[i, 'replies'] = ''
            df.at[i, 'replies_pts'] = ''
            df.at[i, 'comments'] = ''
            df.at[i, 'recent_repliers'] = ''
            df.at[i, 'channel_id'] = ''
            df.at[i, 'max_id'] = ''
            df.at[i, 'read_max_id'] = ''
    else:
        df.at[i, '_replies'] = ''
        df.at[i, 'replies'] = ''
        df.at[i, 'replies_pts'] = ''
        df.at[i, 'comments'] = ''
        df.at[i, 'recent_repliers'] = ''
        df.at[i, 'channel_id'] = ''
        df.at[i, 'max_id'] = ''
        df.at[i, 'read_max_id'] = ''

# Save the processed data back to the csv file
df.to_csv("channel_messages_cleaned.csv", index=False)

# Read the csv file
df = pd.read_csv('channel_messages_cleaned.csv')

df['photo_stripped_size_type'] = ''
df['photo_stripped_size_bytes'] = ''
df['photo_size_type'] = ''
df['photo_size_w'] = ''
df['photo_size_h'] = ''
df['photo_size_size'] = ''
df['photo_size_progressive_type'] = ''
df['photo_size_progressive_w'] = ''
df['photo_size_progressive_h'] = ''
df['photo_size_progressive_sizes'] = ''

for i, row in df.iterrows():
    if pd.notnull(row['sizes']):
        x = ast.literal_eval(row['sizes'])
        for size in x:
            if size['_'] == 'PhotoStrippedSize':
                df.at[i, 'photo_stripped_size_type'] = size.get('type', '')
                df.at[i, 'photo_stripped_size_bytes'] = size.get('bytes', '')
            elif size['_'] == 'PhotoSize':
                df.at[i, 'photo_size_type'] = size.get('type', '')
                df.at[i, 'photo_size_w'] = size.get('w', '')
                df.at[i, 'photo_size_h'] = size.get('h', '')
                df.at[i, 'photo_size_size'] = size.get('size', '')
            elif size['_'] == 'PhotoSizeProgressive':
                df.at[i, 'photo_size_progressive_type'] = size.get('type', '')
                df.at[i, 'photo_size_progressive_w'] = size.get('w', '')
                df.at[i, 'photo_size_progressive_h'] = size.get('h', '')
                df.at[i, 'photo_size_progressive_sizes'] = size.get(
                    'sizes', '')
    else:
        df.at[i, 'photo_stripped_size_type'] = ''
        df.at[i, 'photo_stripped_size_bytes'] = ''
        df.at[i, 'photo_size_type'] = ''
        df.at[i, 'photo_size_w'] = ''
        df.at[i, 'photo_size_h'] = ''
        df.at[i, 'photo_size_size'] = ''
        df.at[i, 'photo_size_progressive_type'] = ''
        df.at[i, 'photo_size_progressive_w'] = ''
        df.at[i, 'photo_size_progressive_h'] = ''
        df.at[i, 'photo_size_progressive_sizes'] = ''

df.drop(columns=['sizes'], inplace=True)
df.drop(columns=['replies'], inplace=True)
df.drop(columns=['entities'], inplace=True)
df.drop(columns=['reply_to'], inplace=True)
df.to_csv("channel_messages_cleaned.csv", index=False)
