import csv
import requests
import os

# specify the csv file name
file_name = input("Enter the name of the csv file (including .csv): ")

# create a directory with the same name as the csv file
directory_name = os.path.splitext(file_name)[0]
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# read the csv file and extract the links
with open(file_name, 'r') as file:
    reader = csv.reader(file)
    links = [row[0] for row in reader]

# download the media from each link and save it in the created folder
for i, link in enumerate(links):
    response = requests.get(link)
    extension = link.split(".")[-1]
    file_path = f"{directory_name}/{i}.{extension}"
    with open(file_path, "wb") as file:
        file.write(response.content)

print("Download complete.")
