import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
import csv

# Read the CSV file
with open('urls.csv', 'r') as file:
    reader = csv.reader(file)
    urls = [row[0] for row in reader]

# Start the web driver
driver = webdriver.Edge("msedgedriver.exe")

for url in urls:
    filename = url.split("me/")[1]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Load the URL
    driver.get(url)

    # Scroll to the top of the page
    for x in range(200):
        driver.execute_script("window.scrollTo(0, 0)")

        # Wait for a few seconds to allow the page to load
        time.sleep(2)

    # Get the page source
    html = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html, "html.parser")

    # Continue with the scraping process as before
    message_wrap = []
    message_wrap_index = 0
    message = []
    texts = []
    video_urls = []
    links = []
    photos = []

    for div in soup.find_all("div", class_="tgme_widget_message_wrap js-widget_message_wrap"):

        video_tag_list = div.find_all("video")

        for video in video_tag_list:
            video_urls.append(video['src'])

        text_tag_list = div.find_all(
            "div", class_="tgme_widget_message_text js-message_text before_footer")

        for text in text_tag_list:
            texts.append(text.text)
            link = text.find(
                "a")
            if link is not None:
                links.append(link['href'])
            else:
                continue

        photo_tag_list = div.find_all(
            "a", class_="tgme_widget_message_photo_wrap")
        for photo in photo_tag_list:
            photos.append(photo['href'])

        message_wrap_index += 1

    df = pd.DataFrame({"video_url": video_urls})
    dc = pd.DataFrame({"text": texts})
    dl = pd.DataFrame({"link": links})
    dp = pd.DataFrame({"photo": photos})

    df.to_csv(f"{filename}_videos.csv", index=True)
    dc.to_csv(f"{filename}_texts.csv", index=True)
    dl.to_csv(f"{filename}_links.csv", index=True)
    dp.to_csv(f"{filename}_photos.csv", index=True)

# Close the web driver
driver.quit()
