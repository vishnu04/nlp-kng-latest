# Scrapper

import requests
from bs4 import BeautifulSoup
from . import config
# from htmllaundry import strip_markup

# importing the module as config
# import importlib.util as util
# spec = util.spec_from_file_location("/nlp_kng/", "config.py")
# config = util.module_from_spec(spec)
# spec.loader.exec_module(config)


# Function to remove tags
# @config.timer
def remove_tags(html):
    soup = BeautifulSoup(html, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    # print(cleaned_text)
    # cleaned_text = ' '.join(cleaned_text.split('\n'))
    return cleaned_text

@config.timer
def scrape_text(WEB_URL):
    page = requests.get(WEB_URL)
    text = remove_tags(page.content)
    return text

# if __name__ == "__main__":
#     # Page content from Website URL
#     page = requests.get(WEB_URL)
#     text = remove_tags(page.content)
#     # html_text = requests.get(url = config.WEB_URL).text
#     # text = html2text.html2text(html_text)
#     with open(f'{RAW_TEXT_FILE_NAME}','w') as file:
#         file.write(text)
#         file.close()
#     del page
#     del text
#     del file