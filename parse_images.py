''' Code to parse the images for Web URLs found in the MS-Marco dataset '''
''' Don't think we will need it actually '''

import argparse
import json
import requests
import pandas as pd
from typing import List, Union
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from tqdm import tqdm

def parse_args():
    
    parser = argparse.ArgumentParser(description='Parse images from URLs.')
    parser.add_argument('-i', '--input', required=True, help='Input file path.')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()
    return args

def is_internal(url, base_url):
    """Check if the URL is internal to the base URL."""
    return urlparse(url).netloc == urlparse(base_url).netloc

#TODO: Maybe just get the actual image
def fetch_images(url: str) -> Union[List[str], List[any]]:
    """Fetch images from a web page, excluding advertisements."""
    try:
        response = requests.get(url)
    except Exception as e:
        print(f'Error: {e}, URL: {url}')
        return [], []
    soup = BeautifulSoup(response.text, 'html.parser')
    images = []
    image_urls = []

    for img in soup.find_all('img'):
        img_url = img.attrs.get('src')
        if img_url and is_internal(img_url, url):
            # We join the URL to ensure it's absolute
            response = requests.get(urljoin(url, img_url))
            if response.headers.get('content-type') in ['image/jpeg', 'image/png']:
                image_urls.append(img_url)
                images.append(response.content)
            else:
                print(f'Invalid image: {img_url}')
    return image_urls, images

def get_urls(data: pd.DataFrame) -> List[str]:
    """Get the URLs from the dataframe."""
    return data['url'].tolist()

def main():
    """ Main function """
    print("Reading data...")
    if args.debug:
        data = pd.read_csv(args.input, sep='\t', header=None, nrows=100)
    else:
        data = pd.read_csv(args.input, sep='\t', header=None)
    data.columns = ['url', 'title', 'body']
    urls = get_urls(data)
    image_data = []

    #TODO: Get the docid and store the docid along with image url and image
    for docid, url in tqdm(enumerate(urls), total=len(urls)):
        # Replace with the actual URL
        image_urls, images = fetch_images(url)
        temp_image_data = {'docid': docid, 'image_urls': image_urls, 'images': images}
        image_data.append(temp_image_data)
    
    with open('data/image_data.jsonl', 'w') as f:
        for entry in image_data:
            f.write(f'{json.dumps(entry)}\n')

if __name__ == '__main__':
    args = parse_args()
    main()