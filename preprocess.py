import os
import json
import glob
from typing import Dict, List, Tuple
import requests
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import tensorflow.compat.v1 as tf
from collections import defaultdict
tf.logging.set_verbosity(tf.logging.ERROR)
Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/wikiweb2m/')
    parser.add_argument('--filepath', type=str, default='wikiweb2m-test')
    parser.add_argument('--split', type=str, default='test')

    args = parser.parse_args()

    args.image_dir = args.path + '/images'
    # make sure the image directory exists
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
    
    args.corpus_dir = args.path + '/documents'
    # make sure the document directory exists
    if not os.path.exists(args.corpus_dir):
        os.makedirs(args.corpus_dir)
    
    return args


class DataParser():
  def __init__(self,
               path: str,
               filepath: str):
    self.filepath = filepath
    self.path = path
    self.data = defaultdict(list)


  def parse_data(self):
    context_feature_description = {
        'split': tf.io.FixedLenFeature([], dtype=tf.string),
        'page_title': tf.io.FixedLenFeature([], dtype=tf.string),
        'page_url': tf.io.FixedLenFeature([], dtype=tf.string),
        'clean_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
        'raw_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
        'is_page_description_sample': tf.io.FixedLenFeature([], dtype=tf.int64),
        'page_contains_images': tf.io.FixedLenFeature([], dtype=tf.int64),
        'page_content_sections_without_table_list': tf.io.FixedLenFeature([] , dtype=tf.int64)
    }

    sequence_feature_description = {
        'is_section_summarization_sample': tf.io.VarLenFeature(dtype=tf.int64),
        'section_title': tf.io.VarLenFeature(dtype=tf.string),
        'section_index': tf.io.VarLenFeature(dtype=tf.int64),
        'section_depth': tf.io.VarLenFeature(dtype=tf.int64),
        'section_heading_level': tf.io.VarLenFeature(dtype=tf.int64),
        'section_subsection_index': tf.io.VarLenFeature(dtype=tf.int64),
        'section_parent_index': tf.io.VarLenFeature(dtype=tf.int64),
        'section_text': tf.io.VarLenFeature(dtype=tf.string),
        'section_clean_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
        'section_raw_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
        'section_rest_sentence': tf.io.VarLenFeature(dtype=tf.string),
        'is_image_caption_sample': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_url': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_mime_type': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_width': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_height': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_in_wit': tf.io.VarLenFeature(dtype=tf.int64),
        'section_contains_table_or_list': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_captions': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_alt_text': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_raw_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_clean_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_raw_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_clean_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_contains_images': tf.io.VarLenFeature(dtype=tf.int64)
    }


    def _parse_function(example_proto):
      return tf.io.parse_single_sequence_example(example_proto,
                                                 context_feature_description,
                                                 sequence_feature_description)

    suffix = '.tfrecord*'

    data_path = glob.glob(self.path + self.filepath + suffix)
    raw_dataset = tf.data.TFRecordDataset(data_path, compression_type='GZIP')
    parsed_dataset = raw_dataset.map(_parse_function)

    for d in parsed_dataset:
      split = d[0]['split'].numpy().decode()
      self.data[split].append(d)


def download_image(url: str, save_dir: str, img_id: str) -> str:
    """
    Download an image from the given URL and save it to the specified directory.
    
    :param url (str): The URL of the image to download.
    :param save_dir (str): The directory where the image will be saved.
    :param img_id (str): The ID to be used for the saved image file.

    :return str: The relative path to the saved image.
    """

    headers = {
        'User-Agent': 'UCSB CS293S (Linux)'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except (requests.HTTPError, requests.ConnectionError) as e:
        print(f"Failed to download image from {url}: {e}")
        return None

    # try catch logic to catch errors and fixing it by increasing the max pixels

    while True:
        try:
            image = Image.open(BytesIO(response.content))
            break
        except (Image.DecompressionBombError) as e:
            print(f"Failed to open image from {url} because of large size: {e}")
            Image.MAX_IMAGE_PIXELS = None

    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_path = os.path.join(save_dir, f"{img_id}.jpg")
    image.save(image_path)

    return image_path


def save_dict_to_json(dictionary: Dict, filepath: str) -> None:
    """
    Save a dictionary to a JSON file.

    :param dictionary (Dict): The dictionary to be saved.
    :param filepath (str): The path to the JSON file.
    """

    for k, v in dictionary.items():
        if isinstance(v, np.int64) or isinstance(v, np.float64):
            dictionary[k] = v.item()
        elif isinstance(v, np.ndarray):
            dictionary[k] = v.tolist()
        elif isinstance(v, bytes):
            dictionary[k] = v.decode()

    with open(filepath, 'w') as f:
        json.dump(dictionary, f)


def process_data_tf(parsed_dataset: tf.data.Dataset) -> None:
    """
    Process the parsed dataset and extract relevant information.

    :param parsed_dataset (tf.data.Dataset): The parsed dataset to be processed.

    :return (Tuple[Dict, List[Dict]]): A tuple containing the document map and the queries.
    """

    document_map = {}
    queries = []
    for i, d in tqdm(enumerate(parsed_dataset), total=len(parsed_dataset), desc='Processing data'):
        context = d[0]
        sequence = d[1]
        image_urls = [url.decode() for url in sequence['section_image_url'].values.numpy()]
        # filter urls that are not jpeg
        image_urls = [url for url in image_urls if url.endswith('.jpg')]
        combined_clean_text = ' '.join([text.decode() for text in sequence['section_text'].values.numpy()])
        image_captions = [caption.decode() for caption in sequence['section_image_captions'].values.numpy()]
        document = {
            k: v.numpy().decode() if v.dtype == np.dtype('O') else v.numpy() for k, v in context.items()
        }
        doc_id = f"{document['split']}_{i}"
        query_id = f"query_{doc_id}"

        # Download images and store local paths
        image_paths = []
        for img_id, url in enumerate(image_urls):
            image_path = download_image(url, args.image_dir, f"{doc_id}_{img_id}")
            if image_path is not None:
                image_paths.append(image_path)

        document.update({
            'text': combined_clean_text,
            'image_urls': image_urls,
            'image_paths': image_paths,
            'image_captions': image_captions,
        })

        # save the document in a separate file
        doc_path = os.path.join(args.corpus_dir, f"{doc_id}.json")
        save_dict_to_json(document, doc_path)

        document_map[doc_id] = doc_path
        query = {
            'query_id': query_id,
            'query': document['page_title'],
            'ground_truth_doc_id': doc_id
        }
        queries.append(query)

    # print some examples
    print('Example document:')
    print(f"{json.dumps(document, indent=4)}")
    print('Example query:')
    print(f"{json.dumps(queries[0], indent=4)}")

    return document_map, queries


def main():
    parser = DataParser(path=args.path, filepath=args.filepath)
    print('Parsing data...')
    parser.parse_data()

    document_map, queries = process_data_tf(parser.data[f'{args.split}'])
    
    with open(os.path.join(args.path, f'{args.split}_document_map.json'), 'w') as f:
        json.dump(document_map, f)

    with open(os.path.join('{args.path}, {args.split}_queries.json'), 'w') as f:
        for query in queries:
            json.dump(queries, f)
            f.write('\n')


if __name__ == '__main__':
    args = parse_args()
    main()
