import argparse
import json
import numpy as np
import glob
from tqdm import tqdm
import tensorflow.compat.v1 as tf
from collections import defaultdict
tf.logging.set_verbosity(tf.logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/wikiweb2m/')
    parser.add_argument('--filepath', type=str, default='wikiweb2m-test')
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()

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

def process_data_tf(parsed_dataset):
    documents = {}
    queries = []
    for i, d in tqdm(enumerate(parsed_dataset), total=len(parsed_dataset), desc='Processing data'):
        context = d[0]
        sequence = d[1]
        image_urls = [url.decode() for url in sequence['section_image_url'].values.numpy()]
        combined_clean_text = ' '.join([text.decode() for text in sequence['section_text'].values.numpy()])
        image_captions = [caption.decode() for caption in sequence['section_image_captions'].values.numpy()]
        document = {
            k: v.numpy().decode() if v.dtype == np.dtype('O') else v.numpy() for k, v in context.items()
        }
        doc_id = f"{document['split']}_{i}"
        query_id = f"query_{document['split']}_{i}"
        document.update({
            'text': combined_clean_text,
            'image_urls': image_urls,
            'image_captions': image_captions,
        })
        doc_id = f"{document['split']}_{i}"
        documents[doc_id] = document
        query = {
            'query_id': query_id,
            'query': document['page_title'],
            'ground_truth_doc_id': doc_id
        }
        queries.append(query)
    return documents, queries

def main():
    parser = DataParser(path=args.path, filepath=args.filepath)
    print('Parsing data...')
    parser.parse_data()
    documents, queries = process_data_tf(parser.data[f'{args.split}'])

    # serialize the documents
    for key, value in documents.items():
        documents[key] = {k: str(v) for k, v in value.items()}
    
    # print some examples
    print('Example document:')
    print(documents['test_0'])
    print('Example query:')
    print(queries[0])
    
    with open(f'{args.path}{args.split}_documents.jsonl', 'w') as f:
        json.dump(documents, f)

    with open(f'{args.path}{args.split}_queries.json', 'w') as f:
        json.dump(queries, f)

if __name__ == '__main__':
    args = parse_args()
    main()
