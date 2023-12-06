# CS293S
Project repo for enhancing text search by using multimodal information

## Steps:
1. Preprocess MS-Marco dataset. Find out if this dataset has images. Otherwise use the different dataset that we found.
    - Parse the url for images using beautiful soup.
    - One time offline step.
    - statistics for number of images per document.
2. Use BM25 to get the top N documents for each query.
3. Embed the text of the documents and query using a Text Encoder ( maybe BERT or something else)
4. Embed the query and images in the documents using a Vision-Text Encoder (OpenClip or OpenFlamingo)
5. Use both the embeddings to train a model to rank the documents for each query.
OR
5. Use both the embeddings to directly compute a cosine score and compute a joint rank.
6. Evaluate the model on the MS-Marco Dataset.

## Comparisons:
1. Using image captions as part of the text.
2. Using image captions in place of image embeddings in above pipeline.
3. 