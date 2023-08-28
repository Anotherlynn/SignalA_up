import openai
import json
import pandas as pd
import numpy as np
import os

import copy
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from itertools import islice


def load_api_key(secrets_file):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]


api_key = load_api_key("../proj/secretS.json")
openai.api_key = api_key


EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

os.environ["http_proxy"] = "10.108.1.166:8088"
os.environ["https_proxy"] = "10.108.1.166:8088"

# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3),
       retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]


# chunk the input
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


# encode the string into tokens
def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


# check the input size

def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


if __name__ == '__main__':
    df = pd.read_csv("../data/report_clean5.csv")
    Flag =True
    for id in range(0,df.shape[0],3000):
        while Flag:
            df_temp = copy.deepcopy(df).loc[id:id+3000]
            embedding = []

            for i,t in df.loc[id:id+3000].iterrows():
                print(i)
                try:
                    x = len_safe_get_embedding(t.Detail, average=True)
                    print("sucess")
                except:
                    x = np.nan
                    print("ERROR")

                embedding.append(x)
            df_temp['embedding'] = embedding
            df_temp.to_csv("data/report_clean_embed_%s.csv"%str(id))
            if np.nan==embedding:
                Flag=False
                print("error!!!!")

# try to use steam //

# try to use parallel request

# def check(str,data,str_replace):
#     x = data['Detail'].apply(lambda x: " ".join(x.split()[:2])).str.contains(str)
#     idx = data[x].index.to_list()
#     data.loc[idx,'Tag'] = str_replace
#     return data

