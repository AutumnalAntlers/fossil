import numpy as np
import openai
from . import core, config
import streamlit as st
from sklearn.cluster import KMeans, SpectralClustering
import openai
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

def resolve_kmeans_algo(kmeans_algo, n_clusters):
  if kmeans_algo == "Standard":
      return KMeans(n_clusters=n_clusters)
  elif kmeans_algo == "Spectral":
      return SpectralClustering(
                 n_clusters=n_clusters,
                 affinity='nearest_neighbors',
                 assign_labels='kmeans')

@st.cache_data
def assign_clusters(hash: int, _toots: list[core.Toot], kmeans_algo, n_clusters):
    print('Assigning')
    kmeans = resolve_kmeans_algo(kmeans_algo, n_clusters)
    embeddings = np.array([toot.embedding for toot in _toots if type(toot.embedding) != type(None)])
    labels = kmeans.fit_predict(embeddings)
    return embeddings, labels

def generate_with_retries(prompt, client):
  response = None
  while response == None:
    try:
      response = client.chat.completions.create(
                   model=config.SUMMARIZE_MODEL.name,
                   messages=[{"role": "user", "content": core.reduce_size(prompt)}],
                   max_tokens=100)
      if response is not None:
        return response
      else:
        raise ValueError('response is None')
    except openai.InternalServerError:
      sleep(10)
      continue
    except ValueError:
      sleep(10)
      continue

def describe_toots(toots: list[core.Toot]):
    client = openai.OpenAI(api_key=config.OPENAI_KEY, base_url="http://localhost:8080/v1")
    joined_text = "\n\n".join([toot.content for toot in toots])

    # Use GPT-3.5-turbo to summarize the joined text
    prompt = f"Create a single label that describes all of these related tweets, make it succinct but descriptive. The label should describe all {len(toots)} of these:\n\n{joined_text}\nSummery label: "
    response = generate_with_retries(prompt, client)
    summary = response.choices[0].message.content.strip()

    return summary

@st.cache_data
def describe_clusters(hash: int, _toots: list[core.Toot], labels: list[int]):
    clusters = {label: {"toots": []} for label in labels}
    for toot, label in zip(_toots, labels):
        clusters[label]['toots'] += [toot]
    for label, cluster in clusters.items():
        cluster['count'] = len(cluster['toots'])
        cluster['desc'] = describe_toots(cluster['toots'])
    return [(c['toots'], c['count'], c['desc']) for c in clusters.values()]
