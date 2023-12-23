import numpy as np
import openai
from . import core, config
import streamlit as st
from sklearn.cluster import KMeans, SpectralClustering
import openai
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

# def print_graphs(embeddings):
#   for n_clusters in range(2, 8):
#     kmeans = KMeans(n_clusters=n_clusters)
#     labels = kmeans.fit_predict(embeddings)
#     u_labels = np.unique(labels)
#     for i in u_labels:
#       plt.scatter(embeddings[label == i, 0], embeddings[label == i, 1], label = i)
#     plt.legend()
#     plt.show()

def get_it(prompt, client):
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
      continue
    except ValueError:
      continue

def toots_to_svg(port, toots: list[core.Toot], n_clusters: int = 5):
    print('Photographing')
    # meh, ignore toots without content. I think this might be just an image, not sure
    toots = [toot for toot in toots if toot.embedding is not None]

    # Perform k-means clustering on the embeddings
    embeddings = np.array([toot.embedding for toot in toots if type(toot.embedding) != type(None)])

    kmeans = SpectralClustering(
               n_clusters=5,
               affinity='nearest_neighbors',
               assign_labels='kmeans')
    cluster_labels = kmeans.fit_predict(embeddings)

    u_labels = np.unique(cluster_labels)

    # import code; code.interact(local=locals())

    for i in u_labels:
        print(f"plt.scatter({embeddings[cluster_labels == i , 0]} , {embeddings[cluster_labels == i , 1]})\n")
        plt.scatter(embeddings[cluster_labels == i , 0] , embeddings[cluster_labels == i , 1])
    plt.legend()
    plt.savefig(port, format='svg')

    st.pyplot(fig=plt)
    st.vega_lite_chart(
      embeddings,
      {
        "mark": {"type": "circle"},
        "encoding": {
          "x": {"field": {"repeat": "repeat"}, "type": "quantitative"},
          "y": {"field": {"repeat": "repeat"}, "type": "quantitative"}
        }
      })

def assign_clusters(toots: list[core.Toot], n_clusters: int = 5):
    print('Assigning')
    # meh, ignore toots without content. I think this might be just an image, not sure
    toots = [toot for toot in toots if toot.embedding is not None]

    # Perform k-means clustering on the embeddings
    embeddings = np.array([toot.embedding for toot in toots if type(toot.embedding) != type(None)])

    kmeans = SpectralClustering(
               n_clusters=5,
               affinity='nearest_neighbors',
               assign_labels='kmeans')
    cluster_labels = kmeans.fit_predict(embeddings)

    client = openai.OpenAI(api_key=config.OPENAI_KEY, base_url="http://localhost:8080/v1")
    for i_clusters in range(n_clusters):
        clustered_toots = [toot for toot, cluster_label in zip(toots, cluster_labels) if cluster_label == i_clusters]
        combined_text = "\n\n".join([toot.content for toot in clustered_toots])

        # Use GPT-3.5-turbo to summarize the combined text
        prompt = f"Create a single label that describes all of these related tweets, make it succinct but descriptive. The label should describe all {len(clustered_toots)} of these\n\n{combined_text}"
        response = get_it(prompt, client)
        summary = response.choices[0].message.content.strip()

        # Do something with the summary
        for toot, cluster_label in zip(toots, cluster_labels):
            if cluster_label == i_clusters:
                toot.cluster = summary

