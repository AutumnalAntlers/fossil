import datetime
from fossil import config, core, science, ui
import streamlit as st
import streamlit.components.v1 as components
import datetime
import random
import io
from functools import partial

@st.cache_data
def default_date():
    return datetime.datetime.utcnow() - datetime.timedelta(days=1)

@st.cache_data
def get_toots(_cache_key: int, timeline_since) -> list[core.Toot]:
    print("get_toots", _cache_key, st.session_state.cache_key)
    toots = core.Toot.get_toots_since(datetime.datetime.utcnow() - timeline_since)
    return toots

# Refresh button
latest_date = core.Toot.get_latest_date()
if latest_date is None:
    is_refreshing = st.button("Download toots")
    if is_refreshing:
        with st.spinner("Downloading toots..."):
            core.create_database()
            core.download_timeline(datetime.datetime.utcnow() - datetime.timedelta(days=1))
            latest_date = core.Toot.get_latest_date()
        st.session_state.cache_key = random.randint(0, 10000)
else:
    is_refreshing = st.button("Refresh toots")
    if is_refreshing:
        with st.spinner("Downloading toots..."):
            core.create_database()
            core.download_timeline(latest_date)
        st.session_state.cache_key = random.randint(0, 10000)

if st.button("Reset"):
    st.session_state.cache_key = random.randint(0, 10000)

# customize timeline segment to analyze
timeline_since = ui.get_time_frame()

# customize clustering algo
n_clusters = st.slider("Number of clusters", 2, 20, 5)

kmeans_algo = st.radio("KMeans Algo", ["Spectral", "Standard"])

if "cache_key" not in st.session_state:
    print("init cache_key", st.session_state)
    st.session_state.cache_key = random.randint(0, 10000)

# Just keeping a list~
if "cache_tracker" not in st.session_state:
    print("init cache tracker", st.session_state)
    st.session_state.cache_tracker = []
def current_hash():
    hash((st.session_state.cache_key, kmeans_algo, n_clusters, timeline_since))
def current_hash_cached():
    if current_hash() in st.session_state.cache_tracker:
       return True
    else:
       return False
def current_hash_cache():
    if not current_hash_cached():
        st.session_state.cache_tracker += [current_hash()]

print(f"state: {st.session_state.cache_key}")

toots = get_toots(st.session_state.cache_key, timeline_since)
toots = [toot for toot in toots if toot.embedding is not None]

# This is just a hack for labels, actual desc. stuff is still at EOF.
if len(toots) != 0:
    embeddings, labels = science.assign_clusters(st.session_state.cache_key, toots, kmeans_algo, n_clusters=n_clusters)
    toot_desc = {}
    if current_hash_cached():
        for cluster_toots, count, desc in science.describe_clusters(hash, toots, labels):
            for toot in cluster_toots:
                toot_desc[toot] = desc

def format_node(i, toot, x, y):
    c = toot.content
    content = c[:97] + '...' if len(c) > 100 else c

    if current_hash_cached() and toot_desc[toot] != "":
        label = toot_desc[toot]
        if len(label) > 45:
            label = label[:42] + "..."
    else:
        label = i

    return {
        "Cluster": label,
        "x": x,
        "y": y,
        "author": toot.author,
        "content": content,
        # This doesn't seem to work very consistently, and I wanted to be able
        # to float-left the graph and hover over toots to highlight everything
        # from that cluster or account (so, interactive the other way around)
        # but ah well that's what I get for not attaching JS triggers to an
        # unencoded matplotlib SVG.
        "url": "#" + str(toot.id)
    }

if len(toots) == 0:
    st.markdown("No toots found. Try clicking **Download toots** or **Refresh toots** above and then click **Show**.")
else:
    data=[]
    for i in range(n_clusters):
      data.extend(map(partial(format_node, i),
                      toots,
                      embeddings[labels == i, 0],
                      embeddings[labels == i, 1],))
    st.vega_lite_chart(
        data,
        {
            "width": 700,
            "height": 500,
            "mark": {
                "type": "circle",
                "size": 60,
                "opacity": 0.8,
            },
            "params": [{
                "name": "view",
                "select": "interval",
                "bind": "scales"
            }],
            "config": {"axis": {"title": False}},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": {"domain": [-9, 11]}},
                "y": {"field": "y", "type": "quantitative", "scale": {"domain": [-7, 13]}},
                "color": {"field": "Cluster", "type": "nominal"},
                "href": {"field": "url", "type": "nominal"},
                "tooltip": [
                    {"field": "author", "type": "nominal"},
                    {"field": "content", "type": "nominal"}
                ]
            }
        })
    is_labeling = st.button("Label clusters", disabled=current_hash_cached())
    if is_labeling:
        with st.spinner("Labeling clusters..."):
            science.describe_toots(toots)
            current_hash_cache()
    if current_hash_cached():
        for toots, count, desc in science.describe_clusters(hash, toots, labels):
            with st.expander(f"{desc} ({count} toots)"):
                for toot in toots:
                    ui.display_toot(toot)

    # import code; code.interact(local=locals())
