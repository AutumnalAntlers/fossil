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

from streamlit.runtime.caching.cache_utils import _make_value_key
from streamlit.runtime.caching.cache_errors import CacheKeyNotFoundError
import types
from typing import Any

def is_cached(
        func: types.FunctionType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] = {}
    ) -> tuple[bool, Any]:

    cached_func = func.__dict__['clear'].__self__
    info = cached_func._info
    cache = info.get_function_cache(cached_func._function_key)
    value_key= _make_value_key(
        cache_type=info.cache_type,
        func=info.func,
        func_args=args,
        func_kwargs=kwargs,
        hash_funcs=info.hash_funcs
    )

    try:
        return True, cache.read_result(value_key).value
    except CacheKeyNotFoundError:
        return False, None

@st.cache_data
def test(var):
    return var

cached_p, value = is_cached(test, (1,))
st.write((cached_p, value))
# => (False, None)

test(1)
cached_p, calue = is_cached(test, (1,))
st.write((cached_p, value))
# => (True, 1)

print(f"state: {st.session_state.cache_key}")

toots = get_toots(st.session_state.cache_key, timeline_since)
toots = [toot for toot in toots if toot.embedding is not None]

def current_hash():
    hash((st.session_state.cache_key, kmeans_algo, n_clusters, timeline_since))

if "last_clusters" not in st.session_state:
    st.session_state.last_clusters = None

# Initialize `described_clusters' to cached value or empty list
if not len(toots) == 0:
    embeddings, labels = science.assign_clusters(st.session_state.cache_key, toots, kmeans_algo, n_clusters=n_clusters)
    cached, described_clusters = is_cached(science.describe_clusters, (current_hash(), toots, labels), {})
    if described_clusters is not None:
        st.session_state.last_clusters = (current_hash(), described_clusters)
    elif st.session_state.last_clusters is not None:
        described_clusters = st.session_state.last_clusters[1]
    else:
        described_clusters = []

def format_node(i, toot, x, y):
    c = toot.content
    content = c[:97] + '...' if len(c) > 100 else c

    descs = [
        desc
        for toots, count, desc in described_clusters
        if toot in toots
    ]
    if cached and len(descs) > 0:
        label = descs[0]
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
    describe_button_type = 'primary' if st.session_state.last_clusters is not None and current_hash() != st.session_state.last_clusters[0] else 'secondary'
    is_describing = st.button("Describe clusters", disabled=cached, type=describe_button_type)
    if is_describing:
        with st.spinner("Describing clusters..."):
            science.describe_clusters(current_hash(), toots, labels)
            st.session_state.last_clusters = (current_hash(), described_clusters)
    for toots, count, desc in described_clusters:
        with st.expander(f"{desc} ({count} toots)"):
            for toot in toots:
                ui.display_toot(toot)

    # import code; code.interact(local=locals())
