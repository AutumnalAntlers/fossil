import datetime
from fossil import config, core, science, ui
import streamlit as st
import streamlit.components.v1 as components
import datetime
import random
import io

@st.cache_data
def default_date():
    return datetime.datetime.utcnow() - datetime.timedelta(days=1)

@st.cache_data
def get_toots(_cache_key: int, timeline_since, n_clusters) -> list[core.Toot]:
    print("get_toots", _cache_key, st.session_state.cache_key)
    toots = core.Toot.get_toots_since(datetime.datetime.utcnow() - timeline_since)
    if len(toots) > 0:
        science.assign_clusters(toots, n_clusters=n_clusters)
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

# customize timeline segment to analyze
timeline_since = ui.get_time_frame()

# customize clustering algo
n_clusters = st.slider("Number of clusters", 2, 20, 5)

if "cache_key" not in st.session_state:
    print("init cache_key", st.session_state)
    st.session_state.cache_key = random.randint(0, 10000)

if st.button("Show"):
    st.session_state.cache_key = random.randint(0, 10000)

print(f"state: {st.session_state.cache_key}")

toots = get_toots(st.session_state.cache_key, timeline_since, n_clusters)
clusters = sorted(list({t.cluster for t in toots if t.cluster}))
if len(toots) == 0:
    st.markdown("No toots found. Try clicking **Download toots** or **Refresh toots** above and then click **Show**.")
else:
    st.vega_lite_chart(
        science.format_for_vega(toots, n_clusters),
        {
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
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
                "color": {"field": "label", "type": "nominal"},
                "href": {"field": "url", "type": "nominal"},
                "tooltip": [
                    {"field": "author", "type": "nominal"},
                    {"field": "content", "type": "nominal"}
                ]
            }
        })
    for cluster in clusters:
        cluster_count = len([t for t in toots if t.cluster == cluster])
        with st.expander(f"{cluster} ({cluster_count} toots)"):
            for toot in toots:
                if toot.cluster == cluster:
                    ui.display_toot(toot)
    # import code; code.interact(local=locals())
