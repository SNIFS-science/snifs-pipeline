from itertools import cycle

import polars.selectors as cs
import streamlit as st

from dashboard.colours import colour_palette
from dashboard.database import SyncDatabase
from dashboard.plots import plot_preprocess, plot_summary
from pipeline.resolver import Resolver

resolver = Resolver.create()
db = SyncDatabase(resolver.database_path)
discriminators = db.get_discriminators()

st.set_page_config(page_title="SNIFS Pipeline Dashboard", page_icon=":telescope:", layout="wide")
st.title("SNIFS Pipeline Dashboard")
with st.sidebar:
    with st.form(key="config"):
        st.header("Filters")
        selected_discriminator = st.selectbox("Select Discriminator", options=["All"] + discriminators)
        if selected_discriminator == "All":
            selected_discriminator = None

        st.form_submit_button("Submit")

summaries = db.get_summary_datas(selected_discriminator)
st.subheader("Summaries")

st.plotly_chart(plot_summary(summaries))

st.subheader("Preprocess Exposure Breakdown")
cols_to_plot = summaries.select(cs.ends_with("_num") & ~cs.contains("time")).columns
cols = cycle(st.columns(3))
colour_palette = colour_palette.colour_cycler()

for column_name in cols_to_plot:
    col = next(cols)
    colour = next(colour_palette)
    with col:
        st.markdown(f"### {column_name.replace('_num', '').replace('_', ' ').title()}")
        st.plotly_chart(plot_preprocess(summaries, y_col=column_name, colour=colour))
