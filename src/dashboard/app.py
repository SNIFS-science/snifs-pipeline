import streamlit as st

from dashboard.database import SyncDatabase
from dashboard.plots import plot_summary
from pipeline.resolver import Resolver

resolver = Resolver.create()
db = SyncDatabase(resolver.database_path)
discriminators = db.get_discriminators()

st.set_page_config(page_title="SNIFS Pipeline Dashboard", page_icon=":telescope:")
st.title("SNIFS Pipeline Dashboard")
with st.sidebar:
    st.header("Filters")
    selected_discriminator = st.selectbox("Select Discriminator", options=["All"] + discriminators)
    if selected_discriminator == "All":
        selected_discriminator = None


summaries = db.get_summary_datas(selected_discriminator)
st.subheader("Summaries")

st.plotly_chart(plot_summary(summaries))
