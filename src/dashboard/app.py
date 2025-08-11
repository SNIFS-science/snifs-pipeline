import streamlit as st

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
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Number of Bad Pixels")
    st.plotly_chart(plot_preprocess(summaries, y_col="num_bad_pixels_num"))

with col2:
    st.markdown("### Overscan RMS")
    st.plotly_chart(plot_preprocess(summaries, y_col="rdnoise_num"))

with col3:
    st.markdown("### Overscan Median")
    st.plotly_chart(plot_preprocess(summaries, y_col="ovscmed_num"))
