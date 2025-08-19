import polars as pl
import polars.selectors as cs
import streamlit as st

from dashboard.database import SyncDatabase
from dashboard.plots import plot_scatter
from pipeline.resolver import Resolver

resolver = Resolver.create()
db = SyncDatabase(resolver.database_path)
discriminators = db.get_discriminators()


def create_page(
    data: pl.DataFrame,
    x_axis: str,
    y_axis: str,
    c_axis: str | None = None,
    discriminator: str | None = None,
    lookback_days: int | None = None,
):
    st.markdown(f"""# {x_axis.replace("_", " ").title()} vs {y_axis.replace("_", " ").title()}

""")
    st.plotly_chart(plot_scatter(data, x_axis + "_num", y_axis + "_num", c_axis + "_num" if c_axis else None))
    st.dataframe(data)


st.set_page_config(
    page_title="Custom Drilldown",
    page_icon="🛠️",
    layout="wide",
)

data = None
with st.sidebar:
    discriminator = st.query_params.get("discriminator", "All")
    days = int(st.query_params.get("days", 30))
    x_axis = st.query_params.get("x_axis", "azimuth")
    y_axis = st.query_params.get("y_axis", "altitude")
    c_axis = st.query_params.get("c_axis", "num_bad_pixels")

    st.header("Options")
    discriminator_options = ["All"] + discriminators
    discriminator = st.selectbox(
        "Discriminator",
        options=discriminator_options,
        index=discriminator_options.index(discriminator) if discriminator in discriminator_options else 0,
    )
    if discriminator == "All":
        discriminator = None

    num_days_to_look_back = st.number_input("Number of days to look back", min_value=0, value=days)

    data = db.get_summary_datas(discriminator=discriminator, days=num_days_to_look_back)
    numeric_cols = [x.removesuffix("_num") for x in data.select(cs.numeric()).columns]

    x_axis = st.selectbox(
        "X-Axis",
        options=numeric_cols,
        index=numeric_cols.index(x_axis) if x_axis in numeric_cols else 0,
    )
    y_axis = st.selectbox(
        "Y-Axis",
        options=numeric_cols,
        index=numeric_cols.index(y_axis) if y_axis in numeric_cols else 0,
    )
    c_axis = st.selectbox(
        "Color Axis",
        options=numeric_cols + ["None"],
        index=numeric_cols.index(c_axis) if c_axis in numeric_cols else 0,
    )
    if c_axis == "None":
        c_axis = None

    st.query_params["discriminator"] = discriminator
    st.query_params["days"] = num_days_to_look_back
    st.query_params["x_axis"] = x_axis
    st.query_params["y_axis"] = y_axis
    st.query_params["c_axis"] = c_axis

create_page(
    data=data,
    x_axis=x_axis,
    y_axis=y_axis,
    c_axis=c_axis,
    discriminator=discriminator,
    lookback_days=days,
)
