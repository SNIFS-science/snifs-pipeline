import base64
from pathlib import Path

import streamlit as st

from dashboard.database import SyncDatabase
from pipeline.resolver import Resolver

resolver = Resolver.create()


def plot_hd_image(path: Path):
    with path.open("rb") as f:
        image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
    st.markdown(f'<img src="data:image/webp;base64,{image_b64}" style="width:100%"/>', unsafe_allow_html=True)


def create_page(task_run_id: str):
    db = SyncDatabase(resolver.database_path)
    summary = db.get_summary_datas(task_run_id=task_run_id)

    st.markdown(f"""# Detailed View

**Task Run ID: {task_run_id}**

""")

    if summary.height > 1:
        st.warning("Multiple rows found in summary - there should only be one!")
        return
    lookup = summary.to_dicts()[0]
    prefect_link = lookup.get("prefect_link", "")
    if prefect_link:
        st.markdown(f"""
**[Click here to view the Prefect run logs and artifacts]({prefect_link})**""")

    st.markdown("Summary Table:")
    st.dataframe(summary.unpivot().sort("variable"))

    # Find image files in the public directory
    public_dir = lookup.get("public_dir", "")
    if public_dir:
        extensions = ["png", "jpg", "jpeg", "gif", "webp"]
        images = sorted([x for e in extensions for x in Path(public_dir).glob(f"*.{e}")])
        if images:
            st.markdown("### Images")
            for img in images:
                st.markdown(f"### {img.stem}")
                plot_hd_image(img)


st.set_page_config(
    page_title="Task Details",
    page_icon="🔍",
    layout="wide",
)

if "task_run_id" in st.query_params:
    create_page(task_run_id=st.query_params["task_run_id"])
else:
    st.write("No task specified. Please provide a task ID in the query parameters.")
