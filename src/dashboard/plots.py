import plotly.graph_objects as go
import polars as pl


def plot_summary(df: pl.DataFrame) -> go.Figure:
    # The dataframe will have columns task_run_id, flow_run_id, discriminator, created_at, key, value_str, value_num
    fig = go.Figure()
    xs = df["created_at"].to_list()
    links = df["link"].to_list()

    run_ids = df["run_id"].to_list()
    texts = []
    for link, run_id in zip(links, run_ids, strict=True):
        if link:
            texts.append(f"<a href='{link}' target='_blank'>{run_id}</a>")
        else:
            texts.append(run_id)
    y_ticks = df["discriminator"].unique().to_list()
    y_val_mapping = {y: i for i, y in enumerate(y_ticks)}
    ys = df["discriminator"].replace(y_val_mapping).to_list()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker={"size": 10},
            hovertemplate="<b>Flow Run ID:</b> %{text}<br><b>",
        )
    )
    # Make sure the y ticks are the discriminators
    fig.update_yaxes(
        tickvals=list(y_val_mapping.values()),
        ticktext=list(y_val_mapping.keys()),
        title_text="Discriminator",
    )
    fig.update_xaxes(title_text="Created At")
    fig.update_layout(
        title="Flow Runs Summary",
        hovermode="closest",
        height=600,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
    )

    return fig
