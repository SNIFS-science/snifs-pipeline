from enum import StrEnum

import plotly.graph_objects as go
import polars as pl


class PlotChoice(StrEnum):
    SCATTER = "Scatter"
    HISTOGRAM = "Histogram"


def plot_summary(df: pl.DataFrame) -> go.Figure:
    # The dataframe will have columns task_run_id, flow_run_id, discriminator, created_at, key, value_str, value_num
    fig = go.Figure()
    xs = df["created_at"].to_list()
    links = df["detailed_link"].to_list()
    names = df["name"].to_list()
    texts = []
    for link, name in zip(links, names, strict=True):
        if link:
            texts.append(f"<a href='{link}' target='_blank'>{name}</a>")
        else:
            texts.append(name)
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
        hovermode="closest",
        height=600,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
    )

    return fig


def plot_preprocess(
    df: pl.DataFrame,
    y_col: str = "num_bad_pixels_num",
    colour: str | None = None,
    plot_choice: PlotChoice = PlotChoice.SCATTER,
) -> go.Figure:
    # Here we want to follow a similar method to the above plot, but we'll
    # plot the number of bad pixels on the y-axis and the created_at on the x-axis.
    fig = go.Figure()
    xs = df["created_at"].to_list()
    ys = df[y_col].to_list()
    texts = []
    for row in df.to_dicts():
        texts.append(f"<a href='{row['detailed_link']}' target='_blank'>{row['name']}</a>")
    fig = go.Figure()
    title = y_col.replace("_num", "").replace("_", " ").title()
    if plot_choice == PlotChoice.SCATTER:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+lines+text",
                marker={"size": 10},
                text=texts,
                textposition="top center",
                hovertemplate=f"<b>Created At:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>",
                line={"color": colour} if colour else None,
            )
        )
        fig.update_xaxes(title_text="Created At")
        fig.update_yaxes(title_text=title)
    elif plot_choice == PlotChoice.HISTOGRAM:
        fig.add_trace(
            go.Histogram(
                x=ys,
                marker={"color": colour} if colour else None,
            )
        )
        fig.update_xaxes(title_text=title)

    fig.update_layout(
        hovermode="closest",
        height=600,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
    )
    return fig


def plot_scatter(data: pl.DataFrame, x_axis: str, y_axis: str, c_axis: str | None = None) -> go.Figure:
    fig = go.Figure()
    xs = data[x_axis].to_list()
    ys = data[y_axis].to_list()
    texts = []
    for row in data.to_dicts():
        texts.append(f"<a href='{row['detailed_link']}' target='_blank'>{row['name']}</a>")

    cbar_settings = {}
    caxis_text = ""
    if c_axis:
        cbar_settings = {
            "marker": {
                "color": data[c_axis].to_list(),
                "colorbar": {"title": c_axis.removesuffix("_num").replace("_", " ").title()},
            },
        }
        caxis_text = f"<br><b>{c_axis.removesuffix('_num').replace('_', ' ').title()}:</b> %{{marker.color}}"
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=texts,
            textposition="top center",
            hovertemplate=f"<b>{x_axis.replace('_', ' ').title()}:</b> %{{x}}"
            f"<br><b>{y_axis.replace('_', ' ').title()}:</b> %{{y}}"
            f"{caxis_text}<extra></extra>",
            **cbar_settings,
        )
    )
    fig.update_xaxes(title_text=x_axis.removesuffix("_num").replace("_", " ").title())
    fig.update_yaxes(title_text=y_axis.removesuffix("_num").replace("_", " ").title())
    fig.update_layout(
        hovermode="closest",
        height=600,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
    )

    return fig
