"""Plotly charts used by the dashboard. Each function takes a tidy frame and returns a figure."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fixed categorical order (validated palette): slot 1 blue, slot 2 orange, slot 3 aqua.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, NEUTRAL = "#d03b3b", "#9a9893"
SEQUENTIAL = [[0, "#cde2fb"], [0.33, "#6da7ec"], [0.66, "#256abf"], [1, "#0d366b"]]
DIVERGING = [[0, "#184f95"], [0.5, "#f0efec"], [1, "#b42f2f"]]


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def time_profile(profile: pd.DataFrame) -> go.Figure:
    """Demand and observed frequency per interval: two aligned panels (never a dual axis)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=("Passenger demand (sum of link loads per 15 min)",
                                        "Observed train allocations (sum of link frequencies per 15 min)"))
    fig.add_trace(go.Scatter(x=profile["interval_start"], y=profile["demand"], name="Demand", mode="lines",
                             line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.12)",
                             hovertemplate="%{y:,.0f} passengers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=profile["interval_start"], y=profile["frequency"], name="Observed trains",
                             mode="lines", line=dict(color=ORANGE, width=2, shape="hv"),
                             hovertemplate="%{y:,.0f} train allocations"), row=2, col=1)
    fig.update_xaxes(nticks=24, row=2, col=1)
    fig.update_annotations(font_size=12, x=0, xanchor="left")
    return _style(fig, "Demand and service across the operating day (05:00 → 04:59)", 520)


def top_links(table: pd.DataFrame, label: str, value: str, title: str, n: int = 15) -> go.Figure:
    top = table.nlargest(n, value).iloc[::-1]
    fig = go.Figure(go.Bar(x=top[value], y=top[label], orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(tickfont=dict(size=11))
    return _style(fig, title, max(320, 26 * len(top) + 80))


def period_bars(summary: pd.DataFrame, periods: list[str]) -> go.Figure:
    totals = summary[periods].sum()
    fig = go.Figure(go.Bar(x=periods, y=totals.values, marker_color=BLUE,
                           text=[f"{v:,.0f}" for v in totals.values], textposition="outside",
                           hovertemplate="%{x}: %{y:,.0f} passenger-link trips<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, "Demand by NUMBAT time-of-day period (from the workbook's period columns)", 340)


def heatmap(pivot: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=SEQUENTIAL,
                               colorbar=dict(title="Passengers"),
                               hovertemplate="%{y}<br>%{x}: %{z:,.0f} passengers<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return _style(fig, title, max(380, 16 * len(pivot) + 120))


def demand_vs_service(obs: pd.DataFrame) -> go.Figure:
    """Each point is one link-interval: offered effective capacity (observed trains x C_eff) vs demand."""
    over = obs["demand"] > obs["offered"]
    fig = go.Figure()
    for mask, name, color in [(~over, "Within effective capacity", BLUE), (over, "Demand above effective capacity", CRITICAL)]:
        d = obs[mask]
        fig.add_trace(go.Scattergl(
            x=d["offered"], y=d["demand"], mode="markers", name=f"{name} ({mask.sum():,})",
            marker=dict(color=color, size=6, opacity=0.55),
            customdata=np.stack([d["link_label"], d["interval"], d["frequency"]], axis=1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} - %{customdata[2]:.0f} trains"
                          "<br>Capacity %{x:,.0f} - demand %{y:,.0f}<extra></extra>"))
    top = float(max(obs["offered"].max(), obs["demand"].max(), 1))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Demand = capacity",
                             line=dict(color=NEUTRAL, dash="dash", width=1.5), hoverinfo="skip"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Observed effective capacity (trains × C × ρ)")
    fig.update_yaxes(title="Passenger demand")
    return _style(fig, "Demand versus observed service capacity (one point per link-interval)", 460)


def before_after(result: pd.DataFrame) -> go.Figure:
    by_t = (result.groupby(["interval_order", "interval"], as_index=False)
            [["existing_frequency", "optimized_frequency", "demand_served", "unmet_demand"]].sum()
            .sort_values("interval_order"))
    x = by_t["interval"].str[:2] + ":" + by_t["interval"].str[2:4]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.55, 0.45],
                        subplot_titles=("Train allocations per 15 min", "Passengers served and unmet per 15 min"))
    fig.add_trace(go.Scatter(x=x, y=by_t["existing_frequency"], name="Observed allocations", mode="lines",
                             line=dict(color=BLUE, width=2, shape="hv")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=by_t["optimized_frequency"], name="Optimised allocations", mode="lines",
                             line=dict(color=ORANGE, width=2, shape="hv")), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=by_t["demand_served"], name="Served", marker_color=AQUA,
                         marker_line=dict(width=0)), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=by_t["unmet_demand"], name="Unmet", marker_color=CRITICAL,
                         marker_line=dict(width=0)), row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.15)
    fig.update_xaxes(nticks=24, row=2, col=1)
    fig.update_annotations(font_size=12, x=0, xanchor="left")
    return _style(fig, "Observed versus optimised service", 560)


def change_heatmap(result: pd.DataFrame, max_links: int = 40) -> go.Figure:
    links = result.groupby("link_label")["frequency_change"].apply(lambda s: s.abs().sum()).nlargest(max_links).index
    sub = result[result["link_label"].isin(links)]
    pivot = sub.pivot_table(index="link_label", columns="interval_order", values="frequency_change", aggfunc="sum")
    labels = sub.drop_duplicates("interval_order").set_index("interval_order")["interval"]
    pivot.columns = [f"{labels[c][:2]}:{labels[c][2:4]}" for c in pivot.columns]
    lim = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=DIVERGING,
                               zmin=-lim, zmax=lim, colorbar=dict(title="Trains"),
                               hovertemplate="%{y}<br>%{x}: %{z:+.0f} trains vs observed<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return _style(fig, f"Change in train allocations vs observed (links with the largest changes, max {max_links})",
                  max(380, 16 * len(pivot) + 120))


def network(edges: pd.DataFrame, metric: str, metric_label: str) -> go.Figure:
    """Schematic topology: node positions come from a force-directed layout, not geography."""
    g = nx.DiGraph()
    for r in edges.itertuples():
        g.add_edge(r.from_station, r.to_station, value=getattr(r, metric), label=r.link_label, line=r.line)
    pos = nx.spring_layout(g.to_undirected(), seed=7, k=1.5 / np.sqrt(max(len(g), 1)), iterations=200)
    values = np.array([d["value"] for _, _, d in g.edges(data=True)], dtype=float)
    vmax = values.max() if len(values) and values.max() > 0 else 1.0
    fig = go.Figure()
    bins = np.linspace(0, vmax, 6)
    colors = ["#b7d3f6", "#86b6ef", "#3987e5", "#256abf", "#104281"]
    for k in range(5):  # one trace per magnitude band keeps the figure light
        xs, ys = [], []
        for (u, v, d) in g.edges(data=True):
            if bins[k] <= d["value"] <= bins[k + 1] + 1e-9 and (k == 4 or d["value"] < bins[k + 1]):
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colors[k], width=1.5 + k),
                                     name=f"{bins[k]:,.0f}–{bins[k + 1]:,.0f}", hoverinfo="skip"))
    mid = [((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, f"{d['label']} [{d['line']}]<br>{metric_label}: {d['value']:,.0f}")
           for u, v, d in g.edges(data=True)]
    fig.add_trace(go.Scatter(x=[m[0] for m in mid], y=[m[1] for m in mid], mode="markers",
                             marker=dict(size=10, color="rgba(0,0,0,0)"), text=[m[2] for m in mid],
                             hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in g.nodes], y=[pos[n][1] for n in g.nodes], mode="markers",
                             marker=dict(size=8, color=NEUTRAL, line=dict(width=2, color="white")),
                             text=list(g.nodes), hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(hovermode="closest", legend_title_text=metric_label)
    return _style(fig, f"Schematic network topology — edges shaded by {metric_label.lower()} (not a geographic map)", 620)


def scenario_bars(table: pd.DataFrame, metric: str, label: str) -> go.Figure:
    ok = table["feasible"]
    fig = go.Figure(go.Bar(
        x=table["scenario"], y=table[metric].where(ok), marker_color=BLUE,
        text=[f"{v:,.1f}" if pd.notna(v) else "infeasible" for v in table[metric].where(ok)],
        textposition="outside", hovertemplate="%{x}<br>" + label + ": %{y:,.2f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, f"{label} by scenario", 380)
