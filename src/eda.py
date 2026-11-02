"""EDA helper functions producing Plotly figures."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def churn_distribution(df):
    counts = df["churned"].value_counts().sort_index()
    labels = ["Not Churned", "Churned"]
    fig = px.bar(
        x=labels, y=counts.values,
        text=[f"{v} ({v/len(df)*100:.1f}%)" for v in counts.values],
        color=labels,
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        title="Churn Distribution",
    )
    fig.update_layout(showlegend=False, xaxis_title="Status", yaxis_title="Count")
    return fig


def age_histogram(df):
    fig = px.histogram(df, x="age", nbins=30, color="churned",
                       color_discrete_sequence=["#2ecc71", "#e74c3c"],
                       title="Age Distribution (by churn)")
    fig.update_layout(barmode="overlay")
    fig.update_traces(opacity=0.7)
    return fig


def monthly_fee_histogram(df):
    fig = px.histogram(df, x="monthly_fee", nbins=20, color="churned",
                       color_discrete_sequence=["#2ecc71", "#e74c3c"],
                       title="Monthly Fee Distribution (derived from subscription tier)")
    fig.update_layout(barmode="overlay")
    fig.update_traces(opacity=0.7)
    return fig


def watch_time_histogram(df):
    fig = px.histogram(df, x="avg_watch_time_minutes_per_week", nbins=40,
                       color="churned",
                       color_discrete_sequence=["#2ecc71", "#e74c3c"],
                       title="Weekly Watch Time Distribution")
    fig.update_layout(barmode="overlay")
    fig.update_traces(opacity=0.7)
    return fig


def box_monthly_fee_vs_churn(df):
    df = df.copy()
    df["churn_label"] = df["churned"].map({0: "Not Churned", 1: "Churned"})
    return px.box(df, x="churn_label", y="monthly_fee", color="churn_label",
                  color_discrete_sequence=["#2ecc71", "#e74c3c"],
                  title="Monthly Fee vs Churn")


def box_watch_time_vs_churn(df):
    df = df.copy()
    df["churn_label"] = df["churned"].map({0: "Not Churned", 1: "Churned"})
    return px.box(df, x="churn_label", y="avg_watch_time_minutes_per_week",
                  color="churn_label",
                  color_discrete_sequence=["#2ecc71", "#e74c3c"],
                  title="Weekly Watch Time vs Churn")


def subscription_vs_churn(df):
    g = df.groupby(["subscription_type", "churned"]).size().reset_index(name="count")
    g["churn_label"] = g["churned"].map({0: "Not Churned", 1: "Churned"})
    return px.bar(g, x="subscription_type", y="count", color="churn_label",
                  barmode="group",
                  color_discrete_sequence=["#2ecc71", "#e74c3c"],
                  title="Subscription Type vs Churn")


def correlation_heatmap(df):
    num = df.select_dtypes(include=np.number)
    corr = num.corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Heatmap (Numerical Features)")
    fig.update_layout(height=700)
    return fig


def outlier_summary(df):
    """IQR-based outlier counts for each numerical column."""
    num = df.select_dtypes(include=np.number)
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "feature": c, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": low, "upper_bound": high, "n_outliers": n_out,
        })
    return pd.DataFrame(rows)


def outlier_boxplots(df):
    num = df.select_dtypes(include=np.number)
    fig = go.Figure()
    for c in num.columns:
        fig.add_trace(go.Box(y=num[c], name=c, boxpoints="outliers"))
    fig.update_layout(title="Outlier Boxplots (all numerical features)",
                      showlegend=False, height=500)
    return fig


def univariate_overview(df):
    """Return dict {col: figure} for a quick univariate view."""
    figs = {}
    num = df.select_dtypes(include=np.number).columns.tolist()
    for c in num:
        figs[c] = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")
    return figs