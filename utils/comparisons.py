
"""
File Name: comparisons.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains functions for calculating intermediate statistics as well as some plotting functions
"""

import numpy as np
import pandas as pd
from utils.config import BLUE, PINK, YELLOW, RED, GRAY, alpha, FORESTGREEN, EMPTYFIG
import plotly.graph_objects as go
from pprint import pprint
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# the function that calculates the intermediate statistics
def create_comparisons(df, pay, gender, comparisons):
    new_categorical = {}

    response = comparisons.get("response", None)
    quantitative = comparisons.get("quantitative", None)
    categorical = comparisons.get("categorical", None)

    male = df[df[gender] == "Male"]
    female = df[df[gender] == "Female"]
    other = df[df[gender] == "Other"]

    for col, params in categorical.items():
        all_lvls = list(df[col].unique())
        lvls = params.get("levels", all_lvls[:])
        base_lvl = params.get("base", lvls[0])

        if lvls is None or len(lvls) < 2:
            raise ValueError(f"Column '{col}' contains too few levels: {lvls}")

        missing = [lvl for lvl in lvls if lvl not in all_lvls]
        if missing:
            raise ValueError(f"Categories {missing} are missing in the column '{col}'. Valid categories: {all_lvls}")

        n_lvls = len(lvls)
        lvls = np.array(["Overall"] + lvls)
        base_idx = np.where(lvls == base_lvl)[0][0]
        dummies_idx = np.where(lvls != base_lvl)[0][1:]
        overall_idx = np.where(lvls == "Overall")[0][0]

        grouped = df.groupby(col)
        grouped_male = male.groupby(col)
        grouped_female = female.groupby(col)
        grouped_other = other.groupby(col)

        n = np.repeat(len(df.index), n_lvls+1)
        n_male = np.repeat(len(male.index), n_lvls+1)
        n_female = np.repeat(len(female.index), n_lvls+1)
        n_other = np.repeat(len(other.index), n_lvls+1)

        mean = np.repeat(df[pay].mean(), n_lvls+1)
        mean_male = np.repeat(male[pay].mean(), n_lvls+1)
        mean_female = np.repeat(female[pay].mean(), n_lvls+1)
        mean_other = np.repeat(other[pay].mean(), n_lvls+1)
        mean_gap = abs(mean_male-mean_female)
        mean_gap_perc = mean_female/mean_male
        mean_comp = np.repeat(np.nan, n_lvls+1)
        mean_comp_perc = np.repeat(np.nan, n_lvls+1)

        median = np.repeat(df[pay].median(), n_lvls+1)
        median_male = np.repeat(male[pay].median(), n_lvls+1)
        median_female = np.repeat(female[pay].median(), n_lvls+1)
        median_other = np.repeat(other[pay].median(), n_lvls+1)
        median_gap = abs(median_male-median_female)
        median_gap_perc = median_female/median_male
        median_comp = np.repeat(np.nan, n_lvls+1)
        median_comp_perc = np.repeat(np.nan, n_lvls+1)

        dunn_p = np.repeat(np.nan, n_lvls+1)
        holm_p = np.repeat(np.nan, n_lvls+1)

        # Get values of base level, calculate mean and median
        base = lvls[base_idx]
        vals_base = grouped[pay].get_group(base)
        mean_base = vals_base.mean()
        median_base = vals_base.median()

        # Initialize sample for kruskal-wallis test
        sample = []
        mask = np.ones(n_lvls+1, dtype=bool)
        mask[0] = False

        # Each Level 1, ..., N
        for i in range(1, n_lvls+1):
            # Get level string
            lvl_i = lvls[i]

            # Get values
            try:
                vals_i = grouped[pay].get_group(lvl_i)
            except KeyError:
                vals_i = pd.Series(name=pay)
            try:
                vals_male_i = grouped_male[pay].get_group(lvl_i)
            except KeyError:
                vals_male_i = pd.Series(name=pay)
            try:
                vals_female_i = grouped_female[pay].get_group(lvl_i)
            except KeyError:
                vals_female_i = pd.Series(name=pay)
            try:
                vals_other_i = grouped_other[pay].get_group(lvl_i)
            except KeyError:
                vals_other_i = pd.Series(name=pay)

            print(f"Level: {lvl_i} ({len(vals_i)})")
            if len(vals_i) == 0:
                mask[i] = False
            else:
                sample.append(vals_i)

            # Calculate counts
            n[i] = len(vals_i)
            n_male[i] = len(vals_male_i)
            n_female[i] = len(vals_female_i)
            n_other[i] = len(vals_other_i)

            # Calculate means
            mean[i] = vals_i.mean()
            mean_male[i] = vals_male_i.mean()
            mean_female[i] = vals_female_i.mean()
            mean_other[i] = vals_other_i.mean()
            mean_gap[i] = abs(mean_male[i] - mean_female[i])
            mean_gap_perc[i] = mean_female[i] / mean_male[i]
            mean_comp[i] = mean[i] - mean_base
            mean_comp_perc[i] = mean_comp[i] / mean_base

            # Calculate medians
            median[i] = vals_i.median()
            median_male[i] = vals_male_i.median()
            median_female[i] = vals_female_i.median()
            median_other[i] = vals_other_i.median()
            median_gap[i] = abs(median_male[i] - median_female[i])
            median_gap_perc[i] = median_female[i] / median_male[i]
            median_comp[i] = median[i] - median_base
            median_comp_perc[i] = median_comp[i] / median_base

        # Conduct kruskal-wallis test
        kruskal_stat, kruskal_p = kruskal(*sample)
        dunn_p = np.repeat(np.nan, n_lvls+1)
        dunn_p[mask] = posthoc_dunn(sample).values[0, :]

        metrics = dict(
             n_levels=n_lvls,
             levels=lvls,
             base=base_idx,
             dummies=dummies_idx,
             overall=overall_idx,

             n=n,
             n_male=n_male,
             n_female=n_female,
             n_other=n_other,

             mean=mean,
             mean_male=mean_male,
             mean_female=mean_female,
             mean_other=mean_other,
             mean_gap=mean_gap,
             mean_gap_perc=mean_gap_perc,
             mean_comp=mean_comp,
             mean_comp_perc=mean_comp_perc,

             median=median,
             median_male=median_male,
             median_female=median_female,
             median_other=median_other,
             median_gap=median_gap,
             median_gap_perc=median_gap_perc,
             median_comp=median_comp,
             median_comp_perc=median_comp_perc,

             kruskal_stat=kruskal_stat,
             kruskal_p=kruskal_p,
             dunn_p=dunn_p,
             holm_p=holm_p,
        )
        new_categorical[col] = metrics
    return {"response": response, "quantitative": quantitative, "categorical": new_categorical}

# helper function for calculating adjusted p-values
def holm_bonferonni(p_values):
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    n = len(p_values)

    adjusted_p_values = np.minimum.accumulate(sorted_p_values * (n - np.arange(n)))
    return adjusted_p_values[np.argsort(sorted_indices)]


# returns the pairwise comparison bar chart
def effect_bars(comparisons, category="all", method="median_comp"):
    comparisons = comparisons["categorical"]

    if category == "none":
        return EMPTYFIG
    elif category != "all":
        comparisons = {category: comparisons[category]}

    unit = "$" if method == "median_comp" else "%"
    x_title = f"Differences in Median Pay ({unit})"

    ncol = len(comparisons)
    max_bars = max(len(metrics['dummies']) for metrics in comparisons.values())

    subplot_heights = [0.25] * ncol
    for i, (col, metrics) in enumerate(comparisons.items(), start=1):
        if len(metrics['dummies']) == max_bars:
            subplot_heights[i-1] = 0.5

    fig = make_subplots(rows=ncol, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=None, row_heights=subplot_heights)

    for i, (col, metrics) in enumerate(comparisons.items(), start=1):
        lvls = metrics["levels"]
        base_idx = metrics["base"]
        dummies_idx = metrics["dummies"]
        base = lvls[base_idx]
        dunn_p = metrics['dunn_p']

        bar_data = []
        for j in np.flip(dummies_idx):
            lvl = lvls[j]
            x_values = [metrics[method][j]]  # Change metric as needed

            bar_width = 1.0 / max_bars

            bar = go.Bar(
                x=x_values,
                y=[f"{lvl} vs. {base}"],
                name=f"{lvl} vs. {base}",
                orientation='h',
                marker=dict(color=p_color(dunn_p[j]), line_color="black", line_width=1),
                #text=[f'Dunn p-value: {p_round(dunn_p[j], 3)}'],
                hoverinfo='text+x',
                width=bar_width # Equal thickness for all bars
            )
            bar_data.append(bar)

        for bar in bar_data:
            fig.append_trace(bar, row=i, col=1)

        # Hide x-axis for all but the lowest subplot
        if i != ncol:
            fig.update_xaxes(showline=False, tickmode="array", ticks='', showticklabels=False, row=i, col=1)
        else:
            fig.update_xaxes(title_text=x_title, row=ncol, col=1)

        # Add y-axis label for each subplot
        fig.update_yaxes(title_text=f'{col.capitalize()} Comparisons', row=i, col=1)

    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=0, y1=1,
        line=dict(color="black", width=2, dash="dash"),
        xref="x",
        yref="paper",
    )

    fig.update_layout(
        title=f"{x_title} Across Categories",
        height=500 * ncol,  # Adjust height based on number of categories
        showlegend=False,  # Remove legend
        template='simple_white'  # Apply template
    )

    return fig

# helper function for rounding p-values
def p_round(p, decimals):
    threshold = 10 ** -decimals
    if p >= threshold:
        return str(round(p, decimals))
    else:
        return "{:.{}e}".format(p, decimals)

# helper function for deciding color of p-values
def p_color(p):
    if p > 0.1:
        return GRAY
    elif p > 0.05:
        return YELLOW
    else:
        return RED



# returns the dumbbell chart on page 3
def dumbbell_chart(comparisons, selected_category, method, mysort):
    categorical = comparisons["categorical"]

    height = 750
    mt, mb, ml, mr = (10, 10, 10, 10)

    metrics = categorical[selected_category]
    n_lvls = metrics["n_levels"]
    lvls = metrics["levels"]

    method_male = metrics[f"{method}_male"]
    method_female = metrics[f"{method}_female"]

    method_gap = metrics[f"{method}_gap"]
    n = metrics["n"]
    n_male = metrics["n_male"]
    n_female = metrics["n_female"]
    nmax = max(np.concatenate((n_male, n_female)))
    inds = dumbsort(lvls, mysort, method_gap, n)


    px = height - mt - mb
    size = int(px / ((n_lvls + 1) * 1.25))

    op = 0.75
    cent = 5

    traces = []
    skips = []
    for i in range(n_lvls+1):
        if np.isnan(method_male[i]) and np.isnan(method_female[i]):
            traces.append(None)
            skips.append(i)
            continue
        trace = go.Scatter(
            x=[method_male[i], method_female[i]] * 2,
            y=[lvls[i]] * 4,
            line={"color": GRAY, "width": 3},
            marker={
                "color": [alpha(BLUE, 1), alpha(PINK, 1), "white", "white"],
                "size":[dumbsize(size, n_male[i], nmax), dumbsize(size, n_female[i], nmax), cent, cent],
                "opacity": [op, op, 1.0, 1.0],
                "line": {"width": 0},
            },
            mode="lines+markers",
            name=lvls[i],
            hoverinfo="y"
        )

        traces.append(trace)

    traces = [traces[0]] + [traces[i] for i in inds if i != 0 and i not in skips]

    layout = go.Layout(
        #title=f"Difference in {method.capitalize()} Pay between Men and Women across {selected_category.capitalize()} Levels",
        xaxis={"title": "Pay",
               "showgrid": True},
        yaxis={"tickfont": {"size": 16, "family": "Gotham"}},
        template='simple_white',
        showlegend=False,
        hovermode="closest",
        autosize=True,
        #height=height,
        margin=dict(l=ml, r=mr, t=mt, b=mb),
    )
    return go.Figure(data=traces, layout=layout)

# helper function for sorting dumbbells
def dumbsort(levels, mysort, diffs, n):
    print(mysort)
    if mysort == "alpha":
        clean = [l if l is not None else "zzzz" for l in levels]
        i = np.argsort(clean)[::-1]
    elif mysort == "diff":
        i = np.argsort(abs(diffs))
    elif mysort == "n":
        print(levels)
        print(n)
        i = np.argsort(n)
    else:
        i = np.arange(len(levels))[::-1]
    return i

# helper function for determining size of each dumbbell datapoint
def dumbsize(pmax, n, nmax):
    pmin = 10
    nmax = n if n > nmax else nmax
    pmax = pmin if pmax < pmin else pmax
    frac = n / nmax
    vals = np.arange(pmin, pmax+1)
    i = int(round(frac * (pmax-pmin), 0))
    return vals[i]


# returns pie chart on page 3
def pie_chart(comparisons, selected_category, lvl):
    categorical = comparisons["categorical"]

    metrics = categorical[selected_category]
    lvls = metrics["levels"]
    i = np.where(lvls == lvl)[0][0]

    labels = ["Male", "Female", "Other"]
    keys = ["n_male", "n_female", "n_other"]
    counts = np.array([metrics[key][i] for key in keys])
    total = metrics["n"][i]
    colors = [BLUE, PINK, FORESTGREEN]

    percs = np.round(100 * counts/total, 1)

    # Define the pie chart
    fig = go.Figure()

    # Add the pie chart trace
    fig.add_trace(go.Pie(
        labels=labels,
        values=counts,
        domain=dict(x=[0, 1], y=[0, 1]),
        hole=0.8,  # This creates the donut shape
        textinfo='none',
        hoverinfo='value',
        marker=dict(colors=colors, line=dict(color="#DEE2E6", width=2)),
        textfont=dict(family="Gotham", size=13),
        hovertemplate= [f"{p}%（" + "{:20,}".format(c).strip() + '）<extra></extra>' for c, p in zip(counts, percs)], #'%{value:20,}',  # Custom hover text format
        hoverlabel_font_size=18,
        hoverlabel_font_family="Gotham",
    ))

    fig.update_layout(
        annotations=[dict(
            text="{:20,}".format(total).strip(),
            x=0.5,
            y=0.5,
            font_size=24,
            font_family="Gotham",
            showarrow=False,
            xref="paper",
            yref="paper",
        )],
        showlegend=False,
        height=150,
        width=150,
        margin=dict(l=2, r=2, t=2, b=2),
        #paper_bgcolor="gray",
        autosize=False,
    )

    return fig

# returns bar chart on page 3 (this function is no longer used)
def metric_bars(comparisons, selected_category, lvl, method):
    categorical = comparisons["categorical"]

    metrics = categorical[selected_category]
    lvls = metrics["levels"]
    i = np.where(lvls == lvl)[0][0]

    labels = np.flip(["Male", "Female", "Other"])
    vals = np.nan_to_num([metrics[f"{method}_{labels[j].lower()}"][i] for j in range(3)])
    colors = np.flip([BLUE, PINK, FORESTGREEN])


    fig = go.Figure()


    fig.add_trace(go.Bar(
        y=labels,
        x=vals,
        #text=[f" {format_currency(vals[j])}" for j in range(3)],
        #textposition='inside',
        #textfont=dict(family="Gotham", size=20),
        orientation='h',
        marker=dict(color=colors, line=dict(color="#DEE2E6", width=2))
    ))

    #fig.update_traces(texttemplate='%{text}', textposition='inside', insidetextanchor='start')

    for label, val in zip(labels, vals):
        fig.add_annotation(
            x=max(vals) * 0.5,  # Adjust the position as needed
            y=label,
            text=format_currency(val),
            showarrow=False,
            xanchor='right',  # Right-align the text
            yanchor='middle',
            font_family="Gotham",
            font_size=20,
            font_color="white",
            #font_variant_numeric="tabular-nums",
            #textfont=dict(family="Gotham", size=8, color="white"),
    )

    fig.update_layout(
        xaxis=dict(
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title=f"{method.capitalize()} Pay",
            title_font_size=22,
            title_font_family="Gotham",
            automargin=True,
            tickmode='array',
            showticklabels=False,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=150,  # Adjust height to match text box space
        width=350,
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.05,
    )

    return fig

# helper function for formatting currency
def format_currency(value):
    if value >= 1_000_000:
        return f"$ {value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"$ {value / 1_000:.1f}k"
    result ="$ " + f"{value:20,.2f}".strip()
    return result
