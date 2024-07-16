
import numpy as np
import pandas as pd
from utils.config import GRAPHCONFIG, BLUE, PINK
import plotly.express as px
import plotly.figure_factory as ff
import os

from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn


import plotly.graph_objects as go
from plotly.subplots import make_subplots

data_folder = 'data'
file_path = os.path.join(data_folder, 'data.csv')
df = pd.read_csv(file_path)

# histogram
fig1 = px.histogram(df, x='pay', color='gender', labels={'pay': 'Pay', 'gender': 'Gender'},
                   barmode="overlay", hover_data={"gender": False},
                   color_discrete_map={'Male': BLUE, 'Female': PINK}, opacity=0.4,
                   template="simple_white")


# boxplots
education_levels = ["Secondary", "Bachelor", "Master", "Doctorate"]
fig2 = px.box(df, x='education', y='pay', color='gender',
             category_orders={'education': education_levels},
             title='Distribution of Pay by Education Level and Gender',
             labels={'Pay': 'Pay ($)', 'Education': 'Education Level', 'Gender': 'Gender'},
             color_discrete_map={'Male': BLUE, 'Female': PINK, "Other": "green"},
             points=False,  # Show all points
             template="simple_white"
)

# Update layout for better visualization
fig2.update_layout(
    boxmode='group',  # Group boxes together
    boxgap=0.2,  # Gap between boxes
    boxgroupgap=0.05,  # Gap between different groups of boxes
    yaxis=dict(
        title='Pay ($)'
    ),
    xaxis=dict(
        title='Education Level'
    )
)



def plot_bar_and_donut(data, categories, values, title1, title2):
    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2), specs=[[{'type':'bar'}, {'type':'pie'}]])

    # Bar chart (subplot 1)
    fig.add_trace(
        go.Bar(x=categories, y=values, marker_color=['#f53333', 'gray']),
        row=1, col=1
    )

    # Add annotations to the bar chart
    for i, v in enumerate(values):
        fig.add_annotation(
            x=categories[i], y=v,
            text=f'{v:.1f}%', showarrow=False,
            font=dict(color='black', size=12),
            align='center',
            yshift=(10 if v > 0 else -10)
        )

    # Pie chart (subplot 2)
    gender_counts = data['gender'].value_counts()
    fig.add_trace(
        go.Pie(labels=gender_counts.index, values=gender_counts.values,
               marker=dict(colors=['#b5170e', '#faaf0c', '#f4f720'], line=dict(color='#FFFFFF', width=2)),
               hole=0.4,
               textinfo='percent',
               hoverinfo='label+percent'),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text=title1 + ' and ' + title2,
        showlegend=False,
        height=500,
        width=1000,
        annotations=[dict(text=title1, x=0.22, y=1.1, font_size=16, showarrow=False),
                     dict(text=title2, x=0.77, y=1.1, font_size=16, showarrow=False)]
    )

    # Show the plot
    return fig



# Calculate median compensation and percentage differences
median_male_comp = df[df['gender'] == 'Male']['pay'].median()
median_female_comp = df[df['gender'] == 'Female']['pay'].median()
median_other_comp = df[df['gender'] == 'Other']['pay'].median()
percent_diff_female = ((median_female_comp - median_male_comp) / median_male_comp) * 100
percent_diff_other = ((median_other_comp - median_male_comp) / median_male_comp) * 100

# Plot bar and donut charts
fig3 = plot_bar_and_donut(df, ['Female', 'Other'], [percent_diff_female, percent_diff_other], 'Gender Pay Gap Compared to Males', 'Headcount by Gender')


comparisons = {
    'gender' : {
        'base': 'male',
        'levels': []
    }
}

def holm_bonferonni(p_values):
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    n = len(p_values)

    adjusted_p_values = np.minimum.accumulate(sorted_p_values * (n - np.arange(n)))
    return adjusted_p_values[np.argsort(sorted_indices)]

def effect_sizes(df, y, comparisons):
    for col, params in comparisons.items():
        all_lvls = list(df[col].unique())
        levels = params.get('levels', all_lvls[:])
        base = params.get('base', levels[0])

        if levels is None or len(levels) < 2:
            raise ValueError(f"Column '{col}' contains too few levels: {levels}")

        if base in levels:
            levels.remove(base)

        missing = [lvl for lvl in [base]+levels if lvl not in all_lvls]
        if missing:
            raise ValueError(f"Categories {missing} are missing in the column '{col}'. Valid categories: {all_lvls}")

        params['base'] = base
        params['levels'] = levels

        grouped = df.groupby(col)
        base_vals = grouped[y].get_group(base)
        base_mean = base_vals.mean()
        base_median = base_vals.median()

        means = []
        mean_diff = []
        mean_perc = []

        medians = []
        median_diff = []
        median_perc = []

        samples = [base_vals]

        for lvl in levels:
            lvl_vals = grouped[y].get_group(lvl)
            samples.append(lvl_vals)

            lvl_mean = lvl_vals.mean()
            lvl_mean_diff = lvl_mean - base_mean
            lvl_mean_perc = 100 * lvl_mean_diff / base_mean


            lvl_median = lvl_vals.median()
            lvl_median_diff = lvl_median - base_median
            lvl_median_perc = 100 * lvl_median_diff / base_median

            means.append(lvl_mean)
            mean_diff.append(lvl_mean_diff)
            mean_perc.append(lvl_mean_perc)

            medians.append(lvl_median)
            median_diff.append(lvl_median_diff)
            median_perc.append(lvl_median_perc)

        kruskal_stat, kruskal_p = kruskal(*samples)
        dunn_p = posthoc_dunn(samples).values[0, 1:]
        holm_p = holm_bonferonni(dunn_p)


        params['means'] = means
        params['mean_diff'] = mean_diff
        params['mean_perc'] = mean_perc
        params['medians'] = medians
        params['median_diff'] = median_diff
        params['median_perc'] = median_perc
        params['kruskal_stat'] = kruskal_stat
        params['kruskal_p'] = kruskal_p
        params['dunn_p'] = dunn_p
        params['holm_p'] = holm_p
        comparisons[col] = params
    return comparisons


def effect_bars(comparisons):
    figures = []

    for col, params in comparisons.items():
        base = params['base']
        levels = params['levels']
        holm_p = params['holm_p']
        median_diff = params['median_diff']

        annotations = []

        for i, lvl in enumerate(levels):
            x_values = [median_diff[i]]
            y_labels = [f"{col.capitalize()} - {lvl}"]

            color = 'blue'
            if holm_p[i] < 0.05:
                color = 'red'

            index = list(comparisons.keys()).index(col)
            offset = -0.2 if index == 0 else 0.2

            data = go.Bar(
                x=x_values,
                y=y_labels,
                name=f"{col.capitalize()} - {lvl}",
                orientation='h',
                marker=dict(color=color),
                text=[f"Holm p-value: {holm_p[i]}"],
                hoverinfo='text+x',
                offset=offset,
            )

            figures.append(data)

            if i == 0:
                annotations.append(dict(
                    xref='paper',
                    yref='y',
                    x=0.5,
                    y=(index + 0.5) / len(comparisons),  # Position annotation
                    xanchor='center',
                    yanchor='middle',
                    text=f"{col.capitalize()} Group",  # Group label
                    showarrow=False,
                    font=dict(size=16, color='black'),
                    align='center',
                    bordercolor='black',
                    borderwidth=2,
                    borderpad=4,
                    bgcolor='lightgrey',
                    opacity=0.8
                ))

    layout = go.Layout(
        title="Median Differences Across Categories",
        xaxis=dict(title="Mean Differences"),
        yaxis=dict(title='', tickvals=np.arange(len(comparisons)), ticktext=list(comparisons.keys())),
        barmode='group',
        annotations=annotations,
        template='simple_white',
        showlegend=False,
    )

    fig = go.Figure(data=figures, layout=layout)
    #return {'data': figures, 'layout': layout}

    return fig


def generate_grouped_bar_charts(data):
    max_bars = max(len(cat_data['levels']) for cat_data in data.values())

    subplot_heights = [0.25] * len(data)
    for i, (category, cat_data) in enumerate(data.items(), start=1):
        if len(cat_data['levels']) == max_bars:
            subplot_heights[i-1] = 0.5

    fig = make_subplots(rows=len(data), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=None, row_heights=subplot_heights)

    for i, (category, cat_data) in enumerate(data.items(), start=1):
        base_level = cat_data['base']
        levels = cat_data['levels']
        holm_p_values = cat_data['holm_p']

        bar_data = []
        for j, level in enumerate(levels):
            if level != base_level:
                x_values = [cat_data['mean_diff'][j]]  # Change metric as needed
                color = 'blue'
                if holm_p_values[j] < 0.05:  # Example significance level
                    color = 'red'

                bar_width = 2.0 / max_bars

                bar = go.Bar(
                    x=x_values,
                    y=[f"{category.capitalize()} - {level}"],
                    name=f"{category.capitalize()} - {level}",
                    orientation='h',
                    marker=dict(color=color),
                    text=[f'Holm p-value: {holm_p_values[j]}'],
                    hoverinfo='text+x',
                    width=bar_width #0.5  # Equal thickness for all bars
                )
                bar_data.append(bar)

        for bar in bar_data:
            fig.append_trace(bar, row=i, col=1)

        # Hide x-axis for all but the lowest subplot
        if i != len(data):
            fig.update_xaxes(showline=False, tickmode="array", ticks='', showticklabels=False, row=i, col=1)

        # Add y-axis label for each subplot
        fig.update_yaxes(title_text=f'{category.capitalize()} Category', row=i, col=1)

    fig.update_layout(
        title='Mean Differences Across Categories',
        height=500 * len(data),  # Adjust height based on number of categories
        showlegend=False,  # Remove legend
        template='simple_white'  # Apply template
    )

    return fig


y = np.random.normal([5, 6, 10, 19, 20], [2, 3, 5, 3, 3], size=[20, 5]).flatten()
x = np.resize(["Male", "Female", "Enby", "Fluid", "Other"], 100)
z = np.resize(["White", "Black", "Asian"], 100)


df = pd.DataFrame({'gender':x, 'pay':y, 'race':z})

comparisons = {
    'gender': {'levels': ['Male', "Female", "Enby", "Fluid", "Other"], 'base': 'Other'},
    'race': {},
}

result = effect_sizes(df, 'pay', comparisons)
print(result)


fig4 = effect_bars(result)

fig5 = generate_grouped_bar_charts(result)

fig5.show()



"""
    df_filtered = df[[col for col in df.columns if col in comparisons.keys()]]

    fig = px.bar(df, x='category', y='value', color='category', barmode='group',
                 labels={'category': 'Categories', 'value': 'Values'},
                 title='Dynamic Bar Chart')


        # Create plotly figure object
    fig = go.Bar(x=categories, y=values)

    # Add PDF curve trace
    fig.add_trace(go.Scatter(
        x=x,
        y=y_m,
        mode='lines',
        line=dict(width=3, color=BLUE),
        fill='tozeroy',
        fillcolor=alpha(BLUE, 0.2),
        showlegend=False,
    ))
"""
