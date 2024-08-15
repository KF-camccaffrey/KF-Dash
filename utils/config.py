
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np

#>----------------->     Theme Options     <-----------------<#
BLUE = "#007da4"
PINK = "#920a7a"
RED = "#971310"
YELLOW = "#ddb307"
GRAY = "#919191"
GREEN ="#08a384"
FORESTGREEN = "#00634f"

def alpha(color, a):
    if isinstance(color, str):
        return help_alpha(color, a)

    elif isinstance(color, (list, np.ndarray)):
        if isinstance(color, list):
            color = np.array(color)

        result = np.empty(color.shape, dtype=object)
        for i, c in enumerate(color):
            result[i] = help_alpha(c, a)
        return result

    else:
        raise TypeError("Input should be a string, list, or NumPy array.")

def help_alpha(color, a):
    if color.startswith("#"):
        r,g,b = tuple(int(color[i:i+2],16) for i in (1,3,5))
    elif color.startswith("rgb"):
        rgb = color.split("(")[1].split(")")[0]
        r,g,b = map(int, rgb.split(","))
    else:
        raise ValueError("Unsupported color format. Use hex (#RRGGBB) or RGB (rgb(r, g, b)) format.")
    return f"rgba({r}, {g}, {b}, {a})"


#>----------------->     Graph Config      <-----------------<#

# source: https://plotly.com/python/configuration-options/

scrollZoom = False     # scroll wheel zooms in and out
responsive = True      # adjust height and width to size of window
staticPlot = False     # static charts are not interactive
displayModeBar = True  # make modebar visible
displaylogo = False    # show plotly logo

toImageButtonOptions = {
    'format': 'png',             # one of png, svg, jpeg, webp
    'filename': 'plotly_image',  # name of downloaded file
    'height': None,              # default = 700 px, None adapts to rendered size
    'width': None,               # default = 450 px, None adpapts to rendered size
    'scale': 1                   # multiply size of image
}

# modebar buttons to remove
typeHighLevel = ['zoom', 'pan', 'select', 'lasso', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
type2D = ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
type2DShapeDrawing = ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
type3D = ['zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d']
typeCartesian = ['hoverClosestCartesian', 'hoverCompareCartesian']
typeGeo = ['zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo']
typeOther = ['hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toImage', 'sendDataToCloud', 'toggleSpikelines', 'resetViewMapbox']

modeBarButtonsToRemove = typeHighLevel + typeOther

doubleClickDelay = 300 # zoom max delay (ms)

GRAPHCONFIG = {
    'scrollZoom': scrollZoom,
    'responsive': responsive,
    'staticPlot': staticPlot,
    'displayModeBar': displayModeBar,
    'displaylogo': displaylogo,
    'toImageButtonOptions': toImageButtonOptions,
    'modeBarButtonsToRemove': modeBarButtonsToRemove,
    'doubleClickDelay': doubleClickDelay
}

#>----------------->     Hover Labels      <-----------------<#

HOVERLABEL = {
    'bordercolor': 'white',
    'font_size': 13,
    'font_family': "Geist Mono",
}



#>----------------->     Styled Print Statements      <-----------------<#

def styled(text, color, bold=False):
    weight = "22" if not bold else "1"
    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    try:
        cc = colors.index(color) + 30
    except ValueError:
        cc = 0
    return f"\033[{weight};{cc}m{text}\033[0m"


#>----------------->      Comparisons      <-----------------<#
BASICCOMPS = {"gender": {}, "race": {}, "department": {}, "level": {}, "education": {}}


EMPTYFIG = {
    'data': [],
    'layout': go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=[go.layout.Annotation(
                text="No Data Available",
                x=0.5, y=0.5,
                font=dict(size=20, color=GRAY, family="Gotham"),
                showarrow=False,
                xref="paper", yref="paper",
                align="center"
        )],
        margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins
    )
}

AXISBLANK = dict(
    autorange=True,
    showgrid=False,
    ticks='',
    showticklabels=False,
    zeroline=False,
    showline=False
)

MARGINBLANK = dict(
    l=0, r=0, t=0, b=0
)
