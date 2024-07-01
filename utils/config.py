
#>----------------->     Theme Options     <-----------------<#
BLUE = "#007da4"
PINK = "#920a7a"



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
