
# Pay Equity Dashboard Demo

## Files & Directories
### Root Directory
| Name                         | Description
|------------------------------|----------------------------------------------
| `/assets`                    | Folder of images, stylesheets, and icons
| `/cache`                     | Git-ignored folder of cached data
| [`/pages`](#pages-directory) | Folder of modular page layouts and callbacks
| [`/utils`](#utils-directory) | Folder of helper functions and configurations
| `/venv`                      | Git-ignored virtual environment
| `.gitignore`                 | List of git-ignored files and directories
| `app.py`                     | Main .py script and app entrypoint
| `README.md`                  | *This* file
| `requirements.txt`           | Text file of all dependencies

### `/pages` Directory

| Name               | Title                   | Order Index |
|--------------------|-------------------------|-------------|
| `not_found_404.py` | *N/A*                   | -1          |
| `home.py`          | Home                    |  0          |
| `page_1.py`        | Data Generation         |  1          |
| `page_2.py`        | Variable Selection      |  2          |
| `page_3.py`        | Wage Gaps               |  3          |
| `page_4.py`        | Pairwise Comparisons    |  4          |
| `page_5.py`        | Interactions            |  5          |
| `page_6.py`        | Multivariate Regression |  6          |

This directory is used to organize and manage the different pages of the Dash app. It leverages Dash’s [page registry](https://dash.plotly.com/urls) feature to keep each page’s functionality separate and manageable. By following this structure, each page is modular and easier to maintain, with its content and layout handled dynamically.

Each page is defined using the `dash.register_page()` function with the following parameters:
- `__name__`: This refers to the name of the current Python module (file). It helps Dash identify the page.
- `path`: The relative URL path where the page will be accessible in the app (e.g., `/data-generation`).
- `title`: The title of the page, which will be displayed in the browser tab.
- `name`: A name for the page, used for internal reference.
- `order`: An optional index that determines the order in which the page appears in any page list or menu (our values range from -1 to 6).
- `layout`: This defines the layout of the page. For most pages, it’s an empty div, and the actual content is defined in the `page_layout()` function.
- `default`: A custom parameter I introduced to use a function (`page_layout()`) instead of a static layout variable. This function provides the content layout dynamically and is handled in `app.py`.

###### Example usage:
```python
dash.register_page(
    __name__,
    path='/example',
    title='Example Page',
    name='Example',
    order=1,
    layout=html.Div(),
    default=page_layout()
)
```

### `/utils` Directory
| Name             | Description
|------------------|-----------------------------------------
| `cache.py`       | Contains helper functions for storing/retrieving cached data
| `comparisons.py` | Contains helper functions for creating intermediate chart data and comparison plots
| `config.py`      | Contains configuration settings and global variables
| `generator.py`   | Contains helper functions for generating synthetic data


## Set Up Instructions

### Clone Repository
```console
$ git clone https://github.com/KF-camccaffrey/KF-Dash
 ```

### (Optional) Create Virtual Environment
```console
$ pip install virtualenv
$ virtualenv venv
```

Windows:
```console
$ venv\Scripts\activate
 ```

macOS/Linux:
```console
$ source venv/bin/activate
```

### Install Dependencies
```console
$ pip install -r requirements.txt
```

### Run Dash App
```console
$ python app.py
```

Then, go to [127.0.0.1:8050](http://127.0.0.1:8050/) in browser.

### (Optional) Deactivate Virtual Environment
```console
$ deactivate
```
