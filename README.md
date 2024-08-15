
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
| Name               | Description
|--------------------|-----------------------------------------
| `home.py`          | Layout for app homepage
| `not_found_404.py` | Layout for 404 error page
| `page_1.py`        | Layout for *Data Generation*
| `page_2.py`        | Layout for *Variable Selection*
| `page_3.py`        | Layout for *Wage Gaps*
| `page_4.py`        | Layout for *Pairwise Comparisons*
| `page_5.py`        | Layout for *Interactions*
| `page_6.py`        | Layout for *Multivariate Regression*

### `/utils` Directory
| Name             | Description
|------------------|-----------------------------------------
| `cache.py`       | Contains helper functions for storing/retrieving cached data
| `comparisons.py` | Contains helper functions for creating intermediate chart data
| `config.py`      | Contains configuration settings and global variables
| `generator.py`   | Contains helper functions for generating synthetic data
| `pairwise.py`    | Contains helper functions for conducting pairwise comparisons


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

Then, go to [127.0.0.1:8050/](http://127.0.0.1:8050/) in browser.

### (Optional) Deactivate Virtual Environment
```console
$ deactivate
```
