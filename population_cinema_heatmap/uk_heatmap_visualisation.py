"""
PYTHON 3.8 (64 BIT)

pip install GDAL-3.3.3-cp38-cp38-win_amd64.whl (https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
pip install rasterio
pip install pgeocode

Population heatmap:
https://towardsdatascience.com/creating-beautiful-population-density-maps-with-python-fcdd84035e06

Alternative:
https://medium.com/@patohara60/interactive-mapping-in-python-with-uk-census-data-6e571c60ff4

Cinema post-code plot - best option so far:
https://github.com/bkontonis/Plot-PostCodes-on-Interactive-Map-Using-Python

"""

import rasterio
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
import matplotlib.colors as colors

import pgeocode
import plotly.express as px
import pandas as pd


def prepare_map_colour_scheme():
    # for get_cmap, see this for colour schemes- https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # we create 10 colour buckets here
    our_cmap = cm.get_cmap('YlGnBu', 10)
    newcolors = our_cmap(np.linspace(0, 1, 10))
    background_colour = np.array([0.9882352941176471, 0.9647058823529412, 0.9607843137254902, 1.0])
    newcolors = np.vstack((background_colour, newcolors))

    # converts the numbers into colours
    newcmap = ListedColormap(newcolors)
    # assigning different levels of population density to a colour
    bounds = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 10, 20]
    norm = colors.BoundaryNorm(bounds, newcmap.N)

    return norm, newcmap

def generate_heatmap():
    '''
    Use csv containing 800~ cinemas (scraped from Google), and plot them (using postcodes) on a map of the UK
    Then compare to a population of the UK (heatmap)
    Comment on whether coverage is good - can add to intro (justification for study)
    '''
    norm, newcmap = prepare_map_colour_scheme()

    # load data containing UK population - obtained from https://hub.worldpop.org/geodata/summary?id=29480
    tif_file = rasterio.open(os.path.join(os.getcwd(), 'population_cinema_heatmap', 'gbr_ppp_2020_UNadj.tif'))

    uk_worldpop_raster_tot = tif_file.read(1)

    uk_worldpop_raster_tot[uk_worldpop_raster_tot < 0] = None
    plt.rcParams['figure.figsize'] = 10, 10
    plt.imshow(np.log10(uk_worldpop_raster_tot + 1), norm=norm, cmap=newcmap)
    bar = plt.colorbar(fraction=0.1)

    # TODO - plot population density values on the right with each colour - note that the max value is 1193 people per pixel (100 metres)


def generate_postcode_mapping():

    # load data containing all cinemas in the UK (google scrape)
    cinema_file = "2023-04-27_cinema_and_post_codes.csv"
    list_of_cinemas_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'google_maps_scraper', 'output', cinema_file), header=0)
    list_of_postcodes = list_of_cinemas_df['postcode'].tolist()

    # TODO - passing postcodes into map
    # Issue with urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
    # https://stackoverflow.com/questions/59521203/using-pgeocode-lib-of-python-to-find-the-latitude-and-longitude - WIP
    nomi = pgeocode.Nominatim('GB')





if __name__ == '__main__':
    #generate_heatmap()

    generate_postcode_mapping()