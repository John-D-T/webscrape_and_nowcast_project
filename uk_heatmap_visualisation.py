'''
Population heatmap:
https://towardsdatascience.com/creating-beautiful-population-density-maps-with-python-fcdd84035e06

Alternative:
https://medium.com/@patohara60/interactive-mapping-in-python-with-uk-census-data-6e571c60ff4

Cinema post-code plot (several options):
https://github.com/bkontonis/Plot-PostCodes-on-Interactive-Map-Using-Python
https://stackoverflow.com/questions/58043978/display-data-on-real-map-based-on-postal-code
https://medium.com/@patohara60/interactive-mapping-in-python-with-uk-census-data-6e571c60ff4
https://stackoverflow.com/questions/61156007/mapping-uk-postcodes-to-geographic-boundaries-for-plotting

      PYTHON 3.8 (64 BIT)

      pip install pip install GDAL-3.3.3-cp38-cp38-win_amd64.whl (https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
      pip install rasterio
'''

import rasterio
import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
import matplotlib.colors as colors


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

    # TODO - plot population density values on the right with each colour - note that the max value is 1193 people per square kilometer

    # load data containing all cinemas in the UK (google scrape)

    # generate cinema plot

if __name__ == '__main__':
    generate_heatmap()