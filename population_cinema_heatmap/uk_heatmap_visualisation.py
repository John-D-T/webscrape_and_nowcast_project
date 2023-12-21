"""
PYTHON 3.11 (64 BIT)

pip install GDAL-3.4.3-cp311-cp311-win_amd64.whl (https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
pip install rasterio
pip install pgeocode
pip install folium
pip install geopy
pip install matplotlib

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

from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium


def prepare_map_colour_scheme():
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
    Plot population of the UK (heatmap)
    Note that the max value is 1193 people per pixel (100 metres)
    '''
    norm, newcmap = prepare_map_colour_scheme()

    # load data containing UK population - obtained from https://hub.worldpop.org/geodata/summary?id=29480
    tif_file = rasterio.open(os.path.join(os.getcwd(), 'gbr_ppp_2020_UNadj.tif'))
    uk_worldpop_raster_tot = tif_file.read(1)

    uk_worldpop_raster_tot[uk_worldpop_raster_tot < 0] = None

    plt.rcParams['figure.figsize'] = 10, 10
    plt.imshow(np.log10(uk_worldpop_raster_tot + 1), norm=norm, cmap=newcmap)
    plt.colorbar(fraction=0.1)


def generate_postcode_mapping():
    """
        Use csv containing 800~ cinemas (scraped from Google), and plot them (using postcodes) on a map of the UK

        https://towardsdatascience.com/geocode-with-python-161ec1e62b89
        https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972
    """
    # load data containing all cinemas in the UK (google scrape)
    cinema_file = "2023-04-27_cinema_and_post_codes.csv"
    list_of_cinemas_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'google_maps_scraper', 'output', cinema_file), header=0)

    locator = Nominatim(user_agent="myGeocoder")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

    # geopy.exc.GeocoderUnavailable at the minute
    list_of_cinemas_df['location'] = list_of_cinemas_df['postcode'].apply(geocode)

    list_of_cinemas_df['point'] = list_of_cinemas_df['location'].apply(lambda loc: tuple(loc.point) if loc else None)

    list_of_cinemas_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(list_of_cinemas_df['point'].tolist(), index=list_of_cinemas_df.index)

    # check for nulls - only 6 at the moment
    list_of_cinemas_df.latitude.isnull().sum()

    # removing nulls
    list_of_cinemas_df = list_of_cinemas_df[pd.notnull(list_of_cinemas_df["latitude"])]

    # https://realpython.com/python-folium-web-maps-from-data/#:~:text=Python's%20Folium%20library%20gives%20you,can%20share%20as%20a%20website.
    map = folium.Map(
        location=[55.3781, 3.4360],
        tiles='cartodbpositron',
        zoom_start=6
    )
    # https://snyk.io/advisor/python/folium/functions/folium.CircleMarker
    list_of_cinemas_df.apply(lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1, fill=True).add_to(map), axis=1)
    map.save("cinema_heatmap_zoom.html")


if __name__ == '__main__':
    generate_heatmap()
    generate_postcode_mapping()