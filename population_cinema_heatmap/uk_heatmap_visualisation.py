"""
PYTHON 3.8 (64 BIT)

pip install GDAL-3.3.3-cp38-cp38-win_amd64.whl (https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
pip install rasterio
pip install pgeocode
pip install folium (WIP)
pip install geopy

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

import pandas as pd
from geopy.geocoders import Nominatim
import folium
from folium.plugins import FastMarkerCluster

import geopandas as gpd
from shapely.geometry import Point, Polygon


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
    tif_file = rasterio.open(os.path.join(os.getcwd(), 'gbr_ppp_2020_UNadj.tif'))
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

    # SOLN 2: https://towardsdatascience.com/geocode-with-python-161ec1e62b89
    #         https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972

    from geopy.extra.rate_limiter import RateLimiter

    locator = Nominatim(user_agent="myGeocoder")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    list_of_cinemas_df['location'] = list_of_cinemas_df['postcode'].apply(geocode)
    list_of_cinemas_df['point'] = list_of_cinemas_df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    # split point column into latitude, longitude and altitude columns
    list_of_cinemas_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(list_of_cinemas_df['point'].tolist(), index=list_of_cinemas_df.index)

    # check for nulls - only 6 at the moment
    list_of_cinemas_df.latitude.isnull().sum()

    # removing nulls
    list_of_cinemas_df = list_of_cinemas_df[pd.notnull(list_of_cinemas_df["latitude"])]

    list_of_cinemas_plt_df = list_of_cinemas_df

    # https://realpython.com/python-folium-web-maps-from-data/#:~:text=Python's%20Folium%20library%20gives%20you,can%20share%20as%20a%20website.
    map = folium.Map(
        location=[55.3781, 3.4360],
        tiles='cartodbpositron',
        zoom_start=6
    )
    # https://snyk.io/advisor/python/folium/functions/folium.CircleMarker
    list_of_cinemas_df.apply(lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=2, fill=True).add_to(map), axis=1)
    map.save("cinema_heatmap_zoom.html")
    #map.save("cinema_heatmap_zoom.png")

    # TODO - attempt a matplotlib plot - https://github.com/rylativity/identifying_wnv_locations-raw_repo-/blob/master/03-Geo_modeling.ipynb
    geometry = [Point(xy) for xy in zip(list_of_cinemas_plt_df.Longitude, list_of_cinemas_plt_df.Latitude)]
    list_of_cinemas_plt_df['geometry'] = geometry
    list_of_cinemas_plt_df.drop(['Latitude', 'Longitude'], axis=1, inplace=True)
    crs = {'init': 'epsg:4326'}
    spray_locs = gpd.GeoDataFrame(list_of_cinemas_plt_df, crs=crs, geometry=geometry)
    city_bound = gpd.read_file(os.path.join(os.getcwd(), "Great_Britain_shapefile", "gb_1km.shp"))

    fig, ax = plt.subplots(figsize=(8, 8))
    spray_locs.geometry.plot(marker='d', color='blue', markersize=1, ax=ax, label='Spray Loc')
    list_of_cinemas_plt_df.plot(color='grey', ax=ax, alpha=0.4)


if __name__ == '__main__':
    #generate_heatmap()

    generate_postcode_mapping()