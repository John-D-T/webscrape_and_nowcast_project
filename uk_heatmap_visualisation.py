'''
Population heatmap:
https://towardsdatascience.com/creating-beautiful-population-density-maps-with-python-fcdd84035e06

Alternative:
https://medium.com/@patohara60/interactive-mapping-in-python-with-uk-census-data-6e571c60ff4

Cinema post-code plot (2 options):
https://github.com/bkontonis/Plot-PostCodes-on-Interactive-Map-Using-Python
https://stackoverflow.com/questions/58043978/display-data-on-real-map-based-on-postal-code
https://medium.com/@patohara60/interactive-mapping-in-python-with-uk-census-data-6e571c60ff4
https://stackoverflow.com/questions/61156007/mapping-uk-postcodes-to-geographic-boundaries-for-plotting

'''
# import rasterio
# import GDAL
'''
      INFO:root:Building on Windows requires extra options to setup.py to locate needed GDAL files. More information is available in the README.
      ERROR: A GDAL API version must be specified. Provide a path to gdal-config using a GDAL_CONFIG environment variable or use a GDAL_VERSION environment variable.
      
      Soln:
      pip install GDAL 64 bit 3.7 python (https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
      pip install rasterio 64 bit 3.7 python - found clash (1.2.10 requires GDAL between 2.3->3.2
      
      C++ compiler errors:
      error: Microsoft Visual C++ 14.0 or greater is required.
      Need to run this code on VS Code? https://stackoverflow.com/questions/44951456/pip-error-microsoft-visual-c-14-0-is-required
'''

'''
Use csv containing 800~ cinemas (scraped from Google), and plot them (using postcodes) on a map of the UK
Then compare to a population of the UK (heatmap)
Comment on whether coverage is good - can add to intro (justification for study)
'''

# load data containing UK population
# tif file


#tif_file = rasterio.open('../../HumanImpact/population/data/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif')
#ghs_data = tif_file.read()

# generate population heatmap


# load data containing all cinemas in the UK (google scrape)

# generate cinema plot