import argparse
import csv
import os
from datetime import datetime
from common.constants import google_maps_scraper_output, LIST_OF_LONDON_POSTCODE_AREAS, LIST_OF_NON_LONDON_POSTCODE_AREAS

from termcolor import colored

from google_maps_scraper.googlemaps import GoogleMapsScraper

"""
PYTHON 3.8 (64 BIT) 

pip install bs4 selenium beautifulsoup4 webdriver_manager pandas termcolor time googlemaps argparse pymongo
"""

price_filter_dict = {'£' : 0 , '££' : 1, '£££' : 2, '££££' : 3}
HEADER = ['cinema_name', 'category', 'cinema_url', 'postcode_category']
HEADER_W_SOURCE = ['cinema_name', 'category', 'cinema_url', 'postcode_category', 'url_source']


def csv_writer(source_field, path=google_maps_scraper_output):
    outfile= str(datetime.now().date()) + '_list_of_cinemas.csv'
    targetfile = open(os.path.join(path, outfile), mode='a', encoding='utf-8', newline='\n')
    writer = csv.writer(targetfile, quoting=csv.QUOTE_MINIMAL)

    if source_field:
        h = HEADER_W_SOURCE
    else:
        h = HEADER
    writer.writerow(h)

    return writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google Maps reviews scraper.')
    parser.add_argument('--N', type=int, default=10, help='Number of cinema to scrape')
    # parser.add_argument('--i', type=str, default='auxilliary_data/urls_location.txt', help='target URLs files')
    parser.add_argument('--sort_by', type=str, default='£', help='most_relevant or closest')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # store reviews in CSV file
    writer = csv_writer(args.source)

    REFINED_POSTCODE_CATEGORIES = []

    url_prefix = 'https://www.google.com/maps/search/cinemas+in+'
    for area in LIST_OF_LONDON_POSTCODE_AREAS:
        REFINED_POSTCODE_CATEGORIES.append(url_prefix + area.replace(' ', '+'))

    for area in LIST_OF_NON_LONDON_POSTCODE_AREAS:
        REFINED_POSTCODE_CATEGORIES.append(url_prefix + area.replace(' ', '+'))

    with GoogleMapsScraper(debug=args.debug) as scraper:
        for url in REFINED_POSTCODE_CATEGORIES:
            postcode_category = url.split('in+')[1]
            print(url)
            scraper.bypass_cookies(url, price_filter_dict[args.sort_by])

            n = 0

            while n < args.N:

                print(colored('[ Cinemas collected: ' + str(n) + ']', 'cyan'))

                list_of_cinemas = scraper.get_cinemas(n, postcode_category)
                if len(list_of_cinemas) == 0:
                    break

                for r in list_of_cinemas:
                    row_data = list(r.values())
                    if args.source:
                        row_data.append(url[:-1])

                    writer.writerow(row_data)

                n += len(list_of_cinemas)
