import argparse
import csv
import os
from datetime import datetime

import pandas as pd
from termcolor import colored

from common.constants import ScrapingProjectConstants as c
from google_maps_scraper.googlemaps import GoogleMapsScraper

"""
PYTHON 3.8 (64 BIT) 

pip install bs4 selenium beautifulsoup4 webdriver_manager pandas termcolor time googlemaps argparse pymongo
"""

price_filter_dict = {
    '£': 0,
    '££': 1,
    '£££': 2,
    '££££': 3
}
HEADER = ['cinema_url', 'full_postcode', 'postcode']
HEADER_W_SOURCE = ['cinema_url', 'full_postcode', 'postcode']


def csv_writer(source_field, path=c.google_maps_scraper_output):
    outfile= str(datetime.now().date()) + '_cinema_and_post_codes.csv'
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
    parser.add_argument('--sort_by', type=str, default='£', help='most_relevant or closest')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # store reviews in CSV file
    writer = csv_writer(args.source)

    cinema_df = pd.read_csv(os.path.join(os.getcwd(), 'output', '2023-03-08_list_of_cinemas_refined_v2.csv'))

    list_of_all_cinema_urls = cinema_df['cinema_url'].values.tolist()

    with GoogleMapsScraper(debug=args.debug) as scraper:
        i = 0
        for url in list_of_all_cinema_urls:
            i += 1
            print(url)
            scraper.bypass_cookies(url, price_filter_dict[args.sort_by])

            print(colored('[ Cinema and post codes collected: ' + str(i) + ']', 'cyan'))

            list_of_cinemas = scraper.get_cinema_postcode(url)
            if len(list_of_cinemas) == 0:
                break

            for r in list_of_cinemas:
                row_data = list(r.values())
                if args.source:
                    row_data.append(url[:-1])

                writer.writerow(row_data)
