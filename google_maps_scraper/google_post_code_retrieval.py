import argparse
import csv
import os
from datetime import datetime

from termcolor import colored

from google_maps_scraper.googlemaps import GoogleMapsScraper

"""
pip install bs4
pip install selenium
pip install beautifulsoup4
pip install webdriver_manager
pip install pandas
pip install termcolor
pip install time
pip install googlemaps
pip install argparse
pip install pymongo
"""

price_filter_dict = {'£' : 0 , '££' : 1, '£££' : 2, '££££' : 3}
HEADER = ['cinema_name', 'category', 'cinema_url', 'postcode_category']
HEADER_W_SOURCE = ['cinema_name', 'category', 'cinema_url', 'postcode_category', 'url_source']

def csv_writer(source_field, path='C:/Users/johnd/OneDrive/Documents/cbq/third_proper_year/diss/code/scraping_project/google_maps_scraper/output'):
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
    parser.add_argument('--sort_by', type=str, default='£', help='most_relevant or closest')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # store reviews in CSV file
    writer = csv_writer(args.source)

    list_of_all_cinemas = []

    # TODO - generate list of urls - create list from dataframe (list_of_cinemas)

    with GoogleMapsScraper(debug=args.debug) as scraper:
        for url in list_of_all_cinemas:
            postcode_category = url.split('in+')[1]
            print(url)
            scraper.bypass_cookies(url, price_filter_dict[args.sort_by])

            n = 0

            while n < args.N:

                # logging to std out
                print(colored('[ Cinemas collected: ' + str(n) + ']', 'cyan'))

                list_of_cinemas = scraper.get_cinema_postcode(n, postcode_category)
                if len(list_of_cinemas) == 0:
                    break

                for r in list_of_cinemas:
                    row_data = list(r.values())
                    if args.source:
                        row_data.append(url[:-1])

                    writer.writerow(row_data)

                n += len(list_of_cinemas)
