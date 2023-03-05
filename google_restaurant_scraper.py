# -*- coding: utf-8 -*-
import argparse
import csv
import os
from datetime import datetime

from termcolor import colored

from googlemaps import GoogleMapsScraper

#https://sites.google.com/site/tomihasa/google-language-codes

price_filter_dict = {'£' : 0 , '££' : 1, '£££' : 2, '££££' : 3}
# TODO - add a filter dictionary for 1. price tags, 2. cuisine, and maybe 3. cuisine?
HEADER = ['restaurant_name', 'restaurant_url']
HEADER_W_SOURCE = ['restaurant_name', 'restaurant_url', 'url_source']


def csv_writer(source_field, ind_sort_by, path='C:/Users/johnd/OneDrive/Documents/cbq/third_proper_year/diss/code/scraping_project/output'):
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
    parser.add_argument('--i', type=str, default='input/urls_location.txt', help='target URLs files')
    parser.add_argument('--sort_by', type=str, default='£', help='most_relevant or closest')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # store reviews in CSV file
    writer = csv_writer(args.source, args.sort_by)

    with GoogleMapsScraper(debug=args.debug) as scraper:
        with open(args.i, 'r') as urls_file:
            for url in urls_file:

                # TODO - At this point it's not sorting the cinemas, just clicking through the cookies
                scraper.sort_restaurants(url, price_filter_dict[args.sort_by])

                n = 0

                while n < args.N:

                    # logging to std out
                    print(colored('[ Cinemas collected: ' + str(n) + ']', 'cyan'))

                    reviews = scraper.get_restaurants(n)
                    if len(reviews) == 0:
                        break

                    for r in reviews:
                        row_data = list(r.values())
                        if args.source:
                            row_data.append(url[:-1])

                        writer.writerow(row_data)

                    n += len(reviews)
