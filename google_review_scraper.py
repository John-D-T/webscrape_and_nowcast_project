# -*- coding: utf-8 -*-
from googlemaps import GoogleMapsScraper
from datetime import datetime, timedelta
import argparse
import csv
from termcolor import colored
import time

#https://sites.google.com/site/tomihasa/google-language-codes


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


ind = {'most_relevant' : 0 , 'newest' : 1, 'highest_rating' : 2, 'lowest_rating' : 3 }
# TODO - add 'location' to HEADER list
HEADER = ['location', 'cinema_name', 'id_review', 'caption', 'relative_date', 'retrieval_date', 'rating', 'username', 'n_review_user', 'n_photo_user', 'url_user']
HEADER_W_SOURCE = ['location', 'cinema_name', 'id_review', 'caption', 'relative_date','retrieval_date', 'rating', 'username', 'n_review_user', 'n_photo_user', 'url_user', 'url_source']


def csv_writer(source_field,
               path='C:/Users/johnd/OneDrive/Documents/cbq/third_proper_year/diss/code/scraping_project/output'):
    outfile= str(datetime.now().date()) + '_gm_reviews.csv'
    targetfile = open(path + outfile, mode='a', encoding='utf-8', newline='\n')
    writer = csv.writer(targetfile, quoting=csv.QUOTE_MINIMAL)

    if source_field:
        h = HEADER_W_SOURCE
    else:
        h = HEADER
    writer.writerow(h)

    return writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google Maps reviews scraper.')
    parser.add_argument('--N', type=int, default=10, help='Number of reviews to scrape')
    parser.add_argument('--i', type=str, default='input/urls.txt', help='target URLs file')
    parser.add_argument('--sort_by', type=str, default='newest', help='most_relevant, newest, highest_rating or lowest_rating')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # store reviews in CSV file
    writer = csv_writer(args.source)

    with GoogleMapsScraper(debug=args.debug) as scraper:
        with open(args.i, 'r') as urls_file:
            for url in urls_file:

                cinema_name = str(url).split('/')[5].replace('+', '_').replace(',', '_')
                if args.place:
                    print(scraper.get_account(url))
                else: # TODO - refactor function to not say 'sort by'
                    error, location = scraper.sort_by(url, ind[args.sort_by])

                if error == 0:

                    n = 0

                    while n < args.N:

                        # logging to std out
                        print(colored('[Review ' + str(n) + ']', 'cyan'))

                        reviews = scraper.get_reviews(n, location, cinema_name)
                        if len(reviews) == 0:
                            break

                        for r in reviews:
                            row_data = list(r.values())
                            if args.source:
                                row_data.append(url[:-1])

                            writer.writerow(row_data)

                        n += len(reviews)
