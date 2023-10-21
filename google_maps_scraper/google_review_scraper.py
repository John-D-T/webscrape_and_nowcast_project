# -*- coding: utf-8 -*-
from google_maps_scraper.googlemaps import GoogleMapsScraper
from datetime import datetime
import argparse
import csv
from termcolor import colored
import pandas as pd
import os
from common.constants import google_maps_scraper_output

"""
PYTHON 3.8 (64 BIT) 

pip install bs4 selenium beautifulsoup4 webdriver_manager pandas termcolor time googlemaps argparse pymongo
"""

ind = {'most_relevant' : 0 , 'newest' : 1, 'highest_rating' : 2, 'lowest_rating' : 3 }
HEADER = ['location', 'cinema_name', 'id_review', 'caption', 'relative_date', 'retrieval_date', 'rating', 'username', 'n_review_user', 'n_photo_user', 'url_user']
HEADER_W_SOURCE = ['location', 'cinema_name', 'id_review', 'caption', 'relative_date','retrieval_date', 'rating', 'username', 'n_review_user', 'n_photo_user', 'url_user', 'url_source']


def csv_writer(source_field, path=google_maps_scraper_output):
    outfile = str(datetime.now().date()) + '_list_of_reviews.csv'
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
    parser.add_argument('--N', type=int, default=5000, help='Number of reviews to scrape')
    parser.add_argument('--i', type=str, default='', help='target URLs file') # need to refactor default to list of cinemas scraped
    parser.add_argument('--sort_by', type=str, default='newest', help='most_relevant, newest, highest_rating or lowest_rating')
    parser.add_argument('--place', dest='place', action='store_true', help='Scrape place metadata')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Run scraper using browser graphical interface')
    parser.add_argument('--source', dest='source', action='store_true', help='Add source url to CSV file (for multiple urls in a single file)')
    parser.set_defaults(place=False, debug=False, source=False)

    args = parser.parse_args()

    # store reviews in CSV file
    writer = csv_writer(args.source)

    list_of_cinemas_csv = 'output\\2023-03-08_list_of_cinemas_refined_v2.csv'
    london_cinemas_df = (os.path.join(os.getcwd(), list_of_cinemas_csv))

    with GoogleMapsScraper(debug=args.debug) as scraper:
            url_df = pd.read_csv(london_cinemas_df)
            for row in url_df.iterrows():
                url = row[1][2]
                print('scraping from %s' % url)
                cinema_name = str(url).split('/')[5].replace('+', '_').replace(',', '_')
                if args.place:
                    print(scraper.get_account(url))
                else:
                    error, location = scraper.collect_all_cinemas(url, ind[args.sort_by])

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
