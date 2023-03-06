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

## TODO - Refactor/test this to get review date:
"""
You're right, the method I mentioned earlier only yields the relative date like '2 weeks ago'. To get the exact date of 
a Google review, you can use the selenium package to automate the process of opening a review and retrieving the review 
date.

Here's an example of how to get the exact date of a Google review using selenium:
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime

# initialize webdriver
driver = webdriver.Chrome('/path/to/chromedriver')

# navigate to the Google Maps review page
driver.get('https://www.google.com/maps/place/...')

# wait for the reviews to load
reviews_section = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, 'section-review'))
)

# find the review you want to scrape and click on it
review = reviews_section.find_elements_by_class_name('section-review')[0]
review.click()

# wait for the review details to load
review_details = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, 'section-review-content'))
)

# extract the review date
date_element = review_details.find_element_by_class_name('section-review-publish-date')
date_text = date_element.get_attribute('innerHTML')
date = datetime.datetime.strptime(date_text, '%b %d, %Y').date()

# print the date in YYYY-MM-DD format
print(date.strftime('%Y-%m-%d'))

# close the webdriver
driver.quit()
"""
#In this example, we use selenium to automate the process of opening a Google Maps review page and retrieving the review 
date. After navigating to the review page, we wait for the reviews to load using WebDriverWait. We then find the review 
we want to scrape and click on it to open the review details. After waiting for the review details to load, we extract 
the review date by finding the HTML element that contains the date information and getting the innerHTML of that element. 
We then convert the date text into a datetime object using strptime and extract the date using the date() method. Finally, 
we print the date in YYYY-MM-DD format and close the webdriver.

#Note that this approach can be slower than using BeautifulSoup and requires more code, but it allows you to extract 
more detailed information about each review, including the exact date and other metadata. Also note that as mentioned 
earlier, Google Maps terms of service prohibit web scraping, so make sure to use this information only for personal or 
non-commercial use, and be respectful of Google's terms of service.
"""

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
                scraper.bypass_cookies(url, price_filter_dict[args.sort_by])

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
