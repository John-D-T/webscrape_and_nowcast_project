# -*- coding: utf-8 -*-
import logging
import time
import timeit
import traceback
from datetime import datetime

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

GM_WEBPAGE = 'https://www.google.com/maps/'
MAX_WAIT = 10
MAX_RETRY = 5
MAX_SCROLLS = 40


class GoogleMapsScraper:

    def __init__(self, debug=False):
        self.debug = debug
        self.driver = self.__get_driver()
        self.logger = self.__get_logger()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

        self.driver.close()
        self.driver.quit()

        return True

    def sort_by(self, url, ind):

        self.driver.get(url)
        self.__click_on_cookie_agreement()

        wait = WebDriverWait(self.driver, MAX_WAIT)

        # open dropdown menu
        clicked = False
        tries = 0
        while not clicked and tries < MAX_RETRY:
            try:
                #obtaining location
                location = '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[9]/div[3]/button/div[1]/div[2]/div[1]'
                location = self.driver.find_element('xpath', location).text

                more_reviews_xpath = '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[49]/div/button/span/span[2]'
                more_reviews = self.driver.find_element('xpath', more_reviews_xpath)
                self.driver.execute_script("arguments[0].scrollIntoView(true);", more_reviews)

                self.driver.execute_script("arguments[0].click();", more_reviews)

                menu_bt = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-value=\'Sort\']')))
                menu_bt.click()

                clicked = True
                time.sleep(3)
            except Exception as e:
                tries += 1
                self.logger.warn('Failed to click sorting button')

            # failed to open the dropdown
            if tries == MAX_RETRY:
                return -1

        #  element of the list specified according to ind
        recent_rating_bt = self.driver.find_elements("xpath", '//div[@role=\'menuitemradio\']')[ind]
        recent_rating_bt.click()

        # wait to load review (ajax call)
        time.sleep(5)

        return 0, location

    # Leaving as optional as we don't need to sort restaurants
    def bypass_cookies(self, url, ind):

        self.driver.get(url)
        status = self.__click_on_cookie_agreement()

        if status:
            print('cookie page bypassed')

        #wait = WebDriverWait(self.driver, MAX_WAIT)

        # open price menu
        # clicked = False
        # tries = 0
        # while not clicked and tries < MAX_RETRY:
        #     try:
        #         # TODO - this code seems kind of useless
        #         restaurant_list = self.driver.find_element('css selector', 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd')
        #         restaurant_list.send_keys(Keys.UP)
        #
        #         # agree = WebDriverWait(self.driver, 10).until(
        #         #     EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "Reject all")]')))
        #         # TODO - figure out alternative
        #         #menu_bt = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@class=\"e2moi \"]')))
        #         menu_bt = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "//button[onclick='CookieInformation.submitAllCategories();')]")))
        #
        #         price_tag = self.driver.find_element("xpath", '//span[contains(text(), "Price")]')
        #         # seems that it is_enabled() but not is_displayed()
        #
        #         rating_tag = self.driver.find_element("xpath", '//span[contains(text(), "Rating")]')
        #
        #         restaurant_list = self.driver.find_element('css selector', 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd')
        #         # restaurant list is displayed and enabled
        #
        #
        #         clicked = True
        #         time.sleep(3)
        #     except Exception as e:
        #         tries += 1
        #         self.logger.warn('Failed to click sorting button')
        #
        #     # failed to open the dropdown
        #     if tries == MAX_RETRY:
        #         return -1

        return 0

    def get_reviews(self, offset, location, cinema_name):

        # scroll to load reviews

        # wait for other reviews to load (ajax)
        time.sleep(4)

        self.__scroll()

        # expand review text
        self.__expand_reviews()

        # parse reviews
        response = BeautifulSoup(self.driver.page_source, 'html.parser')

        rblock = response.find_all('div', class_='jftiEf fontBodyMedium')
        parsed_reviews = []
        for index, review in enumerate(rblock):
            if index >= offset:
                parsed_reviews.append(self.__parse_reviews(review, location, cinema_name))

                # logging to std out
                print(self.__parse_reviews(review, location, cinema_name))

        return parsed_reviews

    def get_cinemas(self, offset, postcode_category):

        # scroll to load restaurants

        # wait for other reviews to load (ajax)
        time.sleep(4)

        #self.__scroll_cinema()

        # parse reviews
        response = BeautifulSoup(self.driver.page_source, 'html.parser')

        ## Function to scroll the side bar down to the end
        start = timeit.default_timer()

        # identify scrolling element first
        scrolling_element_xpath = '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[1]'
        scrolling_element = self.driver.find_element('xpath', scrolling_element_xpath)

        ## need to find a way to keep scrolling until end without specifying
        SCROLL_PAUSE_TIME = 2.0

        # Get scroll height
        last_height = self.driver.execute_script("return arguments[0].scrollHeight", scrolling_element)
        print(last_height)

        t = 0
        while True:
            # print(t)
            t = t + 1
            # Scroll down to bottom
            self.driver.execute_script('arguments[0].scrollTo(0, arguments[0].scrollHeight)', scrolling_element)

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return arguments[0].scrollHeight", scrolling_element)
            # print(new_height)
            if new_height == last_height:
                break
            last_height = new_height

        stop = timeit.default_timer()

        print('Time taken: ', stop - start)

        rblock = response.find_all('div', class_='Nv2PK THOPZb CpccDe')
        parsed_restaurants = []
        for index, review in enumerate(rblock):
            if index >= offset:
                parsed_restaurants.append(self.__parse_cinema(review, postcode_category))

                # logging to std out
                print(self.__parse_cinema(review, postcode_category))

        return parsed_restaurants

    def get_account(self, url):

        self.driver.get(url)

        # ajax call also for this section
        time.sleep(4)

        resp = BeautifulSoup(self.driver.page_source, 'html.parser')

        place_data = self.__parse_place(resp)

        return place_data

    def __parse_reviews(self, review, location, cinema_name):

        item = {}

        try:
            # TODO: Subject to changes
            id_review = review['data-review-id']
        except Exception as e:
            id_review = None

        try:
            # TODO: Subject to changes
            username = review['aria-label']
        except Exception as e:
            username = None

        try:
            # TODO: Subject to changes
            review_text = self.__filter_string(review.find('span', class_='wiI7pd').text)
        except Exception as e:
            review_text = None

        try:
            # TODO: Subject to changes
            rating = float(review.find('span', class_='kvMYJc')['aria-label'].split(' ')[1])
        except Exception as e:
            rating = None

        try:
            # TODO: Subject to changes
            relative_date = review.find('span', class_='rsqaWe').text
        except Exception as e:
            relative_date = None

        try:
            n_reviews_photos = review.find('div', class_='section-review-subtitle').find_all('span')[1].text
            metadata = n_reviews_photos.split('\xe3\x83\xbb')
            if len(metadata) == 3:
                n_photos = int(metadata[2].split(' ')[0].replace('.', ''))
            else:
                n_photos = 0

            idx = len(metadata)
            n_reviews = int(metadata[idx - 1].split(' ')[0].replace('.', ''))

        except Exception as e:
            n_reviews = 0
            n_photos = 0

        try:
            user_url = review.find('a')['href']
        except Exception as e:
            user_url = None

        item['location'] = location
        item['cinema_name'] = cinema_name

        item['id_review'] = id_review
        item['caption'] = review_text
        # depends on language, which depends on geolocation defined by Google Maps
        # custom mapping to transform into date should be implemented
        item['relative_date'] = relative_date

        # store datetime of scraping and apply further processing to calculate
        # correct date as retrieval_date - time(relative_date)
        item['retrieval_date'] = datetime.now()
        item['rating'] = rating
        item['username'] = username
        item['n_review_user'] = n_reviews
        item['n_photo_user'] = n_photos
        item['url_user'] = user_url

        return item

    def __parse_cinema(self, review, postcode):

        item = {}

        try:
            cinema_name = review['aria-label']
        except Exception as e:
            cinema_name = None

        try:
            category = review.contents[1].contents[3].contents[0].contents[0].contents[1].contents[3].contents[7]\
            .contents[3].contents[1].contents[1].contents[3].contents[0]
        except Exception as e:
            category = None

        try:
            cinema_url = review.find('a')['href']
        except Exception as e:
            cinema_url = None

        # store datetime of scraping and apply further processing to calculate
        # correct date as retrieval_date - time(relative_date)
        item['cinema_name'] = cinema_name
        item['category'] = category
        item['cinema_url'] = cinema_url
        item['postcode_category'] = postcode

        return item

    def __parse_place(self, response):

        place = {}
        try:
            place['overall_rating'] = float(response.find('div', class_='gm2-display-2').text.replace(',', '.'))
        except:
            place['overall_rating'] = 'NOT FOUND'

        try:
            place['n_reviews'] = int(response.find('div', class_='gm2-caption').text.replace('.', '').replace(',','').split(' ')[0])
        except:
            place['n_reviews'] = 0

        return place

    # expand review description
    def __expand_reviews(self):
        # use XPath to load complete reviews
        # TODO: Subject to changes
        # TODO refactor 4 - Update selenium functionality
        #links = self.driver.find_elements_by_xpath('//button[@jsaction="pane.review.expandReview"]')
        links = self.driver.find_elements("xpath", '//button[@jsaction="pane.review.expandReview"]')

        for l in links:
            l.click()
        time.sleep(2)

    def __scroll(self):
        # TODO: Subject to changes
        # TODO refactor 5 - Update selenium functionality
        #scrollable_div = self.driver.find_element_by_css_selector('div.m6QErb.DxyBCb.kA9KIf.dS8AEf')
        scrollable_div = self.driver.find_element("css selector", 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf')
        self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
        #self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def __scroll_cinema(self):
        # TODO - add a scroller - https://stackoverflow.com/questions/20986631/how-can-i-scroll-a-web-page-using-selenium-webdriver-in-python
        scrollable_div = self.driver.find_element("css selector", 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd')
        #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);", scrollable_div)
        self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
        # self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def __get_logger(self):
        # create logger
        logger = logging.getLogger('googlemaps-scraper')
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        fh = logging.FileHandler('gm-scraper.log')
        fh.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # add formatter to ch
        fh.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(fh)

        return logger

    def __get_driver(self, debug=False):
        options = Options()

        if not self.debug:
            options.add_argument("--headless")
        else:
            options.add_argument("--window-size=1366,768")

        options.add_argument("--disable-notifications")
        options.add_argument("--lang=en-GB")
        input_driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)

         # click on google agree button so we can continue (not needed anymore)
         # EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "I agree")]')))
        input_driver.get(GM_WEBPAGE)

        return input_driver

    # cookies agreement click
    def __click_on_cookie_agreement(self):
        try:
            agree = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "Reject all")]')))
            agree.click()

            # back to the main page
            # self.driver.switch_to_default_content()

            return True
        except:
            return False

    # util function to clean special characters
    def __filter_string(self, str):
        strOut = str.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        return strOut
