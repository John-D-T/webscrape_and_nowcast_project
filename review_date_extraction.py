from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime

"""
You're right, the method I mentioned earlier only yields the relative date like '2 weeks ago'. To get the exact date of 
a Google review, you can use the selenium package to automate the process of opening a review and retrieving the review 
date.

Here's an example of how to get the exact date of a Google review using selenium:
"""
## TODO - NOT SUCCESSFUL, NEED TO LOOK ELSEWHERE
def review_scrape():
    # initialize webdriver
    driver = webdriver.Chrome('/path/to/chromedriver')

    # navigate to the Google Maps review page
    driver.get('https://www.google.com/maps/place/Picturehouse+Central/@51.5106893,-0.133724,17z/data=!4m8!3m7!1s0x487604d3c1a457dd:0x793e8e616512292f!8m2!3d51.5106893!4d-0.133724!9m1!1b1!16s%2Fg%2F11b7ynnfvr?authuser=0&hl=en&inline=false')

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
    date_element = driver.find_element('xpath','/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[10]/div[1]/div[2]/div[3]/div[4]/div[1]/span[3]')
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

if __name__ == '__main__':
    review_scrape()