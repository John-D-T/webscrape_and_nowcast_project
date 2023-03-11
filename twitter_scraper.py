"""
https://www.natasshaselvaraj.com/how-to-scrape-twitter/
https://github.com/twintproject/twint

On 3.7 venv

pip install pandas
pip install Twint
"""

# TODO - figure out what this means: CRITICAL:root:twint.run:Twint:Feed:noDataExpecting value: line 1 column 1 (char 0)
    # Solution: pip install --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
    # Solution 2: pip install git+git://github.com/ajctrl/twint@patch-1

# ISSUE - twint.token.RefreshTokenException: Could not find the Guest token in HTML
    # SOL: changed the regex - then guest token was found

import twint
import pandas as pd
import datetime
import time

def twitter_scrape(keyword, year):
    for month in range(1, 13):
        for day in range(1, 32):
            if month == 2 and day > 28:
                break
            elif month in [4, 6, 9, 11] and day > 30:
                break
            else:
                start_date = datetime.datetime.strptime('%s-%s-%s' % (year, month, day), '%Y-%m-%d')
                end_date = start_date + datetime.timedelta(days=1)
                formatted_start_date = datetime.datetime.strftime(start_date, '%Y-%m-%d')
                formatted_end_date = datetime.datetime.strftime(end_date, '%Y-%m-%d')

                #file_name = keyword + '_%s_%s_%s.csv' % (year, month, day)
                file_name = keyword + '_%s.csv' % formatted_start_date
                c = twint.Config()
                c.Search = [keyword]  # topic
                # c.Limit = 4000      # limit to the number of Tweets to scrape
                # c.Since = '%s-%s-%s' % (year, month, day)
                # c.Until = '%s-%s-%s' % (year, month, day)
                c.Since = formatted_start_date
                c.Until = formatted_end_date
                c.Store_csv = True  # store tweets in a csv file
                c.Output = 'twitter_output/' + file_name  # path to csv file
                c.Near = 'United Kingdom'
                # key columns we want - language included to filter out non english tweets. Link added for traceability
                c.Custom_csv = ["id", "user_id", "date", "time", "username", "tweet", "language", "near", "link"]

                # refer to https://medium.com/@michael45684568/using-twint-for-twitter-data-gathering-d7197a3d4ce1

                # runs query to web scrape from twitter
                try:
                    print('attempting to scrape for %s' % file_name)
                    twint.run.Search(c)
                except Exception as e:
                    string_issue = 'issue in %s due to %s' % (file_name, e)
                    print(string_issue)
                    with open('errors.txt', 'a+') as file:
                        file.write(string_issue + '\n')


# limit search to tweets in the UK?

def analyze_scrape(output_file):
    # Now load the extracted csv into a df
    df = pd.read_csv(output_file)


if __name__ == '__main__':
    # TODO - consider other keywords: odeon, bfi, cinema (too broad?), just watched
    keyword = "odeon"
    years = [year for year in range(2010, 2011)]
    for year in years:
        twitter_scrape(keyword=keyword, year=year)
    #analyze_scrape(output_file="output/%s_tweets_%s.csv" % (keyword, year))