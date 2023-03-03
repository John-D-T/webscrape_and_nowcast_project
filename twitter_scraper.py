"""
https://www.natasshaselvaraj.com/how-to-scrape-twitter/
https://github.com/twintproject/twint

TODO:
check out https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af


pip install Twint
"""

# TODO - figure out what this means: CRITICAL:root:twint.run:Twint:Feed:noDataExpecting value: line 1 column 1 (char 0)
    # Solution: pip install --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
    # Solution 2: pip install git+git://github.com/ajctrl/twint@patch-1

# ISSUE - twint.token.RefreshTokenException: Could not find the Guest token in HTML
    # SOL: Try again at home. Might be an IP address issue.

import twint
import pandas as pd

def twitter_scrape(keyword):
    c = twint.Config()

    # refer to https://medium.com/@michael45684568/using-twint-for-twitter-data-gathering-d7197a3d4ce1
    c.Search = [keyword]       # topic
    c.Limit = 500      # limit to the number of Tweets to scrape
    c.Store_csv = True       # store tweets in a csv file
    c.Output = keyword + '.csv'     # path to csv file
    #c.Near = 'United Kingdom'

    # runs query to web scrape from twitter
    twint.run.Search(c)

# limit search to tweets in the UK?

def analyze_scrape(output_file):
    # Now load the extracted csv into a df
    df = pd.read_csv(output_file)
    print(df['tweet'])


if __name__ == '__main__':
    # TODO - consider other keywords apart from 'eating out'
    keyword = "eating out"
    twitter_scrape(keyword=keyword)
    analyze_scrape(output_file="output/eating out.csv")