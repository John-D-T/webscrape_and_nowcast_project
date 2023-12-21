"""
PYTHON 3.11 (64 BIT)

pip install pytrends
pip install plotly

https://lazarinastoy.com/the-ultimate-guide-to-pytrends-google-trends-api-with-python/
https://www.npmjs.com/package/google-trends-api#interestByRegion
"""

from pytrends.request import TrendReq
import plotly.express as px


def extract_from_google_trends(word_of_interest):
    """
    Function which extracts google trend data for keywords of interest. We use python package pytrends to aid with this.

    NOTE: It's worth noting that you may face a timeout through too many requests:
        ERROR pytrends.exceptions.TooManyRequestsError: The request failed: Google returned a response with code 429
        SOLN: https://stackoverflow.com/questions/50571317/pytrends-the-request-failed-google-returned-a-response-with-code-429

    :param word_of_interest: the word we want to extract google trend data for
    """
    # Connect to google
    pytrends = TrendReq(hl='en-US', tz=360, geo='GB')

    # build payload
    kw_list = [word_of_interest] # list of keywords to get data

    pytrends.build_payload(kw_list, cat=0, timeframe='all')

    data = pytrends.interest_over_time()
    data = data.reset_index()

    # Plot collected data
    fig = px.line(data, x="date", y=[word_of_interest], title='Web Search Interest for "%s" Over Time' % (word_of_interest))
    fig.show()


if __name__ == '__main__':
    cinemas_nearby = 'cinemas nearby'
    extract_from_google_trends(word_of_interest=cinemas_nearby)
