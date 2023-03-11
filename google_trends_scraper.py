"""
pip install pytrends

https://lazarinastoy.com/the-ultimate-guide-to-pytrends-google-trends-api-with-python/
https://www.npmjs.com/package/google-trends-api#interestByRegion

## ERROR pytrends.exceptions.TooManyRequestsError: The request failed: Google returned a response with code 429
## SOLN: https://stackoverflow.com/questions/50571317/pytrends-the-request-failed-google-returned-a-response-with-code-429
"""

# TODO - weight by graph here: https://www.internetlivestats.com/google-search-statistics/
# NO NEED TO weigh, google trends does that automatically
# connect to google

from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360, geo='GB')

# build payload

kw_list = ["cinemas nearby"] # list of keywords to get data

trend_df = pytrends.build_payload(kw_list, cat=0, timeframe='all')

#1 Interest over Time
data = pytrends.interest_over_time()
data = data.reset_index()


import plotly.express as px

fig = px.line(data, x="date", y=['restaurants nearby'], title='Keyword Web Search Interest Over Time')
fig.show()