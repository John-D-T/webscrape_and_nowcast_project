"""
pip install pytrends

https://lazarinastoy.com/the-ultimate-guide-to-pytrends-google-trends-api-with-python/
https://www.npmjs.com/package/google-trends-api#interestByRegion
"""

# TODO - weight by graph here: https://www.internetlivestats.com/google-search-statistics/
# connect to google

from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360, geo='GB')

# build payload

kw_list = ["cinemas nearby"] # list of keywords to get data

pytrends.build_payload(kw_list, cat=0, timeframe='all')

#1 Interest over Time
data = pytrends.interest_over_time()
data = data.reset_index()


import plotly.express as px

fig = px.line(data, x="date", y=['restaurants nearby'], title='Keyword Web Search Interest Over Time')
fig.show()