# Web Scraping and Nowcasting Project 

Hi! Welcome - this repository was created to facilitate exploring the use of cinema data and web scraping techniques as an alternative economic proxy for economic activity in the UK (United Kingdom). 

Context:

In economic analysis, there has been a trend of using unconventional indicators. 
Examples includes studies on newspaper sentiment, night lights, Google Places API, and more.
These unique metrics are especially useful in times of unexpected volatility (e.g. Covid)

This study investigates film related indicators as one of these potential unconventional indicators.
We ran a few checks to see whether film-based consumption would be a good proxy for economic activity, mainly through the demand side.
Studies have found an average yearly spent of £18.72 pre-pandemic (Statista, 2021), and
YED has been found to be positive in Germany (Dewenter & Westermann, 2005). 

Additionally, using web scraped data from google reviews + population heatmap data, we ran a visual check to see whether cinema density correlated with population density. 
This was to see if cinema distributions fairly represented the population, and were not just confined to specific demographics. 
This was a light check to investigate this, and we found a strong overlap
![alt text](resources/images/heatmaps.png?raw=true)

Due to the volatile nature of events such as covid, we wanted to focus on more short term forecasting: Nowcasting. 
It is less known, but used in capturing current state of the economy.

The high-level logic of the pipeline is as follows:

![alt text](resources/images/pipeline.png?raw=true)

- Obtain data from numerous sources. This includes google reviews, google trends, twitter, imdb, and bfi box office data.

- Generate visualizations of the rating distribution of movies and population heatmap. This is to provide context for the paper

- Generate regressions and run checks. We check for multi-collinearity, distribution of residuals, auto-correlation, and homoskedasticity 

- Analyze our regressions. We then use machine learning to nowcast GDP using the data 
  - The models we used are included in the below image:
    - Note: For the models, we ran a 48 month/1 month test train split, over a 5 year span.

![alt text](resources/images/nowcasting_ml_models.png?raw=true)

- Plot our nowcasting results.

Our results:

![alt text](resources/images/results_table.png?raw=true)

We found that our model performed significantly better than a GDP based autoregression model.

However the degree of information film indicators provide may be limited in isolation.

In conclusion, the takeaway from this is that film indicators would be potentially very useful for economic analysis if they were used alongside a suite of unconventional indicators.

Another takeaway is the power of data from unexpected sources. 
We used several less conventional data sources, and it would have been interesting to see the impact of using the sentiment of google review data as well (there were limitations to using this in a time series, as google reviews stores their reviews using relative date, which is low in granularity). 

If you're interested in reading into more of the code, please see the below. Thanks for reading!

**Breakdown of different folders**

 - `/twitter_scraper` web scraping twitter (utilizing python package Twint)
 - `/google_trends_scraper` web scraping google trends (working off existing code, credit to https://medium.com/@isguzarsezgin/scraping-google-reviews-with-selenium-python-23135ffcc331 and https://towardsdatascience.com/scraping-google-maps-reviews-in-python-2b153c655fc2)
 - `/google_maps_scraper` web scraping google reviews 
 - `/imdb_analysis` code to process and analyse IMDb reviews (data sourced from Kaggle)
 - `/bfi_data_compile` code to compile 800 csv files containing weekly movie gross data in the UK
 - `/regression_analysis` code to generate linear regression using above data sources, then run linear regression violation checks
    - `/machine_learning` code to pass data into various machine learning models, to ultimately nowcast data.
- `/population_cinema_heatmap` using geospatial data to generate a heat map of the UK

