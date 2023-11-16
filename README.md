# Web Scraping and Nowcasting Project 
Hi! Welcome - this repository was created to facilitate exploring the use of cinema data and web scraping techniques as an alternative economic proxy for economic activity in the UK (United Kingdom). 

The high-level logic of the pipeline is as follows:

- Obtain data from numerous sources. This includes google reviews, google trends, twitter, imdb, and bfi box office data.

- Generate visualizations of the rating distribution of movies and population heatmap. This is to provide context for the paper

- Generate regressions and run checks. We check for multi-collinearity, distribution of residuals, auto-correlation, and homoskedasticity 

- Analyze our regressions. We then use machine learning to nowcast GDP using the data 

- Plot our nowcasting results.

**Breakdown of different folders**

 - `/twitter_scraper` web scraping twitter (utilizing python package Twint)
 - `/google_trends_scraper` web scraping google trends (working off existing code, credit to https://medium.com/@isguzarsezgin/scraping-google-reviews-with-selenium-python-23135ffcc331 and https://towardsdatascience.com/scraping-google-maps-reviews-in-python-2b153c655fc2)
 - `/google_maps_scraper` web scraping google reviews 
 - `/imdb_analysis` code to process and analyse IMDb reviews (data sourced from Kaggle)
 - `/bfi_data_compile` code to compile 800 csv files containing weekly movie gross data in the UK
 - `/regression_analysis` code to generate linear regression using above data sources, then run linear regression violation checks
    - `/machine_learning` code to pass data into various machine learning models, to ultimately nowcast data.
- `/population_cinema_heatmap` using geospatial data to generate a heat map of the UK

