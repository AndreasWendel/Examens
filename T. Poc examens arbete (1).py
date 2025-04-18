# Databricks notebook source
!pip install yfinance

# COMMAND ----------

import yfinance as yf

# Define the stock ticker (e.g., Apple)
ticker_symbol = "AAPL"

# Get ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch news
news = ticker.news

# Display news headlines
for i, article in enumerate(news):
    print(f"{i+1}. {article['content']['title']}")
    print(f"   Summary: {article['content']['summary']}")
    print(f"   Source: {article['content']['provider']['displayName']}")
    print(f"   Link: {article['content']['clickThroughUrl']['url']}\n")

# COMMAND ----------

news = ticker.get_news(count = 100, tab='Press Releases')

# Display news headlines
for i, article in enumerate(news):
    print(f"{i+1}. {article['content']['title']}")
    print(f"   Summary: {article['content']['summary']}")
    print(f"   Source: {article['content']['provider']['displayName']}")
    print(f"   Link: {article['content']['clickThroughUrl']['url']}\n")

# COMMAND ----------

temp_news = ticker.get_news(count = 100, tab='Press Releases')

# COMMAND ----------

temp_news

# COMMAND ----------

len(temp_news)

# COMMAND ----------

len(news)

# COMMAND ----------

# 2526405fe949497ea6aeb3355776862e

import requests

# Replace with your own NewsAPI key
API_KEY = "2526405fe949497ea6aeb3355776862e"

# Define the stock or company name
company_name = "'Apple'"
# Define the endpoint and parameters
url = "https://newsapi.org/v2/everything"
params = {
    "q": company_name,  # Search query (e.g., company name)
    "language": "en",   # Language filter
    "sortBy": "relevancy",  # Sort by recent news, relevancy = articles more closely related to q come first. popularity = articles from popular sources and publishers come first. publishedAt = newest articles come first.
    "apiKey": API_KEY,   # API Key
    "from" : "2025-03-16",
    "to" : "2025-03-16"
}

# Make the request
response = requests.get(url, params=params)
data = response.json()

# Check if the request was successful
if data["status"] == "ok":
    articles = data["articles"]
    for i, article in enumerate(articles[:]):  # Show the top 5 articles
        print(f"{i+1}. {article['title']}")
        print(f"   description: {article['description']}")
        print(f"   Source: {article['source']['name']}")
        print(f"   Published: {article['publishedAt']}")
        print(f"   Link: {article['url']}\n")
else:
    print("Error fetching news:", data.get("message", "Unknown error"))


# COMMAND ----------

articles

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import json

def scrape_article(url, source, date, title):
    # Step 1: Request the page
    headers = {
        "User-Agent": "Mozilla/5.0"  # Helps avoid bot detection
    }
    response = requests.get(url, headers=headers)

    # Step 2: Parse with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 3: Extract the article content
    if source == "The Verge":
        article_body = soup.find_all('div', class_='duet--article--article-body-component')

    elif source == "MacRumors":
        content_container = soup.find('div', class_='content--2u3grYDr js-content')
        if not content_container:
            print("Content container not found")
            return None
        article_body = content_container.find_all('div', class_='ugc--2nTu61bm minor--3O_9dH4U')
    else:
        print(source, "this source doens't support scraping")

    paragraphs = []

    for div in article_body:
        p_tags = div.find_all('p')
        for p in p_tags:
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

    # Step 4: Optional – package into JSON
    data = {
        "url": url,
        "title": title,
        "paragraphs": paragraphs,
        "publishedAt": date
    }

    # Step 5: Print or return
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data



# COMMAND ----------

sources = []
for i in range(len(articles)):
    print(i, articles[i]["source"]["name"], articles[i]["publishedAt"], articles[i]["title"])
    sources.append(articles[i]["source"]["name"])

# COMMAND ----------

distinct_sources = list(set(sources))
distinct_sources

# COMMAND ----------


# 🔍 Example URL (replace with real article)
test = scrape_article(articles[4]["url"], articles[4]["source"]["name"], articles[4]["publishedAt"], articles[4]["title"])

# COMMAND ----------

test["paragraphs"][1]

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import json

def scrape_macrumors_article(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')

    # MacRumors article content is in this div
    content_container = soup.find('div', class_='content--2u3grYDr js-content')
    if not content_container:
        print("Content container not found")
        return None

    # Find all the 'ugc' sub-containers with text
    subcontainers = content_container.find_all('div', class_='ugc--2nTu61bm minor--3O_9dH4U')

    paragraphs = []
    for div in subcontainers:
        for p in div.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

    # Package into a dictionary
    data = {
        "url": url,
        "paragraphs": paragraphs
    }

    # Pretty-print it
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data

scrape_macrumors_article(articles[4]["url"])

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import json

def scrape_macrumors_article(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')

    # MacRumors article content is in this div
    content_container = soup.find('div', class_='content--2u3grYDr js-content')
    if not content_container:
        print("Content container not found")
        return None

    # Find all the 'ugc' sub-containers with text
    subcontainers = content_container.find_all('div', class_='ugc--2nTu61bm minor--3O_9dH4U')

    paragraphs = []
    for div in subcontainers:
        for p in div.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

    # Package into a dictionary
    data = {
        "url": url,
        "paragraphs": paragraphs
    }

    # Pretty-print it
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data


# COMMAND ----------

len(test["paragraphs"])

# COMMAND ----------

import yfinance as yf
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
#from dotenv import load_dotenv
import time

# Load environment variables (for API keys)
#load_dotenv()

def get_stock_data(ticker, period="1y"):
    """
    Retrieve historical stock data using yfinance
    
    Parameters:
    ticker (str): Stock symbol
    period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
    DataFrame: Historical price data
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

def plot_stock_chart(data, ticker, save_path=None):
    """
    Create and optionally save a stock price chart
    
    Parameters:
    data (DataFrame): Stock price data
    ticker (str): Stock symbol
    save_path (str): Path to save the chart (optional)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label=f'{ticker} Close Price')
    
    # Add moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    plt.plot(data.index, data['MA50'], label='50-day MA', alpha=0.7)
    plt.plot(data.index, data['MA200'], label='200-day MA', alpha=0.7)
    
    # Add volume as subplot
    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.3)
    ax2.set_ylabel('Volume')
    
    plt.title(f'{ticker} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def get_finnhub_news(ticker, from_date=None, to_date=None, api_key="cv2ott1r01qqpq6jo0v0cv2ott1r01qqpq6jo0vg"):
    """
    Fetch company news from Finnhub API
    
    Parameters:
    ticker (str): Stock symbol
    from_date (str): Start date in format YYYY-MM-DD
    to_date (str): End date in format YYYY-MM-DD
    
    Returns:
    list: News articles
    """
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    finnhub_key = api_key
    if not finnhub_key:
        print("Warning: FINNHUB_API_KEY not found in environment variables")
        return []
        
    url = f'https://finnhub.io/api/v1/company-news'
    params = {
        'symbol': ticker,
        'from': from_date,
        'to': to_date,
        'token': finnhub_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

def get_alpha_vantage_news(keywords, from_date=None, to_date=None, api_key="R6WUON3CB6JRP1K7"):
    """
    Fetch news using Alpha Vantage API
    
    Parameters:
    keywords (str): Search keywords or company name
    from_date (str): Start date in format YYYYMMDD
    to_date (str): End date in format YYYYMMDD
    
    Returns:
    dict: News articles and sentiment information
    """

    alpha_key = api_key
    if not alpha_key:
        print("Warning: ALPHA_VANTAGE_API_KEY not found in environment variables")
        return {}
    
    url = 'https://www.alphavantage.co/query?'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'apikey': api_key,
        'limit': 1000,
        'sort': 'RELEVANCE'
    }
    
    if from_date and to_date:
        params['time_from'] = from_date
        params['time_to'] = to_date
            
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching news: {response.status_code}")
        return {}
    

def display_news(news_data, source="finnhub", max_articles=5):
    """
    Display news articles in a formatted way
    
    Parameters:
    news_data: News data from API
    source (str): Source of news data ('finnhub' or 'alpha')
    max_articles (int): Maximum number of articles to display
    """
    if source == "finnhub":
        articles = news_data[:max_articles]
        for i, article in enumerate(articles, 1):
            print(f"\n--- Article {i} ---")
            print(f"Headline: {article.get('headline', 'N/A')}")
            print(f"Source: {article.get('source', 'N/A')}")
            print(f"Date: {article.get('datetime', 'N/A')}")
            print(f"URL: {article.get('url', 'N/A')}")
            print(f"Summary: {article.get('summary', 'N/A')[:200]}...")
    
    elif source == "alpha":
        if "feed" not in news_data:
            print("No news articles found")
            return
            
        articles = news_data["feed"][:max_articles]
        for i, article in enumerate(articles, 1):
            print(f"\n--- Article {i} ---")
            print(f"Title: {article.get('title', 'N/A')}")
            print(f"Source: {article.get('source', 'N/A')}")
            print(f"Date: {article.get('time_published', 'N/A')}")
            print(f"URL: {article.get('url', 'N/A')}")
            
            # Display sentiment if available
            sentiment = article.get('overall_sentiment_score', 'N/A')
            if sentiment != 'N/A':
                sentiment_label = "Positive" if float(sentiment) > 0 else "Negative" if float(sentiment) < 0 else "Neutral"
                print(f"Sentiment: {sentiment_label} ({sentiment})")
                
            print(f"Summary: {article.get('summary', 'N/A')[:200]}...")

def analyze_news_with_price_movement(ticker, news_articles, price_data, window=1):
    """
    Basic analysis to correlate news with price movements
    
    Parameters:
    ticker (str): Stock symbol
    news_articles (list): News articles data
    price_data (DataFrame): Historical price data
    window (int): Number of days to look ahead for price movement

    Returns:
    DataFrame: News with associated price movements
    """
    # Prepare results dataframe
    results = []
    
    for article in news_articles:
        # Convert Unix timestamp to datetime for Finnhub
        if 'datetime' in article and isinstance(article['datetime'], int):
            article_date = datetime.fromtimestamp(article['datetime'])
        # Parse ISO format for Alpha Vantage
        elif 'time_published' in article:
            article_date = datetime.strptime(article['time_published'], '%Y%m%dT%H%M%S')
        else:
            continue
            
        article_date = article_date.strftime('%Y-%m-%d')
        
        # Find the closing price on news date and window days later
        try:
            # Get the next trading day if article date is not in price data
            trading_dates = price_data.index.strftime('%Y-%m-%d').tolist()
            
            if article_date not in trading_dates:
                # Find the next trading day
                for date in trading_dates:
                    if date > article_date:
                        next_trading_day = date
                        break
                else:
                    continue
            else:
                next_trading_day = article_date
                
            # Get the price on news date
            start_price = price_data.loc[next_trading_day, 'Close']
            
            # Find the price window days later
            idx = trading_dates.index(next_trading_day)
            if idx + window < len(trading_dates):
                future_date = trading_dates[idx + window]
                future_price = price_data.loc[future_date, 'Close']
                
                # Calculate price change
                price_change = ((future_price - start_price) / start_price) * 100
                
                # Get headline from the right source
                headline = article.get('headline', article.get('title', 'N/A'))
                
                results.append({
                    'date': article_date,
                    'headline': headline[:100],
                    'source': article.get('source', 'N/A'),
                    'price_change_pct': round(price_change, 2)
                })
        except Exception as e:
            print(f"Error processing article from {article_date}: {e}")
            continue
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        return df.sort_values('price_change_pct', ascending=False)
    else:
        return pd.DataFrame(columns=['date', 'headline', 'source', 'price_change_pct'])

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    from_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%dT%H%M')
    end_date = (datetime.now() - timedelta(days=0)).strftime('%Y%m%dT%H%M')
    # Get stock data
    print(f"Fetching historical data for {ticker}...")
    stock_data = get_stock_data(ticker, period="1y")
    
    # Plot stock chart
    plot_stock_chart(stock_data, ticker)
    
    # Try to get news from either Finnhub or Alpha Vantage
    # You'll need to set your API keys as environment variables
    # or create a .env file with FINNHUB_API_KEY and/or ALPHA_VANTAGE_API_KEY
    
    print(f"\nFetching news for {ticker}...")
    
    # Try Finnhub first
    #news = get_finnhub_news(ticker, from_date = from_date, to_date = end_date)
    #news_source = "finnhub"
    news = get_alpha_vantage_news(ticker, from_date = from_date, to_date = end_date)
    news_source = "alpha"
    
    # If Finnhub fails, try Alpha Vantage
    #if not news:
    #    print("Trying Alpha Vantage instead...")
    #    news = get_alpha_vantage_news(ticker, from_date = from_date, to_date = end_date)
    #    news_source = "alpha"
    
    # Display news
    if news:
        display_news(news, source=news_source)
        
        # Analyze news with price movements
        print("\n--- News Impact Analysis ---")
        impact_df = analyze_news_with_price_movement(ticker, 
                                                   news if news_source == "finnhub" else news.get("feed", []),
                                                   stock_data)
        
        if not impact_df.empty:
            print("\nMost positive news impact:")
            print(impact_df.head(3).to_string(index=False))
            
            print("\nMost negative news impact:")
            print(impact_df.tail(3).to_string(index=False))
        else:
            print("Could not analyze news impact - insufficient data")
    else:
        print("No news data available. Please check your API keys or try another ticker.")

# COMMAND ----------

news[-2]

# COMMAND ----------

len(impact_df)

# COMMAND ----------

display(impact_df)

# COMMAND ----------

api_key="CRB7WDSMNX8RRXHT"
ticker = "AAPL"
from_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%dT%H%M')
to_date = (datetime.now() - timedelta(days=0)).strftime('%Y%m%dT%H%M')
alpha_key = api_key
if not alpha_key:
    print("Warning: ALPHA_VANTAGE_API_KEY not found in environment variables")
        
url = 'https://www.alphavantage.co/query?'
params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'apikey': api_key,
        'limit': 1000,
        'sort': 'RELEVANCE'
    }
    
if from_date and to_date:
    params['time_from'] = from_date
    params['time_to'] = to_date
        
response = requests.get(url, params=params)
if response.status_code == 200:
    response.json()
else:
    print(f"Error fetching news: {response.status_code}")


# COMMAND ----------

from_date
to_date

# COMMAND ----------

response.json()

# COMMAND ----------

from_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%dT%H%M')

# COMMAND ----------


from_date.strftime('%Y-%m-%d')

# COMMAND ----------

from datetime import datetime

from_date = datetime.strptime(from_date, '%Y%m%dT%H%M')
type(from_date)

# COMMAND ----------

datetime.strptime(from_date, '%Y%m%dT%H%M').strftime('%Y%m%dT%H%M')

# COMMAND ----------

YYYYMMDDTHHMM
20250303T141500

# COMMAND ----------

news_alpha = response.json()

# COMMAND ----------

news_alpha["feed"]

# COMMAND ----------

url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=CRB7WDSMNX8RRXHT'
r = requests.get(url)
data = r.json()

# COMMAND ----------

data

# COMMAND ----------

!pip install finnhub-python

# COMMAND ----------

import finnhub

finnhub_client = finnhub.Client(api_key="cv2ott1r01qqpq6jo0v0cv2ott1r01qqpq6jo0vg")

# COMMAND ----------

finn_AAPL_news = finnhub_client.company_news('AAPL', _from="2024-01-01", to="2024-12-10")

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import os

from datetime import datetime, timedelta
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import shap

# Load environment variables for API keys
# finnhub api key= cv2ott1r01qqpq6jo0v0cv2ott1r01qqpq6jo0vg
# alpha vantage api key = <R6WUON3CB6JRP1K7>

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class MarketSentimentPOC:
    def __init__(self):
        self.finnhub_key = None
        self.alpha_vantage_key = None
        self.news_data = None
        self.price_data = None
        self.labeled_data = None
        self.model = None
        self.vectorizer = None
        self.important_features = None

    def set_keys(self, finnhub_key=None, alpha_vantage_key=None):
        self.finnhub_key = finnhub_key
        self.alpha_vantage_key = alpha_vantage_key
    
    def collect_data(self, tickers, start_date=None, end_date=None, days_back=180):
        """
        Collect historical price and news data for multiple tickers
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Collecting data from {start_date} to {end_date} for {len(tickers)} tickers")
        
        # Initialize empty dataframes
        self.price_data = pd.DataFrame()
        all_news = []
        
        # Process each ticker
        for ticker in tickers:
            print(f"Processing {ticker}...")
            
            # Get price data
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                hist['ticker'] = ticker
                self.price_data = pd.concat([self.price_data, hist])
                
                # Calculate daily returns
                hist['return'] = hist['Close'].pct_change() * 100
                
                # Get news data for this ticker
                news = self.get_news_for_ticker(ticker, start_date, end_date)
                
                # Add ticker to the news data
                for article in news:
                    article['ticker'] = ticker
                
                all_news.extend(news)
                
                print(f"  - Got {len(news)} news articles and {len(hist)} days of price data")
                
            except Exception as e:
                print(f"  - Error processing {ticker}: {e}")
                continue
        
        # Convert news list to DataFrame
        if all_news:
            self.news_data = pd.DataFrame(all_news)
            print(f"Collected {len(self.news_data)} news articles and {len(self.price_data)} days of price data")
        else:
            print("Failed to collect any news data")
            
    def get_news_for_ticker(self, ticker, start_date, end_date):
        """
        Try to get news from Finnhub first, then Alpha Vantage as backup
        """

        if self.alpha_vantage_key:
            news = self.get_alpha_vantage_news(ticker, start_date, end_date)
            if news and "feed" in news and len(news["feed"]) > 0:
                print(f"  - Using Alpha Vantage news for {ticker}")
                return self.process_alpha_vantage_news(news)

        # Try Finnhub first
        if self.finnhub_key:
            news = self.get_finnhub_news(ticker, start_date, end_date)
            if news and len(news) > 0:
                print(f"  - Using Finnhub news for {ticker}")
                return self.process_finnhub_news(news)
        
        print(f"  - No news found for {ticker}")
        return []
    
    def get_finnhub_news(self, ticker, from_date, to_date):
        """Fetch company news from Finnhub API"""
        from_date = datetime.strptime(from_date, '%Y-%m-%d').strftime('%Y%m%dT%H%M')
        to_date = datetime.strptime(to_date, '%Y-%m-%d').strftime('%Y%m%dT%H%M')    
        url = 'https://finnhub.io/api/v1/company-news'
        params = {
            'symbol': ticker,
            'from': from_date_str,
            'to': to_date_str,
            'token': self.finnhub_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  - Finnhub API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"  - Error with Finnhub API: {e}")
            return []
        
    def get_alpha_vantage_news(self, keywords, from_date=None, to_date=None):
        """
        Fetch news using Alpha Vantage API
        
        Parameters:
        keywords (str): Search keywords or company name
        from_date (str): Start date in format YYYYMMDD
        to_date (str): End date in format YYYYMMDD
        
        Returns:
        dict: News articles and sentiment information
        """

        if not self.alpha_vantage_key:
            print("Warning: ALPHA_VANTAGE_API_KEY not found in environment variables")
            return {}
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': self.alpha_vantage_key,
            'limit': 1000,
            'sort': 'RELEVANCE'
        }
        from_date = datetime.strptime(from_date, '%Y-%m-%d').strftime('%Y%m%dT%H%M')
        to_date = datetime.strptime(to_date, '%Y-%m-%d').strftime('%Y%m%dT%H%M')
        if from_date and to_date:
            params['time_from'] = from_date
            params['time_to'] = to_date
                
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching news: {response.status_code}")
            return {}

    
    def process_finnhub_news(self, news_data):
        """Process and standardize Finnhub news data"""
        processed_news = []
        
        for article in news_data:
            if 'datetime' not in article or 'headline' not in article:
                continue
                
            # Convert Unix timestamp to date
            article_date = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
            
            processed_news.append({
                'date': article_date,
                'headline': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'api_source': 'finnhub'
            })
            
        return processed_news
    
    def process_alpha_vantage_news(self, news_data):
        """Process and standardize Alpha Vantage news data"""
        processed_news = []
        
        if "feed" not in news_data:
            return processed_news
            
        for article in news_data["feed"]:
            if 'time_published' not in article or 'title' not in article:
                continue
                
            # Convert time format to date
            try:
                article_date = datetime.strptime(
                    article['time_published'], '%Y%m%dT%H%M%S'
                ).strftime('%Y-%m-%d')
            except:
                continue
                
            # Get sentiment if available
            sentiment_score = None
            if 'overall_sentiment_score' in article:
                try:
                    sentiment_score = float(article['overall_sentiment_score'])
                except:
                    pass
                    
            processed_news.append({
                'date': article_date,
                'headline': article.get('title', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'api_source': 'alpha_vantage',
                'sentiment_score': sentiment_score
            })
            
        return processed_news
    
    def merge_news_with_returns(self, future_days=1):
        """
        Link news articles with subsequent price movements
        """
        if self.news_data is None or self.price_data is None:
            print("No data available for merging")
            return None
            
        # Make a copy of news data for processing
        labeled_news = self.news_data.copy()
        
        # Initialize columns for price movements
        labeled_news['market_move'] = np.nan
        labeled_news['next_day_return'] = np.nan
        labeled_news['future_return'] = np.nan
        
        # Process each news article
        for idx, article in labeled_news.iterrows():
            ticker = article['ticker']
            article_date = article['date']
            
            # Get ticker-specific price data
            ticker_prices = self.price_data[self.price_data['ticker'] == ticker]
            
            ticker_prices['return'] = ((ticker_prices['Open'] - ticker_prices['Close']) / ticker_prices['Close']) * 100

            # Check if there is price data for this date or after
            future_prices = ticker_prices[ticker_prices.index >= article_date]
            
            if len(future_prices) <= future_days:
                continue
                
            # Get the next trading day
            next_trading_day = future_prices.index[0]
            
            # Get price on next trading day
            if next_trading_day in ticker_prices.index:
                next_day_return = ticker_prices.loc[next_trading_day, 'return']
                labeled_news.at[idx, 'next_day_return'] = next_day_return
                
                # Label as positive or negative movement
                if not pd.isna(next_day_return):
                    if next_day_return > 0.5:  # Threshold for positive move
                        labeled_news.at[idx, 'market_move'] = 'positive'
                    elif next_day_return < -0.5:  # Threshold for negative move
                        labeled_news.at[idx, 'market_move'] = 'negative'
                    else:
                        labeled_news.at[idx, 'market_move'] = 'neutral'
            
            # Get return for the future period
            if len(future_prices) > future_days:
                future_trading_day = future_prices.index[future_days]
                if future_trading_day in ticker_prices.index:
                    # Calculate cumulative return over the period
                    start_price = ticker_prices.loc[next_trading_day, 'Close']
                    end_price = ticker_prices.loc[future_trading_day, 'Close']
                    future_return = ((end_price - start_price) / start_price) * 100
                    labeled_news.at[idx, 'future_return'] = future_return
        
        # Drop rows with missing returns
        self.labeled_data = labeled_news.dropna(subset=['future_return'])
        print(f"Created dataset with {len(self.labeled_data)} labeled news articles")
        
        return self.labeled_data
    
    def preprocess_text(self, use_lemmatization=True):
        """
        Preprocess headlines and summaries for NLP
        """
        if self.labeled_data is None or len(self.labeled_data) == 0:
            print("No labeled data available for preprocessing")
            return
            
        # Combine headline and summary
        self.labeled_data['text'] = self.labeled_data['headline'] + ' ' + self.labeled_data['summary'].fillna('')
        
        # Initialize lemmatizer if needed
        lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add financial terms that are too common to be useful
        financial_stopwords = {'stock', 'stocks', 'market', 'markets', 'company', 
                              'share', 'shares', 'price', 'prices', 'investor', 
                              'investors', 'trading', 'trader', 'traders', 'report',
                              'reports', 'reported', 'quarter', 'quarterly', 'fiscal',
                              'financial', 'earnings', 'revenue', 'revenues', 'profit',
                              'profits', 'loss', 'losses'}
        stop_words.update(financial_stopwords)
        
        # Function to clean text
        def clean_text(text):
            if not isinstance(text, str):
                return ""
                
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize if required
            if use_lemmatization:
                cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            else:
                cleaned_tokens = [token for token in tokens if token not in stop_words]
                
            # Join tokens back into text
            return ' '.join(cleaned_tokens)
            
        # Apply text cleaning
        print("Preprocessing text data...")
        self.labeled_data['cleaned_text'] = self.labeled_data['text'].apply(clean_text)
        
        # Remove empty texts
        self.labeled_data = self.labeled_data[self.labeled_data['cleaned_text'].str.strip() != '']
        print(f"After preprocessing: {len(self.labeled_data)} articles")
    
    def vectorize_text(self, method='tfidf', max_features=1000):
        """
        Convert preprocessed text to numerical features
        """
        if self.labeled_data is None or 'cleaned_text' not in self.labeled_data.columns:
            print("No preprocessed text available for vectorization")
            return None
            
        print(f"Vectorizing text using {method} method...")
        
        # Choose vectorizer
        if method == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features)
        else:  # default to tfidf
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            
        # Fit and transform the text data
        X = self.vectorizer.fit_transform(self.labeled_data['cleaned_text'])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create DataFrame with features
        X_df = pd.DataFrame(X.toarray(), columns=feature_names)
        
        print(f"Created {X_df.shape[1]} features from text")
        return X_df
    
    def build_model(self, X_df, target='future_return', test_size=0.2, random_state=42):
        """
        Build and evaluate a model for predicting market movements from text
        """
        if self.labeled_data is None or X_df is None:
            print("No data available for model building")
            return
            
        # Check if target exists in data
        if target not in self.labeled_data.columns:
            print(f"Target column '{target}' not found in data")
            return
            
        print(f"Building model to predict {target}...")
        
        # Get target values
        y = self.labeled_data[target].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=random_state
        )
        self.X_train = X_train
        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Build models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
        
        # Select best model based on R²
        best_model_name = max(results, key=lambda k: results[k]['r2'])
        self.model = results[best_model_name]['model']
        
        print(f"Best model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
        
        # Get feature importance for the best model
        self.extract_important_features(X_df.columns, best_model_name, results[best_model_name]['model'])
        
        return results
    
    def extract_important_features(self, feature_names, model_name, model):
        """
        Extract and visualize important features from the model
        """
        print("Extracting important features...")
        
        if model_name == 'Linear Regression':
            # For linear regression, use coefficients
            importance = np.abs(model.coef_)
            """elif model_name == 'Random Forest':
            # For Random Forest, use feature importance
            importance = model.feature_importances_"""
        elif model_name == 'Random Forest':
            # Use SHAP values
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer(self.X_train)
            # Calculate mean absolute SHAP value per feature
            importance = np.abs(shap_values.values).mean(axis=0)
        else:
            print(f"Feature importance extraction not implemented for {model_name}")
            return
            
        # Create a DataFrame with features and importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Save the top features
        self.important_features = feature_importance.head(30)
        
        print("Top 10 most important features:")
        print(self.important_features.head(10))
        
        return self.important_features
    
    def visualize_results(self, results, target='future_return'):
        """
        Create visualizations of model performance and feature importance
        """
        if results is None or self.important_features is None:
            print("No results or feature importance to visualize")
            return
            
        # Set up plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Plot actual vs predicted values for each model
        for i, (name, result) in enumerate(results.items()):
            ax = axes[0, i]
            ax.scatter(result['y_test'], result['y_pred'], alpha=0.5)
            ax.plot([-10, 10], [-10, 10], 'r--')  # Perfect prediction line
            ax.set_title(f'{name}: Actual vs Predicted {target}')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.text(0.05, 0.95, f"R² = {result['r2']:.4f}\nMSE = {result['mse']:.4f}", 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Plot top features
        top_n = min(15, len(self.important_features))
        ax = axes[1, 0]
        sns.barplot(x='importance', y='feature', data=self.important_features.head(top_n), ax=ax)
        ax.set_title(f'Top {top_n} Important Features')
        ax.set_xlabel('Importance')
        
        # 3. Plot distribution of target variable
        ax = axes[1, 1]
        sns.histplot(self.labeled_data[target], kde=True, ax=ax)
        ax.set_title(f'Distribution of {target}')
        ax.set_xlabel(target)
        
        plt.tight_layout()
        plt.show()
        
    def create_sentiment_lexicon(self, threshold=0.001):
        """
        Create a market-based sentiment lexicon
        """
        if self.model is None or self.important_features is None:
            print("No model or feature importance available")
            return None
            
        print("Creating market-based sentiment lexicon...")
        
        # Get all coefficients for linear regression
        if isinstance(self.model, LinearRegression):
            # Get all feature names
            all_features = self.vectorizer.get_feature_names_out()
            
            # Get all coefficients
            coefficients = self.model.coef_
            
            # Create sentiment dictionary
            sentiment_dict = {}
            
            for feature, coef in zip(all_features, coefficients):
                # Only include features with significant coefficients
                if abs(coef) > threshold:
                    sentiment = 'positive' if coef > 0 else 'negative'
                    sentiment_dict[feature] = {
                        'sentiment': sentiment,
                        'score': coef
                    }
            
            # Create DataFrame
            lexicon = pd.DataFrame.from_dict(sentiment_dict, orient='index')
            lexicon = lexicon.reset_index().rename(columns={'index': 'word'})
            lexicon = lexicon.sort_values('score', ascending=False)
            
            print(f"Created sentiment lexicon with {len(lexicon)} words")
            
            # Show examples
            print("\nTop positive words:")
            print(lexicon[lexicon['sentiment'] == 'positive'].head(10))
            
            print("\nTop negative words:")
            print(lexicon[lexicon['sentiment'] == 'negative'].tail(10))
            
            return lexicon
        elif isinstance(self.model, RandomForestRegressor):
            # Get all feature names
            all_features = self.vectorizer.get_feature_names_out()
            
            # Use SHAP values
            explainer = shap.Explainer(self.model, self.X_train)
            shap_values = explainer(self.X_train)
            # Calculate mean absolute SHAP value per feature
            coefficients = (shap_values.values).mean(axis=0)
            
            # Create sentiment dictionary
            sentiment_dict = {}
            
            for feature, coef in zip(all_features, coefficients):
                # Only include features with significant coefficients
                if abs(coef) > threshold:
                    sentiment = 'positive' if coef > 0 else 'negative'
                    sentiment_dict[feature] = {
                        'sentiment': sentiment,
                        'score': coef
                    }
            
            # Create DataFrame
            lexicon = pd.DataFrame.from_dict(sentiment_dict, orient='index')
            lexicon = lexicon.reset_index().rename(columns={'index': 'word'})
            lexicon = lexicon.sort_values('score', ascending=False)
            
            print(f"Created sentiment lexicon with {len(lexicon)} words")
            
            # Show examples
            print("\nTop positive words:")
            print(lexicon[lexicon['sentiment'] == 'positive'].head(10))
            
            print("\nTop negative words:")
            print(lexicon[lexicon['sentiment'] == 'negative'].head(10))
            
            return lexicon
        else:
            print("Sentiment lexicon creation only implemented for Linear Regression")
            return None
    
    def test_shap(self):
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_train)
        return explainer, shap_values
            
    def run_complete_poc(self, tickers, lookback_days=90, future_days=1, 
                        vectorizer_method='tfidf', max_features=1000):
        """
        Run the complete POC pipeline
        """
        print(f"Starting Market-Based Sentiment Analysis POC with {len(tickers)} tickers")
        
        # 1. Collect data
        self.collect_data(tickers, days_back=lookback_days)
        
        # 2. Merge news with returns
        self.merge_news_with_returns(future_days=future_days)
        
        # 3. Preprocess text
        self.preprocess_text(use_lemmatization=True)
        
        # 4. Vectorize text
        X_df = self.vectorize_text(method=vectorizer_method, max_features=max_features)
        
        # 5. Build model
        results = self.build_model(X_df, target='future_return')
        
        # 6. Visualize results
        self.visualize_results(results)
        shap_explainer, shap_values = self.test_shap()
        # 7. Create sentiment lexicon
        lexicon = self.create_sentiment_lexicon()
        #return shap_explainer, shap_values
        return {
            'labeled_data': self.labeled_data,
            'model': self.model,
            'features': self.important_features,
            'lexicon': lexicon,
            'shap_explainer': shap_explainer,
            'shap_values': shap_values
        }


# COMMAND ----------

del to_date

# COMMAND ----------

time = ((datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))



# COMMAND ----------

datetime.strptime(time, '%Y-%m-%d').strftime('%Y%m%dT%H%M')

# COMMAND ----------

# Example usage
if __name__ == "__main__":
    # Define tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'] # 'MSFT', 'GOOG', 'AMZN', 'META'
    keys = ["cv2ott1r01qqpq6jo0v0cv2ott1r01qqpq6jo0vg", "R6WUON3CB6JRP1K7"] # gmail key: D8GMBK8EBBM99BS4, telia key: NC68XXJ9Z54NFW7O
    # Initialize and run POC
    poc = MarketSentimentPOC()
    poc.set_keys(keys[0], keys[1])
    results = poc.run_complete_poc(tickers, lookback_days=365, future_days=1)
    
    # The results dictionary now contains:
    # - labeled_data: News articles with price movements
    # - model: The best performing model
    # - features: Important features (words) from the model
    # - lexicon: Market-based sentiment lexicon

# COMMAND ----------

display(results["lexicon"])

# COMMAND ----------

results["lexicon"]

# COMMAND ----------

(results["shap_values"].values).mean(axis=0)

# COMMAND ----------

keys = ["cv2ott1r01qqpq6jo0v0cv2ott1r01qqpq6jo0vg", "R6WUON3CB6JRP1K7"] 
poc = MarketSentimentPOC()
poc.set_keys(keys[0], keys[1])

# COMMAND ----------

datetime.strptime("2025-02-15", '%Y-%m-%d').strftime('%Y%m%dT%H%M')

# COMMAND ----------

results = poc.run_complete_poc(tickers, lookback_days=60, future_days=1)

# COMMAND ----------

results["lexicon"]

# COMMAND ----------

price_data = pd.DataFrame()

start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

end_date = datetime.now().strftime('%Y-%m-%d')

stock = yf.Ticker("AAPL")
hist = stock.history(start=start_date, end=end_date)
hist['ticker'] = "AAPL"
price_data = pd.concat([price_data, hist])

# COMMAND ----------

price_data['return'] = ((price_data['Open'] - price_data['Close']) / price_data['Close']) * 100
#((future_price - start_price) / start_price) * 100

# COMMAND ----------

price_data

# COMMAND ----------

def get_finnhub_news(self, ticker, from_date, to_date):
        """Fetch company news from Finnhub API"""
        from_date_str = from_date.strftime('%Y%m%dT%H%M')
        to_date_str = to_date.strftime('%Y%m%dT%H%M')
            
        url = 'https://finnhub.io/api/v1/company-news'
        params = {
            'symbol': ticker,
            'from': from_date_str,
            'to': to_date_str,
            'token': self.finnhub_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  - Finnhub API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"  - Error with Finnhub API: {e}")
            return []