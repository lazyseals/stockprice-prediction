from requests import get
import re
from textblob import TextBlob
import pandas as pd
import time

# Performs sentiment analysis on New York Time articles given a concrete keyword
class SentimentAnalyzer():

    def __init__(self, api_key):
        self.api_key = api_key

    # perform request
    def perform_request(self, year, month):
        if self.api_key is not None:
            req = 'https://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}'
            req = req.format(str(year), str(month), self.api_key)
            # return fetched response
            return get(req)

    # match keywords with article id
    def match_id_to_keyword(self, response):
        id_to_keyword = {}
        keywords = []
        for article in response.json()["response"]["docs"]:
            id = article["_id"]
            for keyword_data in article["keywords"]:
                keywords.append(keyword_data["value"].lower())
            id_to_keyword[id] = keywords
            keywords = []
        return id_to_keyword

    # find articles which contain keyword
    def get_article_id(self, queue, id_to_keyword):
        id_with_keyword = []
        for id, keywords in id_to_keyword.items():
            for keyword in keywords:
                if str(queue) in keyword:
                    id_with_keyword.append(id)
        return id_with_keyword

    # get article information for article_id
    def get_article_information_for_queue(self, queue, response, id_to_keyword):
        ids = self.get_article_id(str(queue), id_to_keyword)
        id_to_date = {}
        id_to_snippet = {}
        id_to_headline = {}
        for article in response.json()["response"]["docs"]:
            for id in ids:
                if str(id) == str(article["_id"]):
                    id_to_date[id] = article["pub_date"]
                    id_to_snippet[id] = article["snippet"]
                    id_to_headline[id] = article["headline"]["main"]
                    break
        return (id_to_date, id_to_snippet, id_to_headline)

    def clean_text(self, text):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|"
                               "(\w+:\/\/\S+)", " ", text).split())

    def get_sentiment(self, text):
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_text(text))
        # set sentiment
        return analysis.sentiment.polarity

    # perform sentiment analysis for headline snippets
    def get_sentiments(self, id_to_snippet):
        id_to_sentiment = {}
        for id, snippet in id_to_snippet.items():
            id_to_sentiment[id] = self.get_sentiment(snippet)
        return id_to_sentiment

    def perform_analysis(self, keyword, start_year, start_month, source_file_path, target_file_path):
        year = start_year
        month = start_month

        # For each row add sentiment
        df = pd.read_csv(source_file_path)

        # initial sentiment to 0
        for index, row in df.iterrows():
            df.at[index, 'Sentiment'] = 0

        # add sentiment to dates where news exist
        for index, row in df.iterrows():
            try:
                year_df = int(row['Date'][:4])
                month_df = int(row['Date'][5:7])
                day_df = int(row['Date'][8:10])

                # only perform request when params changed
                if year_df > year or month_df > month:
                    year = year_df
                    month = month_df
                    response = self.perform_request(year, month)
                    id_to_keyword = self.match_id_to_keyword(response)
                    (id_to_date, id_to_snippet, id_to_headline) = self.get_article_information_for_queue(
                        keyword, response, id_to_keyword)
                    id_to_sentiment = self.get_sentiments(id_to_snippet)
                    print('did request')
                    print("iteration year: " + str(year) + " month: " + str(month))

                # add sentiment to df when news exist
                for id, date in id_to_date.items():
                    if day_df == int(date[8:10]):
                        df.at[index, 'Sentiment'] = id_to_sentiment[id]
                        print("sentiment changed")

            # catching error which occurs due to too many requests
            except KeyError:

                try:
                    # wait 1 minute to do the request again due to request limit of api
                    time.sleep(60)

                    # do the request again after 1 minute
                    # no checking for year required because error only occurs when request happens
                    response = self.perform_request(year, month)
                    print(response)
                    id_to_keyword = self.match_id_to_keyword(response)
                    (id_to_date, id_to_snippet, id_to_headline) = self.get_article_information_for_queue(
                        keyword, response, id_to_keyword)
                    id_to_sentiment = self.get_sentiments(id_to_snippet)
                    print('did request')
                    print("iteration year: " + str(year) + " month: " + str(month))

                    # add sentiment to df when news exist
                    for id, date in id_to_date.items():
                        if day_df == int(date[8:10]):
                            df.at[index, 'Sentiment'] = id_to_sentiment[id]
                            print("sentiment changed")

                # catch all other errors which occur inside the KeyError
                except:
                    pass

            # catch all other errors
            except:
                pass


        # return csv file with sentiments
        return df.to_csv(target_file_path)
