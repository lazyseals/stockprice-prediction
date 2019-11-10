from SentimentAnalysis import SentimentAnalyzer
from Evaluation import Evaluator

api_key = 's6J10zzMfa9Vm5q1AufUu4gTovcVJEmo'

'''
# perform Sentiment Analysis
sa = SentimentAnalyzer(api_key)
target_file = sa.perform_analysis('dow jones', 1980, 11, "../data/indices/DJI.csv", 
                                  "../data/indices/DJI_sentiments.csv")
'''

# model evaluation
ev = Evaluator(model_path='../models/normalized_sentiments.h5',
               eval_data_path='../data/dow_jones_stocks/sentiments/dataset_3/NKE.csv',
               scaler_path='../models/normalized_sentiments_scaler.save')
ev.create_prediction_data(normalize=True)
print("Market Return: {}\nModel Return: {}".format(ev.market_return(), ev.model_return(kapital=100)))
ev.plot_predictions()
