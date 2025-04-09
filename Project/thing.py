# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
# from transformers import pipeline
# import numpy as np
# from transformers import DebertaV2Tokenizer


# # absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# absa_tokenizer = DebertaV2Tokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1", use_fast=False)
# absa_model = AutoModelForSequenceClassification \
#   .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# def sentimeter(sentence, aspect):
#   inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
#   outputs = absa_model(**inputs)
#   probs = F.softmax(outputs.logits, dim=1)
#   probs = probs.detach().numpy()[0]
#   max_index = np.argmax(probs)
#   max_value = probs[max_index]
#   labels = ["negative", "neutral", "positive"]
#   max_label = labels[max_index]
#   return max_value, max_label 
#   #return probs[0], probs[1], probs[2] # -- this does all 3 probs which are in order of negative, neutral, positive.

# print(sentimeter("I got the phone", "phone"))
# #"I got the phone", "phone"
# #"I really liked the phone but I really hated how the screen was", "screen" or "phone"

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

def sentimeter(sentence):
    scores = vader_analyzer.polarity_scores(sentence)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    return compound, label


print(sentimeter("Solid phone for the price!"))
