from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline
import numpy as np
from transformers import DebertaV2Tokenizer


# absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_tokenizer = DebertaV2Tokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1", use_fast=False)
absa_model = AutoModelForSequenceClassification \
  .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")


sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                          tokenizer=sentiment_model_path)

# sentence = "We had a great experience at the restaurant, food was delicious, but " \
#   "the service was kinda bad"
# print(f"Sentence: {sentence}")
# print()

# aspect = "food"
# inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
# outputs = absa_model(**inputs)
# probs = F.softmax(outputs.logits, dim=1)
# probs = probs.detach().numpy()[0]
# print(f"Sentiment of aspect '{aspect}' is:")
# for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#   print(f"Label {label}: {prob}")
# print()


# aspect = "service"
# inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
# outputs = absa_model(**inputs)
# probs = F.softmax(outputs.logits, dim=1)
# probs = probs.detach().numpy()[0]
# print(f"Sentiment of aspect '{aspect}' is:")
# for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#   print(f"Label {label}: {prob}")
# print()

# sentiment = sentiment_model([sentence])[0]
# print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")

def sentimeter(sentence, aspect):
  inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
  outputs = absa_model(**inputs)
  probs = F.softmax(outputs.logits, dim=1)
  probs = probs.detach().numpy()[0]
  max_index = np.argmax(probs)
  max_value = probs[max_index]
  labels = ["negative", "neutral", "positive"]
  max_label = labels[max_index]
  #return max_value, max_label 
  return probs[0], probs[1], probs[2] # -- this does all 3 probs which are in order of negative, neutiral, positive.

print(sentimeter("This product was decent and mid", "product"))