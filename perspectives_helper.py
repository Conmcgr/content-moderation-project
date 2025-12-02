import os
import json
import time

import numpy as np
from dotenv import load_dotenv
from googleapiclient import discovery

load_dotenv()
API_KEY = os.getenv("PERSPECTIVES_API_KEY")

print(API_KEY)

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
]

def predict_single_comment(comment):
    #Predict classification of a single comment
    analyze_request = {
        "comment": {"text": comment},
        "languages": ["en"],
        "requestedAttributes": {attr: {} for attr in ATTRIBUTES},
        "doNotStore": True,
    }

    response = client.comments().analyze(body=analyze_request).execute()

    scores = {}
    for attr in ATTRIBUTES:
        scores[attr] = response["attributeScores"][attr]["summaryScore"]["value"]
    
    return translate_scores(scores)

def softmax(x):
    #Implementation of softmax function
    e = np.exp(x - np.max(x))
    return e / e.sum()

def translate_scores(scores):
    #Function to translate the perspectives scores for given attributes into 0, 1, 2 severity scale from dataset
    hate_score = max(
        scores["SEVERE_TOXICITY"],
        scores["IDENTITY_ATTACK"],
        scores["THREAT"],
    )

    offensive_score = max(
        scores["TOXICITY"],
        scores["INSULT"],
        scores["PROFANITY"],
    )

    safe_score = 1 - max(hate_score, offensive_score)

    p_hate, p_offensive, p_safe = softmax(
        np.array([hate_score, offensive_score, safe_score])
    )

    predicted_label = 2

    if p_hate == max(p_hate, p_offensive, p_safe):
        predicted_label = 0
    elif p_offensive == max(p_hate, p_offensive, p_safe):
        predicted_label = 1
    else:
        predicted_label = 2 
    
    return predicted_label


def predict_batch(comment_list):
    #Run prediction on a batch (list of comments)
    output = []

    for comment in comment_list:
        output.append(predict_single_comment(comment))
        time.sleep(1.5)
    
    return output