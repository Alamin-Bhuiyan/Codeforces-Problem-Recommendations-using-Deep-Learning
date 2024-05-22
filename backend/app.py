import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS extension
import numpy as np
import pandas as pd
import pickle
import time
import requests
from urllib.parse import quote
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

app = Flask(__name__)
CORS(app) 
# Load the pre-trained model
model = load_model(r'C:\Users\HP\OneDrive\Desktop\Thesis\Recommendation Model\Backend\model.h5')

# Load tokenizer and other necessary preprocessing objects if required
with open(r'C:\Users\HP\OneDrive\Desktop\Thesis\Recommendation Model\Backend\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

logging.basicConfig(level=logging.INFO)

def dsconvert(tags):
    return ' '.join(''.join(ch for ch in tag if ch.isalpha()) for tag in tags)

def convert_verdict(verdict):
    return 1 if verdict == "OK" else 0

@app.route('/recommend', methods=['POST'])
def recommend():
    user_handle = request.json.get('handle')
    if not user_handle:
        return jsonify(error="User handle is required"), 404

    # Initialize an empty DataFrame
    Dataset = pd.DataFrame()
    encoded_handle = quote(user_handle)
    
    # Fetch submission data
    submission_url = f'https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=10000'
    submission_request = requests.get(submission_url)
    
    if submission_request.status_code != 200:
        logging.error(f"Failed to get submission data for handle: {user_handle}")
        return jsonify(error="Failed to get submission data"), 404

    try:
        submission_data = submission_request.json()
        if submission_data.get('status') != 'OK':
            logging.error(f"No submission data found for handle: {user_handle}")
            return jsonify(error="No submission data found"), 404

        user_data = pd.DataFrame(submission_data.get('result'))
        user_data['handle'] = user_handle

        time.sleep(2)  # Adding a 2-second delay before the rating request

        # Fetch rating data
        rating_url = f'https://codeforces.com/api/user.rating?handle={encoded_handle}'
        rating_request = requests.get(rating_url)
        if rating_request.status_code != 200:
            logging.error(f"Failed to get rating data for handle: {user_handle}")
            return jsonify(error="Failed to get rating data"), 404

        rating_data = rating_request.json()
        if rating_data.get('status') != 'OK':
            logging.error(f"No rating data found for handle: {user_handle}")
            return jsonify(error="No rating data found"), 404

        rating = pd.DataFrame(rating_data.get('result'))
        user_data['userRating'] = None

        for index, row in user_data.iterrows():
            creation_time = row['creationTimeSeconds']
            filtered_ratings = rating[rating['ratingUpdateTimeSeconds'] < creation_time]
            if not filtered_ratings.empty:
                latest_rating = filtered_ratings.iloc[-1]['newRating']
                user_data.at[index, 'userRating'] = latest_rating

        Dataset = pd.concat([Dataset, user_data], ignore_index=True)

    except ValueError as e:
        logging.error(f"JSON Decode Error: {e}")
        return jsonify(error="Error processing data"), 404

    # Data preprocessing steps
    Dataset[['contestId', 'problemsetName', 'index', 'name', 'type', 'points', 'problemRating', 'tags']] = pd.DataFrame(
        Dataset['problem'].apply(lambda x: [x.get('contestId'), x.get('problemsetName'), x.get('index'), x.get('name'), x.get('type'), x.get('points'), x.get('rating'), x.get('tags')]).tolist()
    )
    Dataset.reset_index(drop=True, inplace=True)
    Dataset = Dataset.fillna(0)
    Dataset['contestId'] = Dataset['contestId'].astype(int)
    Dataset['Problem'] = Dataset['contestId'].astype(str) + Dataset['index'].astype(str) + ' ' + Dataset['name']
    Dataset['attempts'] = Dataset.groupby(['handle', 'Problem'])['handle'].transform('size')
    new_data1 = Dataset.drop_duplicates(subset=['handle', 'Problem'], keep='first')
    Dataset = new_data1.reset_index(drop=True)
    Dataset['userID'] = 1
    Dataset['verdict'] = Dataset['verdict'].apply(convert_verdict)
    Dataset['tags'] = Dataset['tags'].apply(dsconvert)
    Dataset['tags'] = Dataset['tags'].str.split()
    Dataset = Dataset.explode('tags').dropna(subset=['tags']).reset_index(drop=True)
    Dataset['problemRating'] = Dataset['problemRating'].astype(int)
    Dataset['userRating'] = Dataset['userRating'].astype(int)
    Dataset['tags'] = Dataset['tags'].astype(str) + Dataset['problemRating'].astype(str)

    def user_threshold(group):
        tag_counts = {}
        result = []
        for index, row in group.iterrows():
            tag = row['tags']
            if tag not in tag_counts:
                tag_counts[tag] = 0
            if tag_counts[tag] < 1:
                result.append(row)
                tag_counts[tag] += 1
        return pd.DataFrame(result)

    limited_df = user_threshold(Dataset)
    limited_df.reset_index(drop=True, inplace=True)
    Dataset = limited_df
    user_result = Dataset.groupby('userID')['tags'].apply(lambda x: ' '.join(x)).reset_index()
    user_result['tags'] = user_result['tags'].apply(lambda x: ' '.join(x.split()[::-1]))
    text = user_result['tags'].astype(str).str.cat(sep=' ')
    print(text)
    # Generate predictions
    tags_list = []
    for _ in range(10):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=963, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += f" {word}"
                print(word)
                tags_list.append(word)
                break
        #time.sleep(2)

    # Fetch problems from Codeforces API
    problems_url = 'https://codeforces.com/api/problemset.problems'
    problems_request = requests.get(problems_url)
    problems_data = problems_request.json()
    problems_df = pd.json_normalize(problems_data['result']['problems']).fillna(0)
    problems_df['contestId'] = problems_df['contestId'].astype(int)
    problems_df['Problem'] = problems_df['contestId'].astype(str) + problems_df['index'].astype(str) + ' ' + problems_df['name']
    problems_df['tags'] = problems_df['tags'].apply(dsconvert)
    problems_df['tags'] = problems_df['tags'].str.split()
    problems_df = problems_df.explode('tags').dropna(subset=['tags']).reset_index(drop=True)
    problems_df['problemRating'] = problems_df['rating'].astype(int)
    problems_df['tags'] = problems_df['tags'].astype(str) + problems_df['problemRating'].astype(str)

    # Filter problems based on tags_list and exclude already attempted problems
    filtered_problems = problems_df[problems_df['tags'].isin(tags_list)]
    problems_not_in_dataset = filtered_problems[~filtered_problems['Problem'].isin(Dataset['Problem'])]
    unique_problems = problems_not_in_dataset.drop_duplicates(subset=['Problem'])
    recommendations = []
    #print("User Dataset")
    #print(Dataset.head())
    #print("Problem Dataset")
    #print(problems_df.head())
    #print("Filtered Problems")
    #print(filtered_problems.head())
    #print("Not in Dataset")
    #print(problems_not_in_dataset.head())
    #print("Uniques")
    #print(unique_problems.head())
    for index, problem in unique_problems.iterrows():
        link = f'https://codeforces.com/problemset/problem/{problem["contestId"]}/{problem["index"]}'
        recommendations.append({'Problem': problem['Problem'], 'Link': link})
        if len(recommendations) >= 10:
            break

    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
