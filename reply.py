import hdbscan
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from keras import models  


#smart reply

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    #print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()

def get_responses(seed_text, n):
    #responses = list()


    tokenizer = pickle.load(open(f'{SMART_REPLY_MODEL_DIRECTORY}/tokenizer.pickle', 'rb'))
    hdbscan= pickle.load(open(f'{SMART_REPLY_MODEL_DIRECTORY}/target_hdbscan_auto.pickle', 'rb')) #change to test
    output_texts= pickle.load(open(f'{SMART_REPLY_MODEL_DIRECTORY}/target_texts.pickle', 'rb'))
   
    max_sequence_len = 49
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    with strategy.scope():
        model = models.Sequential()
        model = tf.keras.models.load_model(f'{SMART_REPLY_MODEL_DIRECTORY}/sreply_lstmhdbscan.h5')#change to test
        predictions = model.predict(token_list, verbose=10)
        #print(predictions)
        predicted_indices = predictions.argsort()[0][::-1][:n]
        
    #print(predicted_indices)

    responses = []
    for predicted_index in predicted_indices:
        score = 0
        if predicted_index == len(set(hdbscan.labels_)) - 1: #edit point
            predicted_index = -1
        else:
            score = predictions[0][predicted_index]
       
        # randomly pick 1 index
        possible_response = np.where(hdbscan.labels_== predicted_index)[0]  #edit point
        
        response_index = random.sample(possible_response.tolist(), 5)
        # responses.append([output_texts[response_index].replace("\t", "").replace("\n", ""), score])
        response = [[output_texts[item].replace("\t", "").replace("\n", ""), score] for item in response_index]
        responses.extend(response)

    #print(responses)
    #suggested_texts=[]
    #for response in responses:
        #suggested_text = {"response": response[0], "score": response[1] }
    #     #print(suggested_text)
        #suggested_texts.extend(suggested_text)
    #print(suggested_text)
    return responses
  
  
  
  
#auto suggest
def suggest_text_response():
    text_data : dict = request.get_json()
    text = text_data['text']
    
    #if len(text.split()) <= 3:
        
     #restricting to 2 words
    suggested_responses = get_responses(text, 3) 
    suggested_responses.sort(key=lambda x: x[1], reverse=True)
       

    upper_threshold = 0.8
    lower_threshold = 0.005
    
    #print(suggested_responses)
    # lst = [item['response'] for item in suggested_responses if item['score'] < upper_threshold and item['score'] > lower_threshold]
    lst = [item[0] for item in suggested_responses if item[1] < upper_threshold and item[1] > lower_threshold]
    lst = list(set(lst))
    
    rpunct = RestorePuncts()

    punct_lst = [rpunct.punctuate(item, lang = 'en') for item in lst]
        
    response = JsonResponse(data= punct_lst , message='quick replies has been successfully suggested').get()

    #else:
        #response = JsonResponse(data =[] , message='quick replies has been successfully suggested').get()

    return response
