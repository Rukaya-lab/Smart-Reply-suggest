# Smart-Reply-suggest

#### Problem Statement

- Problem: In today's fast-paced world, it can be difficult to keep up with all of the messages we receive. This can lead to missed opportunities, misunderstandings, and even stress.
- Solution: A smart reply system could help people to save time and improve their communication. By automatically suggesting relevant responses to incoming messages, a smart reply system could free up people to focus on more important tasks.
- Approach: A smart reply system could be implemented using a variety of techniques, including natural language processing (NLP) and long short-term memory (LSTM) models. NLP techniques could be used to identify the key words and phrases in a message, while LSTM models could be used to predict the most likely response.
  - Building Similarity indexes with the ANNOY library.
  - Empolying Hdbscan clustering to cluster similar replies.
  - LSTM since they have memory and able to keep context.
- Benefits: 
  - A smart reply system could offer a number of benefits for people in everyday life, including:
  - Increased productivity: People could save time by not having to type out every response.
  - Improved communication: People could be more confident that their messages are clear and concise.
  - Reduced stress: People could experience less stress from feeling overwhelmed by their communication workload.

### The Data

[Data can be found here](https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat)


This is a Topical Chat dataset from Amazon! It consists of over 8000 conversations and over 184000 messages!

Within each message, there is: A conversation id, which is basically which conversation the message takes place in. Each message is either the start of a conversation or a reply from the previous message. There is also a sentiment, which represents the emotion that the person who sent the message is feeling. There are 8 sentiments: Angry, Curious to Dive Deeper, Disguised, Fearful, Happy, Sad, and Surprised.

### Wrangling
Since the dataset was originally compiled for some other text calssification task, I had to re process and collect only the information that is necessary for the project.

#### Data Preprocessing
1. 
For each row of the dataset, a function checks if the current row is the first row in a conversation. If it is, the function skips to the next row.
If the current row is not the first row in a conversation, the function then checks if the current row and the next row have the same conversation ID. If they do, the function then extracts the message text from both rows and stores them in the input_texts and target_texts lists, respectively.

The function then checks if the input and target texts meet the following criteria:
  - The input text must be at least 3 words long.
  - The target text must be at least 1 word long.
  - The input text must be less than 50 words long.
  - The target text must be less than 10 words long.
  - The input text and target text must both be non-empty strings.
If all of these criteria are met, the function then appends the input and target texts to the input_texts and target_texts lists, respectively.

2. Tokenization of the input and target texts to convert the input and target texts to sequences of integers. 

3. Creating Annoy Index for both the Input and Target Texts
  - Annoy algorithm is to find the nearest neighbors of a given text. The Annoy algorithm is a fast approximate nearest neighbor search algorithm that is based on random projections.
  - created an AnnoyIndex object and specifies the length of the item vectors that will be indexed. The item vectors in this case are the sequences of integers that were created by the tokenizers.
  - Once all of the sequences have been added to the AnnoyIndex object, the code builds the index by creating 100 trees. 
  - A similarity matrix is built for both texts bags that shows the similarity between each pair of texts.

#### Clustering

4. Clustering using the similarity matrix for both input and target texts.
  - Tested both the hdbscan algorithm and dbscan algorithm using different epsilon value while finding the optimal value for the epsilon parameter.

5. Generate padded sequences from a list of input sequences 
  The purpose of this is to prepare the input sequences for training a machine learning model. The padding ensures that all input sequences are the same length, which can improve the performance of the model.

#### Modeling

6. Modeling with LSTM
  The model is a sequential model with the following layers:

    - An embedding layer that converts the input text into a sequence of dense vectors.
    - A long short-term memory (LSTM) layer that learns long-range dependencies in the input text.
    - A dropout layer that randomly drops out some of the neurons in the model, which helps to prevent overfitting.
    - A dense layer that outputs the predicted probability distribution over the possible target labels.

  The model is compiled with the following loss function, optimizer, and metrics:
    - Loss function: categorical crossentropy
    - Optimizer: Adam
    - Metrics: accuracy

#### Result
  The model achieved a loss of 0.1177 and  accuracy of 97%.

Example usage
   Input - bye

    Response 1 -> bye  
    Response 2 -> have a good one  
    Response 3 -> same to you  
    Response 4 -> have a good weekend  
    Response 5 -> have a good evening  
    Response 6 -> good night
    Response 7 -> see ya 
    Response 8 -> take care 


#### Potential Stonewall
  - The similarity index building is quite large and takes time to create hence you need GPU access.
  - Since the texts have been stripped of punctuations in the preprocessing, one needs to find a way to add them back.

#### Next Steps
  - A dataset can be built speciafically for this project that will be robust enough to capture conversation complexities. The current dataset contains some converstaions that are topic specific.
  - I tried to add back the punctation using the rpunct library i found on Hugging face but the libraray hasn't been updated for a while and hence didn't captuer all punctions. So a work could be done focused on that.
  - The clustering method can be improved.
  - Deploying the model.
