import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

#load data from the files
test_file = "WP_test_mod.npy"
test_data = np.load(test_file, allow_pickle=True)
train_file = "WP_train_mod.npy"
train_data = np.load(train_file, allow_pickle=True)

#extract embeddings from train and test data
test_embeddings = [item['embedding'] for item in test_data]
train_embeddings = [item['embedding'] for item in train_data]

updated_test_set = []

#Loop: through each test data point with tqdm for a progress bar
for i, test_embedding in tqdm(enumerate(test_embeddings), desc="Processing test set"):
    #cosine similarities between the test embedding and all train embeddings
    similarities = cosine_similarity([test_embedding], train_embeddings)[0]

    #indices of the top 2 closest embeddings in the training set
    top_indices = np.argsort(similarities)[-2:]

    #information from the closest training set data points
    closest_train_data1 = {key: train_data[top_indices[0]][key] for key in train_data[0].keys()}
    closest_train_data2 = {key: train_data[top_indices[1]][key] for key in train_data[0].keys()}

    #adding the information to the test data point
    test_data_point = {
        'question': test_data[i]['question'],
        'choice_list': test_data[i]['choice_list'],
        'embedding': test_embedding,
        'closest_train_data1': closest_train_data1,
        'closest_train_data2': closest_train_data2
    }
    updated_test_set.append(test_data_point)
    
np.save('updated_WP_test_set.npy', updated_test_set)