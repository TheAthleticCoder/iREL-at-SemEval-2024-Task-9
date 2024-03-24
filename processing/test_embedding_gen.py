import numpy as np
from transformers import BertTokenizer, BertModel
import argparse
from tqdm import tqdm
import torch

#arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using BERT embedding and store in .npy file.")
    parser.add_argument('--input_file', required=True, help='Path to the input .npy file')
    parser.add_argument('--output_file', required=True, help='Path to the output .npy file')
    return parser.parse_args()
args = parse_args()

#load the .npy file
file_path = args.input_file
data = np.load(file_path, allow_pickle=True)

#load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased').to('cuda')

print("\nLoaded BERT model and tokenizer")


#iterating through entries and format them with tqdm
for entry in tqdm(data, desc="Generating responses", unit=" entry"):
    question = entry['question']
    #BERT encoding of the question
    tokens = tokenizer(question, return_tensors='pt').to('cuda')
    with torch.no_grad():
        output = bert_model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    #storing BERT embedding as a new entry in the data
    entry['embedding'] = embedding.tolist() 

#store the data in a new .npy file
output_file_path = args.output_file
np.save(output_file_path, data, allow_pickle=True)

print(f"\nEmbeddings generated and saved to {output_file_path}")

