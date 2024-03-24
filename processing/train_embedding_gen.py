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

#loading the .npy file
file_path = args.input_file
data = np.load(file_path, allow_pickle=True)

#load gemini model
import google.generativeai as genai
genai.configure(api_key="")
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]
generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings)

#loading BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased').to('cuda')

print("\nLoaded Gemini and BERT model and tokenizer")

#sentence puzzle template
template = """
### CONTEXT
We are presented with a question based on a sentence-play puzzle. It means that the question is a sentence-type brain teaser where the puzzle defying commonsense is centered on sentence snippets. 

### OBJECTIVE
We have provided you below with the question, the answer choices and the correct option number and choice. 
We have also provided you with what the distractor choice was. This distractor was aimed to throw you off the correct answer.
You need to provide the reasoning for why the option is correct.
Question: {}
Choices:\n{}
Correct Option: Option {} : {}
Distractor Choice: {}

### RESPONSE
Provide the reasoning for why the given correct option is correct and what should be taken care of so that the distractor choice is not chosen.

### REASONING
"""

#word puzzle template
template = """
### CONTEXT
We are presented with a question based on a word-play puzzle. It means that the question is a brain teaser where the answer violates the default meaning of the word and focuses on the letter composition of the target question. 

### OBJECTIVE
We have provided you below with the question, the answer choices and the correct option number and choice. 
We have also provided you with what the distractor choice was. This distractor was aimed to throw you off the correct answer.
You need to provide the reasoning for why the option is correct.
Question: {}
Choices:\n{}
Correct Option: Option {} : {}
Distractor Choice: {}

### RESPONSE
Provide the reasoning for why the given correct option is correct and what should be taken care of so that the distractor choice is not chosen.

### REASONING
"""

print("Storing Embeddings and Responses...\n")
filtered_data = []
# iterate through entries and format them with tqdm
for entry in tqdm(data, desc="Generating responses", unit=" entry"):
    question = entry['question']
    choices = entry['choice_list']
    label = entry['label']
    correct_option = label + 1
    answer = entry['answer']
    distractor = entry['distractor1']
    #format the choices with options and newline characters
    formatted_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(choices, start=1)])
    #format the entry using the template
    formatted_entry = template.format(question, formatted_choices, correct_option, answer, distractor)

    try:
        #generate Gemini model response
        response = model.generate_content(formatted_entry)
        #storing Gemini response as a new entry in the data
        entry['reason'] = response.text
    except Exception as e:
        #skip entry
        print(f"Exception occurred for entry:\n{formatted_entry}\nError: {e}")
        continue

    tokens = tokenizer(question, return_tensors='pt').to('cuda')
    with torch.no_grad():
        output = bert_model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    entry['embedding'] = embedding.tolist()

    #add the entry to the filtered data
    filtered_data.append(entry)

#convert the filtered data to a new NumPy array
filtered_data_np = np.array(filtered_data)

#store the data in a new .npy file
output_file_path = args.output_file
np.save(output_file_path, filtered_data_np, allow_pickle=True)

print(f"\nResponses generated and saved to {output_file_path}")
