import numpy as np
import google.generativeai as genai
import csv
import argparse
from tqdm import tqdm

# loading the .npy file
def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using Gemini model and store in CSV file.")
    parser.add_argument('--input_file', required=True, help='Path to the input .npy file')
    parser.add_argument('--output_csv', required=True, help='Path to the output CSV file')
    return parser.parse_args()

# parse command-line arguments
args = parse_args()

# load the .npy file
file_path = args.input_file
data = np.load(file_path, allow_pickle=True)

# template
template = """
Welcome to the word-play puzzle challenge! Here, you will be presented with a question based on a word-play puzzle. It means that the question is a brain teaser where the answer violates the default meaning of the word and focuses on the letter composition of the target question.
We have given you three examples below to help you understand the puzzle challenge better. To help you understand better, along with the correct option for the example, we have also provided you with the reason for why the option is correct.
Example 1:
Question: What type of bar doesn't serve alcohol?
Choices:
Option 1: Cocktail bar.
Option 2: Sandbar.
Option 3: Hotel bar.
Option 4: None of above.
Correct Option: 2
Reason for the Correct Option: To solve this sentence-play puzzle, let's analyze each choice based on the information given and the nature of the question, which is a play on words or meanings.

1. *Cocktail bar.* A cocktail bar is known for serving alcohol, as cocktails are alcoholic mixed drinks. Therefore, it doesn't fit the criteria of a bar that doesn't serve alcohol.

2. *Sandbar.* A sandbar is not a place that serves drinks; it's a natural landform made of sand, typically found in water bodies like rivers, oceans, or seas. This option plays on the word "bar" being used in a different context than an establishment serving drinks. It fits the criteria of a "bar" that doesn't serve alcohol because it's not a place of business but a geographical feature.

3. *Hotel bar.* A hotel bar, similar to a cocktail bar, serves alcoholic beverages to its guests. It is a part of a hotel where alcohol is definitely served.

4. *None of above.* This option suggests that all other options serve alcohol, which is not true based on the analysis.

Given the playful nature of the question and analyzing the options, the correct answer is:

Option 2: *Sandbar.*

This is because a sandbar is the only type of "bar" mentioned that doesn't serve alcohol, playing on the dual meaning of the word "bar."

Example 2:
Question: Which side of the turkey has the most feathers?
Choices:
Option 1: The rightside. 
Option 2: The outside.
Option 3: The leftside.
Option 4: None of above.
Correct Option: 2
Reason for the Correct Option: To solve this sentence-play puzzle, let's examine the options given and think about the nature of a turkey, particularly focusing on its feathers. 

1. The right side.
2. The outside.
3. The left side.
4. None of the above.

Upon consideration, the puzzle seems to play on the literal meaning of the sides of a turkey and incorporates a play on words. The key here is to recognize that every turkey, regardless of its orientation, has feathers all around its body. However, the distinction isn't about the orientation (right or left) but rather about where feathers are actually present in a more significant amount.

- Option 1 (The right side) and Option 3 (The left side) suggest a specific side of the turkey, implying that one side might have more feathers than the other, which doesn't make sense anatomically since feathers are distributed all around a turkey's body.

- Option 2 (The outside) is the clever play on words here. It humorously points out that the most feathers are found on the outside of the turkey, which is literally true for all birds.

- Option 4 (None of the above) is incorrect because there is indeed a correct answer among the options.

Therefore, the correct option is:

*Option 2: 'The outside.'* This answer plays on the literal interpretation of where feathers are located on a turkey, making it the side with the most feathers in a manner of speaking.

Example 3:
Question: What type of pot never boils?
Choices:
Option 1: Flowerpot.
Option 2: Teapot.
Option 3: Coffee pot.
Option 4: None of above.
Correct Option: 1
Reason for the Correct Option: The question "What type of pot never boils?" is a classic example of a play on words or a riddle. Let's analyze each option based on common sense and the literal interpretation of the phrase "never boils."

1. *Flowerpot.* A flowerpot is used for planting flowers and does not involve boiling water or any liquid. Therefore, it literally cannot boil.
2. *Teapot.* A teapot is used for brewing tea by steeping tea leaves in hot water. While the water is typically heated to boiling before being placed in the teapot, the teapot itself is not used for boiling water.
3. *Coffee pot.* A coffee pot is used for brewing coffee. Similar to a teapot, the water may be heated to near boiling or boiled before being run through coffee grounds, but the coffee pot itself is not typically used to boil the water.
4. *None of above.*

Given the playful nature of the question, the answer hinges on the interpretation of "never boils." While teapots and coffee pots are associated with hot water, they are not directly used for boiling. The flowerpot, on the other hand, is completely unrelated to the process of boiling water. Therefore, the most accurate answer to the question, considering the literal impossibility of boiling, would be:

Option 1: Flowerpot.

Now, we shall be giving you the puzzle you need to solve. Remember to pay attention to the details mentioned and indicate the option number you believe is correct for the question:
Question: {}
Choices:\n{}
"""

# list to store formatted entries
formatted_data = []

# iterating through entries and format them
for entry in tqdm(data, desc="Formatting Entries", unit="entry"):
    question = entry['question']
    choices = entry['choice_list']
    #t1
    t1_question = entry['closest_train_data1']['question']
    t1_choices = entry['closest_train_data1']['choice_list']
    t1_label = entry['closest_train_data1']['label']
    t1_label += 1 #to make it 1-indexed
    t1_reason = entry['closest_train_data1']['reason']
    #t2
    t2_question = entry['closest_train_data2']['question']
    t2_choices = entry['closest_train_data2']['choice_list']
    t2_label = entry['closest_train_data2']['label']
    t2_label += 1 #to make it 1-indexed
    t2_reason = entry['closest_train_data2']['reason']
    #t3
    t3_question = entry['closest_train_data3']['question']
    t3_choices = entry['closest_train_data3']['choice_list']
    t3_label = entry['closest_train_data3']['label']
    t3_label += 1 #to make it 1-indexed
    t3_reason = entry['closest_train_data3']['reason']
    #t4
    t4_question = entry['closest_train_data4']['question']
    t4_choices = entry['closest_train_data4']['choice_list']
    t4_label = entry['closest_train_data4']['label']
    t4_label += 1 #to make it 1-indexed
    t4_reason = entry['closest_train_data4']['reason']
    #t5
    t5_question = entry['closest_train_data5']['question']
    t5_choices = entry['closest_train_data5']['choice_list']
    t5_label = entry['closest_train_data5']['label']
    t5_label += 1 #to make it 1-indexed
    t5_reason = entry['closest_train_data5']['reason']
    # format the choices with options and newline characters
    formatted_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(choices, start=1)])
    formatted_t1_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(t1_choices, start=1)])
    formatted_t2_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(t2_choices, start=1)])
    formatted_t3_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(t3_choices, start=1)])
    formatted_t4_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(t4_choices, start=1)])
    formatted_t5_choices = '\n'.join([f"Option {i}: {choice}" for i, choice in enumerate(t5_choices, start=1)])
    # format the entry using the template
    formatted_entry = template.format(question, formatted_choices)
    # append the formatted entry to the list
    formatted_data.append(formatted_entry)

print("Data Loading Complete\n")
# initialize the generative model
import google.generativeai as genai
genai.configure(api_key="AIzaSyBu3mT8ni1L07S3Jm0YFIduJRISf-rThvs")
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
print("Model Loading Completed\n")
# list to store input and output for CSV
output_data = []

# iterate through formatted data and generate responses
for i, prompt in enumerate(tqdm(formatted_data, desc="Generating Responses", unit="entry"), start=1):
    # run the gemini model on the input
    response = model.generate_content(prompt)
    # append input and output to the list
    input_output_pair = {'input': prompt, 'output': response.text}
    output_data.append(input_output_pair)

# storing the output in a CSV file
csv_file_path = args.output_csv
csv_columns = ['input', 'output']

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    for pair in output_data:
        writer.writerow(pair)

print(f"\nCSV file with input-output pairs saved at: {csv_file_path}")

