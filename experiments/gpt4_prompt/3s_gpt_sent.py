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
Welcome to the sentence-play puzzle challenge! Here, you will be presented with a question based on a sentence-play puzzle. It means that the question is a sentence-type brain teaser where the puzzle-defying commonsense is centred on sentence snippets.
We have given you three examples below to help you understand the puzzle challenge better. To aid your understanding, we've explained our approach and reasoning before presenting the correct option.
Example 1:
Question: An electric train is going south at 98 mph. The wind is blowing northeast. Which direction is the smoke blowing?
Choices:
Option 1: Northeast.
Option 2: West.
Option 3: South.
Option 4: None of above.
Correct Option: 4
Reason for Correct Option: To solve this sentence-play puzzle, let's analyze the key elements of the question:

Electric Train: This is a crucial piece of information. Electric trains do not produce smoke as they do not burn fuel in the same way that steam or diesel trains do.

Direction of Train: The train is going south. This would only be relevant if we were dealing with a train that produces smoke.

Wind Direction: The wind is blowing northeast. Normally, this would affect the direction of the smoke if the train produced any.

Given these points, the key detail here is that the train is electric. Therefore, the direction of the smoke is a trick question because there would be no smoke produced by an electric train.

Example 2:
Question: Two mothers and two daughters go shopping. They equally split the $21 they have between them. How on earth is this even possible?
Choices:
Option 1: One mother gave her dollars to two girls.
Option 2: Seven dollar is enough for two girls.
Option 3: The group included a grandmother, her daughter and her daughter's daughter.
Option 4: None of above.
Correct Option: 3
Reason for Correct Option: To solve this puzzle, let's analyze the given choices by considering the information provided in the question: "Two mothers and two daughters go shopping. They equally split the $21 they have between them."

For the group to split $21 equally, each person must get an equal share. Since $21 is divisible by 3, it suggests that there are actually three people in the group, not four, because each person would receive $7.

Option 1 ("One mother gave her dollars to two girls.") doesn't necessarily explain how $21 can be equally split among them, especially if we're considering the group to consist of only two mothers and two daughters specifically.

Option 2 ("Seven dollar is enough for two girls.") does not address the puzzle's requirement of how $21 is split equally among them, nor does it consider the group's composition.

Option 3 ("The group included a grandmother, her daughter and her daughter's daughter.") provides a plausible explanation. In this scenario, the grandmother is both a mother (to her daughter) and a mother-in-law (or just in the maternal line to the granddaughter, depending on interpretation), the daughter is both a mother (to her daughter) and a daughter (to her mother), and the granddaughter is a daughter. This configuration allows for there to be two mothers and two daughters, fitting the description, and there are three people in the group, allowing the $21 to be equally split into three parts of $7 each.

Option 4 ("None of above.") is incorrect because option 3 offers a valid explanation of how the scenario is possible.

Given this analysis, the correct answer is:

Option 3: "The group included a grandmother, her daughter, and her daughter's daughter." This explanation accounts for the presence of two mothers and two daughters, while also explaining how $21 could be equally split between them, with each receiving $7.

Example 3:
Question: I buried a stone in the ground, and it grew and produced fruit. How is this possible?
Choices:
Option 1: The whether condition is pleasant.
Option 2: The ground is close to a lake.
Option 3: The stone was actually a seed, not an ordinary stone. The seed of a cherry is called a stone.
Option 4: None of above.
Correct Option: 3
Reason for Correct Option: The puzzle presents an interesting play on words and meanings, particularly focusing on the statement "I buried a stone in the ground, and it grew and produced fruit." At first glance, this seems impossible since stones do not grow or produce fruit. However, by examining the choices provided, we can find the logical explanation.

1. *The whether condition is pleasant.* - While pleasant weather conditions are beneficial for growth, they cannot make a stone grow or produce fruit.
2. *The ground is close to a lake.* - Proximity to a lake would indeed provide ample water, which is essential for growth, but again, it cannot make a stone grow or produce fruit.
3. *The stone was actually a seed, not an ordinary stone. The seed of a cherry is called a stone.* - This choice offers a plausible explanation. In botany, the hard, inner part of some fruits, such as cherries, peaches, plums, and apricots, is called a "stone." These stones contain the seed of the fruit. When such a "stone" is planted or buried in the ground, it can indeed grow into a new plant and eventually produce fruit, assuming the conditions are right for its growth.

Given the information and the choices provided, the correct option that explains how a "stone" could be buried, grow, and produce fruit is:

3. *The stone was actually a seed, not an ordinary stone. The seed of a cherry is called a stone.*

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

