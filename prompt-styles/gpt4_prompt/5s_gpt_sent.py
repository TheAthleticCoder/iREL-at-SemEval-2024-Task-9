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
We have given you five examples below to help you understand the puzzle challenge better. To aid your understanding, we've explained our approach and reasoning before presenting the correct option.
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

Example 4:
Question: A thief was walking down the street. He walked past the security guards and also the people whom he had stolen from. Nevertheless, nobody was catching him. So why not?
Choices:
Option 1: He tried to get away from the security and succeeded in the end.
Option 2: At that time, he was just walking down the street. He stolen from others previously.
Option 3: The people whom he had stolen is kind and rich.
Option 4: None of above.
Correct Option: 2
Reason for Correct Option: The key to solving this puzzle is understanding the context and timing of the actions described. The statement implies that the thief is not being pursued or caught at the moment he is walking down the street, which suggests that his thefts occurred at a different time. Thus, his current action (walking down the street) is not directly linked to any immediate crime that would prompt the security guards or the victims to catch him right then.

Given the choices:

"He tried to get away from the security and succeeded in the end." - This option suggests a chase that isn't explicitly mentioned in the puzzle. It also doesn't directly answer why nobody is catching him at the moment.

"At that time, he was just walking down the street. He stolen from others previously." - This option provides a logical explanation that aligns with the information given. It indicates that the act of theft and the act of walking down the street are separate events, which is why he is not being caught in the moment.

"The people whom he had stolen is kind and rich." - This choice introduces an unrelated reason regarding the victims' characteristics, which doesn't directly address the question of why he isn't being caught.

"None of above." - This option would be selected if none of the other options were correct.

Example 5:
Question: A man took his son to a baseball game. The coach of the team saw the man and said, "That boy is my son." Yet, the man at the baseball game was the boy's father. How is that possible?
Choices:
Option 1: The coach of the team is the boy's mother.
Option 2: The boy's father gets a promotion.
Option 3: The boy has two fathers.
Option 4: None of above.
Correct Option: 1
Reason for Correct Option: The puzzle presents a situation where a boy is accompanied by his father to a baseball game, and the coach, upon seeing them, identifies the boy as his son. This situation seems puzzling because it appears there are two fathers claiming the same boy as their son, defying a traditional understanding of family structures. The key to solving this puzzle lies in re-examining our assumptions about gender roles, specifically the gender of the coach.

1. "The coach of the team is the boy's mother." - This option offers a straightforward solution by challenging the often implicit assumption that a coach or a person in a position of authority in sports must be male. If the coach is the boy's mother, it reconciles the apparent contradiction without any need for further complications.

2. "The boy's father gets a promotion." - This option doesn't address the puzzle's central question about the relationship between the boy, the man at the game, and the coach.

3. "The boy has two fathers." - While modern family structures can indeed include a child having two fathers, this option doesn't directly resolve the confusion presented by the puzzle's narrative unless additional context is given to explain the relationship dynamics.

4. "None of above." - This option would be chosen if none of the other answers were correct.

Given the details of the puzzle, the most logical and straightforward solution is option 1: "The coach of the team is the boy's mother." This explanation directly addresses the confusion by clarifying that the coach's gender was the key detail challenging common assumptions, making it the correct choice.

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

