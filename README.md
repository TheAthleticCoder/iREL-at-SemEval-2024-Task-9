# iREL at SemEval-2024 Task 9: Improving Conventional Prompting Methods for Brain Teasers

**This repo contain the code for our approach towards SemEval-2024 Task 9: BRAINTEASER: A Novel Task Defying Common Sense.** The BRAINTEASER task comprises multiple-choice Question Answering, designed to evaluate the models’ lateral thinking capabilities. It consists of Sentence Puzzle and Word Puzzle subtasks that require models to defy default common-sense associations and exhibit unconventional thinking. We propose a unique strategy aimed at improving the performance of pre-trained language models, particularly the Gemini 1.0 Pro Model, in both subtasks. We employ static and dynamic few-shot prompting techniques and introduce a model-generated reasoning strategy that utilizes the LLM’s reasoning capabilities to improve its performance. Our approach demonstrated significant improvements, showing that it performed better than the baseline models by a significant margin but fell short of performing as well as the human annotators, thus highlighting the efficacy of the proposed strategies.

---

### Final Data Files

After applying Contextualised Example Selection and Self-Generated Reasoning, the final modified test set is given in the folder `final_data_files`.
The dictionary keys after reading each file are given below:

```py
dict_keys(['question', 'choice_list', 'embedding', 'closest_train_data1', 'closest_train_data2', 'closest_train_data3', 'closest_train_data4', 'closest_train_data5'])
```

---

### File Structure

For the folder `processing`:

1. `train_embedding_gen.py`: This code generates BERT embeddings for all the questions in the training data. It also generates reasoning using the Google Gemini Pro 1 Model for why the valid option is correct for the particular brain teaser (part of Self-Generated Reasoning). You can execute the code using:

```bash
python train_embedding_gen.py --input_file {Path to the input .npy file} --output_file {Path to the output .npy file}
```

2. `test_embedding_gen.py`: This code generates BERT embeddings for all the questions in the test data. You can execute the code using:

```bash
python test_embedding_gen.py --input_file {Path to the input .npy file} --output_file {Path to the output .npy file}
```

3. `reason_mapper.py`: Code that applies cosine similarity over the BERT embeddings to find the top 5 most contextually similar questions from the training data for each data point in the testing data.
4. `gen_final_sub.py`: The code file organizes the output data into the required submission format for the evaluation script for the shared task.

For the folder `experiments`:
It contains the code for all the experiments with different approaches and methodologies. Each code file contains the prompt template used during the experiment. Remember to give your Gemini API key to the model in `genai.configure(api_key="")` before executing the code. The code file can be executed by:

```bash
python {code file} --input_file {Path to the input .npy file} --output_csv {Path to the output CSV file}
```

---

### Semeval Task Description Paper

```
@inproceedings{jiang-semeval-2024-brainteaser, title = "SemEval-2024 Task 9: BRAINTEASER: A Novel Task Defying Common Sense", author = "Jiang, Yifan and Ilievski, Filip and Ma, Kaixin", booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation", year = "2024", publisher = "Association for Computational Linguistics"}
```

---
