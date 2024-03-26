import csv
import re
import argparse

def extract_first_number(text):
    # Use regular expression to find the first number in the text
    match = re.search(r'\d+', text)
    return int(match.group()) - 1 if match else None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract the first number from the 'output' column in a CSV file and store in a text file.")
    parser.add_argument('--csv_file', required=True, help='Path to the CSV file')
    parser.add_argument('--txt_file', required=True, help='Path to the output text file')
    args = parser.parse_args()

    # List to store extracted numbers
    extracted_numbers = []

    try:
        # Read the CSV file
        with open(args.csv_file, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Extract the first number from the 'output' column and subtract 1
                output_text = row['output']
                number = extract_first_number(output_text)
                if number is not None:
                    extracted_numbers.append(number)

        # Write the extracted numbers (subtracted by 1) to a text file
        with open(args.txt_file, 'w') as txt_file:
            for number in extracted_numbers:
                txt_file.write(f"{number}\n")

        print(f"\nNumbers (subtracted by 1) extracted from the 'output' column and saved to {args.txt_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    
#python eval.py --csv_file SP_out.csv --txt_file answer_sen.txt
#python eval.py --csv_file WP_out.csv --txt_file answer_word.txt
