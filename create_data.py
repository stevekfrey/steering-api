import os
import json
import csv

def process_emotion_files(input_dir, output_file, lines_per_file=10):
    flat_data = []

    # Iterate through JSON files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            with open(file_path, 'r') as f:
                # Load the JSON data
                data = json.load(f)
                # Take the first 10 items (or less if the file has fewer items)
                flat_data.extend(data[:lines_per_file])

    # Limit the total number of items to 40
    flat_data = flat_data[:40]

    # Save the combined data to the output file
    output_path = os.path.join(os.path.dirname(input_dir), output_file)
    with open(output_path, 'w') as f:
        json.dump(flat_data, f, indent=2)

    print(f"Combined data saved to {output_path}")

def process_facts_file(input_dir, output_file, max_lines=40):
    facts_data = []
    
    # Path to the facts CSV file
    facts_file = os.path.join(input_dir, 'facts_true_false.csv')
    
    with open(facts_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        
        for row in reader:
            if len(facts_data) >= max_lines:
                break
            
            statement, label = row
            facts_data.append(f"{statement} {label}")

    # Save the facts data to the output file
    output_path = os.path.join(os.path.dirname(input_dir), output_file)
    with open(output_path, 'w') as f:
        json.dump(facts_data, f, indent=2)

    print(f"Facts data saved to {output_path}")

# Set the input directories and output file names
emotions_input_directory = 'data_full/emotions'
emotions_output_filename = 'emotions_small.json'

facts_input_directory = 'data_full/facts'
facts_output_filename = 'facts_small.json'

# Process the emotion files
# process_emotion_files(emotions_input_directory, emotions_output_filename)

# Process the facts file
process_facts_file(facts_input_directory, facts_output_filename)
