import json

# sample data
with open('evals/multishot_examples.json', 'r') as file:
    data = json.load(file)

def combine_harmful_entries(data):
    combined_data = []
    for entry in data:
        if 'Harmful' in entry['category']:
            combined_string = f"User: {entry['user']}\nAssistant: {entry['assistant']}"
            combined_data.append(combined_string)
    return combined_data

# run the function and print the result
combined_entries = combine_harmful_entries(data)

# Save combined entries to a text file
with open('evals/combined_harmful_entries.txt', 'w') as output_file:
    for entry in combined_entries:
        output_file.write(entry + '\n')
        output_file.write("---\n")  # separate each entry

print("Combined harmful entries have been saved to 'combined_harmful_entries.txt'")
