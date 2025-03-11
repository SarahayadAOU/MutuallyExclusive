import csv
from transformers import pipeline

# Load pre-trained NLI model (Natural Language Inference)
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# Path to the input CSV file
csv_file_path = r"C:\Users\ayads\Downloads\Results_all\Results_all\Generated_gateway_names_part_10.csv"

# Path to the output CSV file
output_csv_file_path = r"C:\Users\ayads\Downloads\Results_all\Results_all\gateway_names_results10.csv"

# Open the output CSV file to write the results
with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as output_file:
    # Define the CSV writer and write the header
    writer = csv.writer(output_file)
    writer.writerow(["Sentence 1", "Sentence 2", "Relationship", "Confidence", "Mutual Exclusivity"])

    # Open the input CSV file and process each row
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Loop through each row in the CSV
        for row in reader:
            # Extract the 'Gateway Names' column and split by ','
            gateway_names = row['Gateway Names'].split(',')

            # Ensure there are at least two phrases (sentence_1 and sentence_2)
            if len(gateway_names) >= 2:
                sentence_1 = gateway_names[0].strip()  # First part
                sentence_2 = gateway_names[1].strip()  # Second part

                # Classify the relationship between the two sentences
                result = nli_model(f"{sentence_1} </s> {sentence_2}")

                # Get the label and confidence score
                label = result[0]['label']
                score = result[0]['score']

                # Determine if the sentences are mutually exclusive
                mutual_exclusivity = "Mutually Exclusive" if label.lower() == "contradiction" else "Not Mutually Exclusive"

                # Write the result to the CSV file
                writer.writerow([sentence_1, sentence_2, label, score, mutual_exclusivity])
            else:
                # Skip the row if there aren't two phrases to compare
                print(f"Skipping row due to insufficient phrases in 'Gateway Names': {row['Gateway Names']}")
