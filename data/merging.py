import csv

# Specify your input files and the output file name
# We changed the names each time needed
file1 = 'validation_merged_v4_720.csv'
file2 = 'processed_weather/H=720/validation.csv'
merged_file = 'validation_merged_final720.csv'

# Open the files
with open(file1, 'r', newline='') as f1, open(file2, 'r', newline='') as f2, open(merged_file, 'w', newline='') as out:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(out)
    
    # Write all rows from the first file (including header)
    for row in reader1:
        writer.writerow(row)
    
    # Skip the header in the second file
    next(reader2, None)
    
    # Write the remaining rows from the second file
    for row in reader2:
        writer.writerow(row)

print(f"Merged CSV file created: {merged_file}")
