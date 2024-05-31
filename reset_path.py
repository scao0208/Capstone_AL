import os
# Specify the cycle variable (assuming it's predefined or passed as an argument)
CYCLE = 14  # Example value, set it as needed

for cycle in range(CYCLE):
    # Define the input file path using the cycle variable
    input_file_path = f'/home/siyu/Documents/al/scapstone/first_mAP50/batch_{cycle}.txt'

    if not os.path.isdir("mAP50_path"):
        os.mkdir("mAP50_path")
    # Define the output file path
    output_file_path = f'/home/siyu/Documents/al/scapstone/mAP50_path/batch_{cycle}.txt'

    # Initialize an empty list to store the extracted parts
    samples = []

    # Open and read the input file
    with open(input_file_path, 'r') as f:
        # Read all lines from the file
        lines = f.readlines()

        # Process each line
        for line in lines:
            # Remove any trailing newline or space characters
            line = line.strip()
            # Split the line by "_" and take the first part
            extracted_part = line.split('_')[-1]
            base_dir = '/home/siyu/Documents/al/scapstone'
            result = os.path.relpath(extracted_part, base_dir)
        
            # Append the extracted part to the samples list
            samples.append(result)

    # Write the samples to the output file
    with open(output_file_path, 'a') as f:
        for sample in samples:
            # f.write('./' + sample + '\n')
            f.write('./' + sample + '\n')
    # Print a message indicating that the process is complete
    print(f'Samples have been written to {output_file_path}')
