#!/bin/bash

# Set the maximum number of concurrent processes
max_processes=4

# Counter for running processes
running_processes=0

# Loop through padding values from 0.1 to 0.5 with 0.1 increments
for padding in $(seq 0.1 0.1 0.5); do
    echo "Starting scripts with padding = $padding"

    # Run each Python script with the `padding` argument in the background
    for script in c22_seq.py CIF_seq.py MiniRocket_seq.py WEASEL_seq.py; do
        # Check if the maximum number of processes is reached
        if ((running_processes >= max_processes)); then
            # Wait for one of the background processes to finish
            wait -n
            # Decrement the running process counter
            ((running_processes--))
        fi

        # Execute the Python script with the `padding` argument in the background
        python3 "$script" --padding $padding &
        
        # Increment the running process counter
        ((running_processes++))
    done
done

# Wait for any remaining background processes to finish
wait

echo "All scripts have finished execution for all padding values."
