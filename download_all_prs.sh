#!/bin/bash

# Set the owner and repo name
owner="pytorch"
repo="pytorch"

# Set the state of the pull requests
state="all"

# Set the output file name
output="$repo.prs.json"

# Initialize the page number and the result count
page=1
count=0

# Loop until there are no more results
while true; do
  # Call the gh api with the parameters
  result=$(gh api /repos/$owner/$repo/pulls?state=$state\&per_page=100\&page=$page)
  
  # Check if the result is "[]"
  if [ "$result" == "[]" ]; then
    # Break the loop
    break
  fi
  
  # Append the result to the output file
  echo "$result" | jq . | tee -a $output
  
  # Increment the page number
  page=$((page + 1))
  
  # Increment the result count by the length of the result array
  count=$((count + $(echo "$result" | jq length)))
done

# Print the total number of pull requests
echo "Downloaded $count pull requests from $owner/$repo"