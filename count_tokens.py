from collections import defaultdict
import os
from transformers import AutoTokenizer
import os

def count_tokens(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        tokens = tokenizer.encode(content)
        return len(tokens)

def main():
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description='count tokens')
    parser.add_argument('--project-dir', type=str, default='.', help='project directory')

    args = parser.parse_args()

    project_directory = args.project_dir

    # Define the file suffixes for source files
    source_suffixes = [".py", ".c", ".cpp", ".h", ".hpp"]

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Specify the bins for the histogram
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, float('inf')]

    # Use defaultdict to store the distribution of token counts
    token_distribution = defaultdict(int)

    # Traverse the project directory
    for root, _, files in os.walk(project_directory):
        for file in files:
            if file.endswith(tuple(source_suffixes)):
                abs_path = os.path.join(root, file)
                file_path = os.path.relpath(abs_path, project_directory)
                num_tokens = count_tokens(abs_path, tokenizer)
                print(f"{file_path}: {num_tokens}")

                # Update the distribution
                for i in range(len(bins) - 1):
                    if bins[i] <= num_tokens < bins[i + 1]:
                        token_distribution[bins[i]] += 1
                        break

    # Print the distribution of token counts using a histogram
    print("\nDistribution of Token Counts:")
    for bin_start, count in sorted(token_distribution.items()):
        bin_end = bins[bins.index(bin_start) + 1] if bin_start != float('inf') else float('inf')
        print(f"{bin_start}-{bin_end} tokens: {'*' * count} ({count} files)")

if __name__ == "__main__":
    main()