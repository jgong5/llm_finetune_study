import argparse
import os

def generate_prompt(folder_path, file_suffixes):
    prompt = """You are a software architect and an expert in programming. Please describe the functionality for the given programming code.
Please start with a high-level summary for functionality of all the given source files.
After that, the description should contain the functional description for each class or module and key functions or methods of it that are defined in the code.
It should also describe the relationship to other classes or modules. If a class or module is not defined in the code, please also guess its functionality.
The code content is laid out for each program file as follows. There could be multiple files:
FILENAME: {file name}
CODE:
```
{code content}
```

"""

    def enumerate_files(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                yield (dirpath, filename)

    # List all files in the folder
    files = [(dirpath, filename) for dirpath, filename in enumerate_files(folder_path) if any(filename.endswith(suffix) for suffix in file_suffixes)]

    for dirpath, filename in files:
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as file:
            file_content = file.read()

        prompt += f"FILENAME: {filename}\nCODE:\n```\n{file_content}\n```\n\n"

    return prompt

def main():
    parser = argparse.ArgumentParser(description='Generate a prompt based on files in a folder.')
    parser.add_argument('folder_path', help='Path to the folder containing the files')
    parser.add_argument('file_suffixes', nargs='+', help='File suffixes to filter files (e.g., ".py .java")')

    args = parser.parse_args()

    resulting_prompt = generate_prompt(args.folder_path, args.file_suffixes)
    print(resulting_prompt)

if __name__ == "__main__":
    main()