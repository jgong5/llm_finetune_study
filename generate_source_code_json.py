import os
from utils import file_path_gen

def main():
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description='generate source code json configuration file')
    parser.add_argument('--project-dir', type=str, default='.', help='project directory')
    parser.add_argument('--suffixes', type=str, nargs='+', default=['.py', '.c', '.cpp', '.h', '.hpp'], help='source file suffixes')
    parser.add_argument('--excludes', type=str, nargs='+', default=['build/*', 'third_party/*', 'test/*'], help='excluded patterns')
    parser.add_argument('output', type=str, help='output json file name')

    args = parser.parse_args()

    project_directory = os.path.abspath(args.project_dir)
    source_suffixes = args.suffixes
    excluded_patterns = args.excludes

    # Traverse the project directory and generate json configuration file with following format
    # {"project_base": "...", "file_paths": ["...", "...", ...]}
    import json
    config = {}
    config["project_base"] = project_directory
    config["file_paths"] = list(file_path_gen(project_directory, source_suffixes, excluded_patterns))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()