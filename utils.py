import os

def file_path_gen(base, source_suffixes, excluded_patterns):
    def is_excluded(file_path):
        from fnmatch import fnmatch
        for pattern in excluded_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    for root, _, files in os.walk(base):
        for file in files:
            if not file.endswith(tuple(source_suffixes)):
                continue
            abs_path = os.path.join(root, file)
            file_path = os.path.relpath(abs_path, base)
            if is_excluded(file_path):
                continue
            yield file_path