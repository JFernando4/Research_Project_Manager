import argparse
import os
import zipfile

from mlproj_manager.file_management import concatenate_results, get_file_paths_that_contain_keywords, get_indices


def read_arguments():

    arguments = argparse.ArgumentParser()
    arguments.add_argument("--results-dir", action="store", type=str, required=True, help="Path to a directory.")
    arguments.add_argument("--skip-list", action="store", type=str, default="none",
                           help="Comma separated list with names of directories to skip.")
    arguments.add_argument("--verbose", action="store_true", default=False)
    arguments.add_argument("--zip-original-index-files", action="store_true", default=False)
    arguments.add_argument("--delete-original-index-files", action="store_true", default=False)
    arguments.add_argument("--omit-warnings", action="store_true", default=False)
    return arguments.parse_args()


def concatenate_files_in_directory(dir_path: str, list_of_file_paths: list, zip_original_files: bool = True,
                                   delete_original_files: bool = False, raise_warning: bool = True):
    """
    Concatenates results files in the directory into a single file. The result files are assumed to have names using the
    format: "index-$RUN_NUMBER.npy". The concatenated file is stored in dir_path in a file named
    "indices-$MIN_RUN_NUMBER-$MAX_RUN_NUMBER.npy".

    If zip_original_files is true, the original files are zipped into a zip file with the same name as the concatenated
    file, but with zip extension.

    If delete_original_files is true, the original files are deleted but only if they were first zipped.

    param dir_path: path to a dictionary containing results
    param list_of_file_paths: list of paths corresponding to the files contained in dir_path
    param zip_original_files: bool indicating whether to compress original files into a single zip file
    param delete_original_files: bool indicating whether to delete original
    param raise_warning: bool indicating whether to raise a warning when deleting files
    """

    delete_original_files = zip_original_files and delete_original_files    # only delete files if they're zipped first

    # concatenate results
    indices = get_indices(dir_path)
    concatenate_results(dir_path, store_concatenated_results=True, indices=indices)

    if not zip_original_files: return
    # zip files
    zip_file_name = "indices-{0}-{1}.zip".format(indices[0], indices[-1])
    zip_file_path = os.path.join(dir_path, zip_file_name)
    if not os.path.isfile(zip_file_path):
        with zipfile.ZipFile(zip_file_path, mode="w") as archive:
            for file_path in list_of_file_paths:
                archive.write(file_path, arcname=os.path.basename(file_path))

    if not delete_original_files: return
    # delete original files
    if raise_warning:           # raise warning if indicated
        warning_message = "The following files are going to be deleted:"
        for path in list_of_file_paths:
            warning_message += "\n\t{0}".format(path)
        print(warning_message)
        user_input = input("Enter x to cancel or enter to continue... ")
        if user_input == "x": return

    for file_path in list_of_file_paths:
        os.remove(file_path)


def main():
    arguments = read_arguments()

    results_dir = arguments.results_dir
    skip_list = arguments.skip_list.split(",")

    stack_of_dir_path = [os.path.join(results_dir, name) for name in os.listdir(results_dir)]

    while len(stack_of_dir_path) > 0:
        current_dir_path = stack_of_dir_path.pop(0)

        # skip paths containing any keyword in the skip list
        if any(keyword in current_dir_path for keyword in skip_list): continue
        # skip any files
        if os.path.isfile(current_dir_path): continue

        file_paths = get_file_paths_that_contain_keywords(current_dir_path, ("index", "npy"))
        contains_results_files = len(file_paths) > 0

        if not contains_results_files:
            list_of_dir_paths = [os.path.join(current_dir_path, name) for name in os.listdir(current_dir_path)]
            stack_of_dir_path.extend(list_of_dir_paths)
        else:
            if arguments.verbose:
                print("Concatenating files in: {0}".format(current_dir_path))
            concatenate_files_in_directory(current_dir_path, file_paths,
                                           zip_original_files=arguments.zip_original_index_files,
                                           delete_original_files=arguments.delete_original_index_files,
                                           raise_warning=not arguments.omit_warnings)


if __name__ == "__main__":
    main()
