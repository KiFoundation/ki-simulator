import os
from datetime import datetime


def create_res_forlder(script):
    # Build path
    this_file = os.path.abspath(os.path.dirname(__file__))
    directory = os.path.join(this_file, '../' + script + '/res/' + datetime.today().strftime("%Y-%m-%d"))

    # Check if folder exists
    if not os.path.exists(directory):
        print("Creating result folder")
        # Create folder
        os.makedirs(directory)
    else:
        print("Result folder already exists")

    print("Creating run folder")
    run_directory = directory + "/Run-" + datetime.today().strftime("%H.%M.%S.%f")[:-3]
    os.makedirs(run_directory)

    return run_directory + "/"
