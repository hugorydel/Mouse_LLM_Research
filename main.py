from pathlib import Path

from amadeusgpt import AMADEUS, create_project
from amadeusgpt.utils import parse_result
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), True)

BASE = Path(__file__).parent

# data folder contains video files and keypoint files.
# You can change this to your own data folder.


# The folder to which the results will be saved to
result_folder = BASE / "video_results"

# The folder where the data (including 2d .h5 files for each trial and camera and .avi videos for each camera and trial) is coming from
data_folder = BASE / "data"

# Folder to which the Amadeus config file is saved to.
CONFIG_PATH = BASE / "video_results" / "config.yaml"

# Folder to which the data config file is saved to.
DATA_CONFIG_PATH = BASE / "data_configs" / "config.yaml"

# Folder in which the pickle file folders are stored in.
PICKLE_FILES_PATH = BASE / "data_configs" / "pickle_files"

# NOTE: must add your API key to the .env file (example.env file provided to show structure of code; create a .env file and reproduce the structure with the real API key)

input_arguments = {
    "data_info.video_suffix": ".avi",
    "keypoint_info.use_3d": True,
}

# Create a project
project_config = create_project(data_folder, result_folder, **input_arguments)


# Create an AMADEUS instance
amadeus = AMADEUS(project_config)

query = "Plot the trajectory of the animal using the animal center and color it by time"
qa_message = amadeus.step(query)
# we made it easier to parse the result
parse_result(amadeus, qa_message)
