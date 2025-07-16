from pathlib import Path

import deeplabcut as dlc
from amadeusgpt import AMADEUS, create_project
from amadeusgpt.utils import parse_result
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), True)

BASE = Path(__file__).parent

# data folder contains video files and keypoint files.
# You can change this to your own data folder.


# where the results are saved
result_folder = BASE / "video_results"
data_folder = BASE / "data"

CONFIG_PATH = BASE / "video_results" / "config.yaml"

# NOTE: must add your API key to the .env file (example.env file provided to show structure of code; create a .env file and reproduce the structure with the real API key)

input_arguments = {
    "animal_info": {"individuals": 1, "species": "topview_mouse"},
    "data_info.video_suffix": ".avi",
    "llm_info": {
        "max_tokens": 4096,
        "temperature": 0.0,
        # you can switch this to gpt-4o-mini for cheaper inference at the cost of worse performance.
        "gpt_model": "gpt-4o",
        "keep_last_n_messages": 2,
    },
    "keypoint_info": {
        "use_3d": True,  # use 3D keypoints
    },
    "video_info": {"scene_frame_number": 300},
}

# Create a project
project_config = create_project(data_folder, result_folder, **input_arguments)


# Create an AMADEUS instance
amadeus = AMADEUS(project_config)

query = "Plot the trajectory of the animal using the animal center and color it by time"
qa_message = amadeus.step(query)
# we made it easier to parse the result
parse_result(amadeus, qa_message)
