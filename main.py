import os
from pathlib import Path
from amadeusgpt import create_project
from amadeusgpt import AMADEUS
from amadeusgpt.utils import parse_result

# NOTE: must add your API key to the .environment file (example.env file provided to show structure of code; create a .env file and reproduce the structure with the real API key)
if 'OPENAI_API_KEY' not in os.environ:  
     os.environ['OPENAI_API_KEY'] = '[[your key]]'

BASE = Path(__file__).parent

# data folder contains video files and optionally keypoint files
# please pay attention to the naming convention as described above
data_folder = BASE.parent / "videos_and_segments"
# where the results are saved 
result_folder = BASE / "video_results"
# Create a project
config = create_project(data_folder, result_folder, video_suffix = ".mp4")

# Create an AMADEUS instance
amadeus = AMADEUS(config)

query = "Plot the trajectory of the animal using the animal center and color it by time"
qa_message = amadeus.step(query)
# we made it easier to parse the result
parse_result(amadeus, qa_message)