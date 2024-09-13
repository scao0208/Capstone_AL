# Active Learning for Object Detection Using Top-k Sampling Strategy
Installation:
`conda create -n myenv python=3.12.2 -y`
open the folder bdd100k and clone the Github repo download dataset. Also clone the cocoapi. 
install the dependency referred to the requirements.txt
`pip install -r requirements.txt`. 
After that, we need to get to the cocoapi/PythonAPI:
`pip install -e .`
Then we will see the pycocotools listed in the conda pip list. 



When packaging `pycocotool`, there might need modify the np.float as np.float64. You can check the bugs when running `train.py` and add `val_map.pop("classes")` in line 59

