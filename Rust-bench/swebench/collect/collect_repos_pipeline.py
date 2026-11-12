import json
import os
from get_tasks_pipeline import main as get_tasks_pipeline

def load_json_to_list(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  
    return data

if __name__ == '__main__':
    pass


