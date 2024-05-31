import pickle
import numpy as np
import argparse

class load:
    def __init__(self, lookupfile_path, activity_name):
        
        with open(lookupfile_path, 'rb') as f:
            roadFile = pickle.load(f)
            
        self.exercise_array = np.array(roadFile[activity_name])
        
    def array(self):
        return self.exercise_array
    
    def array_shape(self):
        return self.exercise_array.shape
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--activity", required=True,
        help="activity to be recorder")
    ap.add_argument("-v", "--video", required=True,
        help="video file from which keypoints are to be extracted")
    ap.add_argument("-l", "--lookup", default="lookup_new.pickle",
        help="The pickle file to dump the lookup table")
    args = vars(ap.parse_args())
    with open(args["lookup"], 'rb') as f:
        roadFile = pickle.load(f)
    exercise_array = roadFile[args["activity"]]
    print(exercise_array.shape) #(images, name, coordinates)