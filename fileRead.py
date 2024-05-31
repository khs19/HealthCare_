#dataset folder
  #level_dataset label
    #ㄴdataset_name
from logging import root
import os
import random
if __name__ == '__main__':
    
    root_path = './dataset' #여기에 dataset folder를 넣는다.
    dataset_labels = os.listdir(root_path)  #데이터 셋 레이블들

    print(dataset_labels)

    dataset_label_paths = []
    for dataset_label in dataset_labels:
        dataset_label_paths.append(os.path.join(root_path, dataset_label))#dataset의 경로들

    print(dataset_label_paths)

    datasets_dict = {}
    for dataset_label_path in dataset_label_paths:
        datasets_path = []
        datasets = os.listdir(dataset_label_path)
        for dataset in datasets:
            datasets_path.append(os.path.join(dataset_label_path, dataset))

        dataset_label = dataset_label_path.split('/')[2]
        datasets_dict[dataset_label] = datasets_path
        
    print(datasets_dict)

    for datasets_list in datasets_dict.items():
        dataset_label = datasets_list[0]
        dataset_labeled_path = datasets_list[1]
        print(dataset_label, dataset_labeled_path)

class file:
    def __init__(self, root_path):
        self.root_path = root_path
        
        self.datasets_dict = self.find_file_dict()
        self.file_paths = self.find_all_file_paths()
        self.datasets_labels = self.find_all_label()
        self.random_file_path = self.find_random_file_path() #root/label_exerciseName/video.mp4(train.mp4: sep video, test video: full video)
        
    def find_file_dict(self):
            dataset_labels = os.listdir(self.root_path)  #데이터 셋 레이블들
            dataset_label_paths = []
            for dataset_label in dataset_labels:
                dataset_label_paths.append(os.path.join(self.root_path, dataset_label))#dataset의 경로들

            datasets_dict = {}
            for dataset_label_path in dataset_label_paths:
                datasets_path = []
                datasets = os.listdir(dataset_label_path)
                for dataset in datasets:
                    datasets_path.append(os.path.join(dataset_label_path, dataset))

                dataset_label = dataset_label_path.split('/')[2]
                datasets_dict[dataset_label] = datasets_path
            
            return datasets_dict

    def find_all_file_paths(self):
        file_dict = self.find_file_dict()
        file_paths = []
        for file_path in file_dict.values():
            file_paths += file_path
        
        return file_paths

    def find_all_label(self):
        file_dict = self.find_file_dict()
        file_labels = []
        for file_label in file_dict.keys():
            file_labels.append(file_label)
        
        return file_labels

    def find_random_file_path(self):
        file_path_list = self.find_all_file_paths()
        random_data_number = random.randrange(len(file_path_list))
        random_file_path = file_path_list[random_data_number]
        return random_file_path