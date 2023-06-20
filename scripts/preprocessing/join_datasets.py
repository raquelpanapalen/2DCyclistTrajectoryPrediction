import os
import glob
import argparse
import torch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        dest='input_dir',
        default=f'{os.getcwd()}/waymo_dataset_images/',
    )
    parser.add_argument(
        '-o',
        dest='output_file',
        default=f'{os.getcwd()}/waymo_dataset_images/training.pt',
    )

    args = parser.parse_args()
    return args


class JoinDatasets:
    def __init__(self, input_dir, output_file):
        self.input_dir = input_dir
        self.output_file = output_file
        self.data = None

    def join_data(self):
        datasets = sorted(glob.glob(f'{self.input_dir}/*.pt'))
        for dataset in tqdm(datasets):
            data1 = torch.load(dataset)
            if self.data is None:
                self.data = data1
            else:
                data2 = self.data
                self.data = {
                    key: torch.cat((data1[key], data2[key]))
                    if 'images' not in key
                    else data1[key] + data2[key]
                    for key in data1
                }

    def run(self):
        self.join_data()
        torch.save(self.data, self.output_file)


if __name__ == '__main__':
    args = get_args()
    normalizer = JoinDatasets(input_dir=args.input_dir, output_file=args.output_file)
    normalizer.run()
