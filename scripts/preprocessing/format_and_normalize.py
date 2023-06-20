import os
import glob
import argparse
from tqdm import tqdm

from scripts.preprocessing.bb_crops.create_subsequences import Formatter
from scripts.preprocessing.bb_crops.normalize import Normalizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', dest='out_dir', default=f'{os.getcwd()}/waymo_dataset_images/segments'
    )

    args = parser.parse_args()
    return args


class JoinSegments:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def run(self):
        segments = sorted(glob.glob(f'{self.out_dir}/segments*.pt'))
        for segment in tqdm(segments):          
            dataset_file = segment.replace('segments', 'formatted_dataset')
            if not os.path.exists(dataset_file):
                data_generator = Formatter(input_file=segment, output_file=dataset_file)
                data_generator.run()

            norm_file = segment.replace('segments', 'normalized_dataset')
            if not os.path.exists(norm_file):
                data_normalizer = Normalizer(input_file=dataset_file, output_file=norm_file, min_size=80, padding=40)
                data_normalizer.run()



if __name__ == '__main__':
    args = get_args()
    joiner = JoinSegments(
        out_dir=args.out_dir
    )
    joiner.run()
