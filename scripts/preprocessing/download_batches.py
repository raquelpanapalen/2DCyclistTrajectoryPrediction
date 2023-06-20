import os
import glob
import argparse

import tensorflow.compat.v1 as tf
import torch
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as open_label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['training', 'testing', 'validation'])
    parser.add_argument(
        '-o', dest='output_file', default=f'{os.getcwd()}/waymo_dataset_images/segments.pt'
    )
    parser.add_argument(
        '-d', dest='out_dir', default=f'{os.getcwd()}/waymo_dataset_images'
    )
    parser.add_argument('-i', dest='index', default=0, type=int)
    parser.add_argument('-b', dest='batch_size', default=-1, type=int)

    args = parser.parse_args()
    return args


class BatchDownloader:
    def __init__(self, split, index, batch_size, out_dir, output_file):
        self.split = split
        self.index = index
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.output_file = output_file

        self.bucket_url = 'gs://waymo_open_dataset_v_1_4_0/archived_files/{split}/{split}_%04d.tar'.format(
            split=split
        )
        self.object_length = {'training': 32, 'testing': 8, 'validation': 8}
        self.target_class = 'TYPE_CYCLIST'
        self.segments_done = []

        if os.path.exists(self.output_file):
            self.data = torch.load(self.output_file)
        else:
            self.data = []

    def download_object(self, object_id):
        flag = os.system(f'gsutil cp {self.bucket_url % object_id} {self.out_dir}')
        assert flag == 0, (
            'Failed to download object %d. Make sure gsutil is installed' % object_id
        )
        os.system(
            'cd {}; tar -xf {}_{:04d}.tar'.format(self.out_dir, self.split, object_id)
        )
        os.remove('{}/{}_{:04d}.tar'.format(self.out_dir, self.split, object_id))
        os.remove(f'{self.out_dir}/LICENSE')

    def filter_segment(self, segment_path):
        dataset = tf.data.TFRecordDataset(segment_path, compression_type='')
        flag = 0
        frames = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)

        if not any(
            open_label.Label.Type.Name(camera_object.type) == self.target_class
            for frame in frames
            for camera_object in frame.context.stats.camera_object_counts
        ):
            print(f'No cyclist data in segment {segment_path}')
            flag = 1
        else:
            print(f'Extracting cyclist data from segment {segment_path}')
            self.create_dataset(frames)

        os.remove(segment_path)
        self.segments_done.append(segment_path.split('/')[-1])
        return flag

    def create_dataset(self, frames):
        cyclist_set = {}
        for frame in frames:
            for label in frame.camera_labels[0].labels:
                if open_label.Label.Type.Name(label.type) == self.target_class:
                    coords = (
                        np.array(
                            [
                                label.box.center_x,
                                label.box.center_y,
                                label.box.length,
                                label.box.width,
                            ]
                        )
                        / 1920
                    ).tolist()
                    if label.id not in cyclist_set:
                        cyclist_set[label.id] = {
                            'coords': [coords],
                            'images': [frame.images[0].image],
                        }
                    else:
                        cyclist_set[label.id]['coords'].append(coords)
                        cyclist_set[label.id]['images'].append(frame.images[0].image)

        for c_id in cyclist_set:
            self.data.append(cyclist_set[c_id])

    def run(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        batch_size = (
            self.object_length[self.split] if self.batch_size == -1 else self.batch_size
        )
        for object_id in range(self.index, self.index + batch_size):
            self.download_object(object_id)
            segments = sorted(glob.glob(f'{self.out_dir}/*.tfrecord'))
            n_discarded = 0

            for segment in segments:
                n_discarded += self.filter_segment(segment)

            print(
                f'Object {object_id} done: \n'
                f'\t Total segments: {len(segments)} \n'
                f'\t Discarded segments: {n_discarded}'
            )

            torch.save(self.data, self.output_file)

        with open(f'{self.out_dir}/segments_done.txt', 'a+') as fp:
            if os.path.exists(f'{self.out_dir}/segments_done.txt'):
                fp.write('\n')
            fp.write('\n'.join(self.segments_done))


if __name__ == '__main__':
    args = get_args()
    batch_download = BatchDownloader(
        split=args.split,
        index=args.index,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        output_file=args.output_file
    )
    batch_download.run()
