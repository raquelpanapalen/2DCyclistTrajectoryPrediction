import os
import numpy as np
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        dest='input_file',
        default=f'{os.getcwd()}/waymo_dataset_images/formatted_dataset.pt',
    )
    parser.add_argument(
        '-o',
        dest='output_file',
        default=f'{os.getcwd()}/waymo_dataset_images/normalized_dataset.pt',
    )
    parser.add_argument('-s', '--min-size', dest='min_size', default=70, type=int)
    parser.add_argument('-p', '--padding', dest='padding', default=40, type=int)

    args = parser.parse_args()
    return args


class Normalizer:
    def __init__(self, input_file, output_file, min_size=70, padding=40):
        self.input_file = input_file
        self.output_file = output_file
        self.min_size = min_size
        self.padding = padding
        self.data = torch.load(self.input_file)

    def normalize_data(self, data):
        # Get x,y to normalize
        x = data['x']['coords']
        y = data['y']['coords']

        delta_x = []
        delta_y = []
        centroid_nf = []
        size_nf = []

        data_x = torch.tensor(x).numpy()
        data_y = torch.tensor(y).numpy()

        # Iterate for each data sequence
        for i in range(len(data_x)):
            xi = []
            yi = []

            for j in range(len(data_x[i]) - 1):
                xi.append(data_x[i][j + 1] - data_x[i][j])

            yi.append(data_y[i][0] - data_x[i][-1])
            for j in range(len(data_y[i]) - 1):
                yi.append(data_y[i][j + 1] - data_y[i][j])

            xi = np.array(xi)
            yi = np.array(yi)

            # Normalize motion vectors
            x_motion = np.array(xi[:, :2])
            y_motion = np.array(yi[:, :2])
            motion_nf = np.max(np.linalg.norm(x_motion, axis=1))
            if motion_nf != 0:
                x_motion = x_motion / motion_nf
                y_motion = y_motion / motion_nf

            # Normalize size difference
            x_size = np.array(xi[:, 2:])
            y_size = np.array(yi[:, 2:])
            box_nf = np.max(np.linalg.norm(x_size, axis=1))
            if box_nf != 0:
                x_size = x_size / box_nf
                y_size = y_size / box_nf

            delta_x.append(np.concatenate((x_motion, x_size), axis=1))
            delta_y.append(np.concatenate((y_motion, y_size), axis=1))
            centroid_nf.append(motion_nf)
            size_nf.append(box_nf)

        return {
            'x': torch.tensor(np.array(x)),
            'y': torch.tensor(np.array(y)),
            'delta_x': torch.tensor(np.array(delta_x)),
            'delta_y': torch.tensor(np.array(delta_y)),
            'centroid_nf': torch.tensor(np.array(centroid_nf)),
            'size_nf': torch.tensor(np.array(size_nf)),
            'x_images': data['x']['images'],
            'y_images': data['y']['images'],
            'index': torch.arange(len(data_x))
        }

    def check_bb_size_limits(self):
        """
        self.data is structured like:
        {
            'x': {
                'coords': [
                    [[..], [..], [..]],
                    [[..], [..], [..]],
                ]
                'images': []
            },
            'y': {
                'coords': [],
                'images': []
            }
        }
        """
        output_data = {
            'x': {'coords': [], 'images': []},
            'y': {'coords': [], 'images': []},
        }
        for i in range(len(self.data['x']['coords'])):
            # Check global
            global_ok = self.check_coords(
                np.concatenate((
                    np.array(self.data['x']['coords'][i]), 
                    np.array(self.data['y']['coords'][i])
                ))
            )
            if not global_ok:
                continue

            # Check input
            input_ok = self.check_coords(self.data['x']['coords'][i])
            if not input_ok:
                continue

            # Check target
            target_ok = self.check_coords(self.data['y']['coords'][i])
            if not target_ok:
                continue

            output_data['x']['coords'].append(self.data['x']['coords'][i])
            output_data['x']['images'].append(self.data['x']['images'][i])
            output_data['y']['coords'].append(self.data['y']['coords'][i])
            output_data['y']['images'].append(self.data['y']['images'][i])

        return output_data

    def check_coords(self, coords):
        coords = np.array(coords) * 1920
        avg_coords = np.mean(coords, axis=0)
        if avg_coords[2] < self.min_size and avg_coords[3] < self.min_size:
            return False

        left_limit = np.mean(coords[:, 0] - coords[:, 2] / 2)
        right_limit = np.mean(coords[:, 0] + coords[:, 2] / 2)
        top_limit = np.mean(coords[:, 1] - coords[:, 3] / 2)
        bottom_limit = np.mean(coords[:, 1] + coords[:, 3] / 2)

        if (
            left_limit < self.padding
            or top_limit < self.padding
            or right_limit > 1920 - self.padding
            or bottom_limit > 1280 - self.padding
        ):
            return False

        return True

    def create_data(self):
        # discard small bounding boxes
        output_data = self.check_bb_size_limits()
        output_data = self.normalize_data(output_data)

        return output_data

    def run(self):
        data = self.create_data()
        torch.save(data, self.output_file)


if __name__ == '__main__':
    args = get_args()
    normalizer = Normalizer(
        input_file=args.input_file,
        output_file=args.output_file,
        min_size=args.min_size,
        padding=args.padding,
    )
    normalizer.run()
