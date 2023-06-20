import os
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        dest='input_file',
        default=f'{os.getcwd()}/waymo_dataset_images/segments.pt',
    )
    parser.add_argument(
        '-o',
        dest='output_file',
        default=f'{os.getcwd()}/waymo_dataset_images/formatted_dataset.pt',
    )
    parser.add_argument('--seq_length', dest='seq_length', default=10, type=int)
    parser.add_argument('--stride', dest='stride', default=1, type=int)
    parser.add_argument('--pos_predict', dest='pos_predict', default=1, type=int)
    parser.add_argument('--target_len', dest='target_len', default=9, type=int)
    parser.add_argument('--freq', dest='freq', default=1, type=int)

    args = parser.parse_args()
    return args


class Formatter:
    def __init__(
        self, input_file, output_file, seq_length=10, stride=1, pos_predict=1, target_len=9, freq=1
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.seq_length = seq_length
        self.stride = stride
        self.pos_predict = pos_predict
        self.target_len = target_len
        self.freq = freq

        self.segments = torch.load(self.input_file)
        self.x = {'coords': [], 'images': []}
        self.y = {'coords': [], 'images': []}

    def create_data(self, segment_data):
        """
        Dataset recorded at 10Hz --> T = 0.1s
        Given seq_length frames with Tframe = T * freq,  we want to predict target_len frames,
        also working with Tframe, after predict frames.
        """
        segment_length = len(segment_data['coords'])
        n_training = self.seq_length * self.freq - self.freq + 1
        n_predict = self.pos_predict + (self.target_len - 1) * self.freq
        n_loops = (
            (segment_length - n_training - n_predict) // self.stride + 1
            if segment_length >= (n_predict + n_training)
            else 0
        )
        x = {'coords': [], 'images': []}
        y = {'coords': [], 'images': []}
        for i in range(n_loops):
            x_index = i * self.stride
            y_index = x_index + n_training + self.pos_predict - 1

            x['coords'].append(
                [
                    segment_data['coords'][e]
                    for e in range(x_index, x_index + n_training, self.freq)
                ]
            )
            x['images'].append(
                [
                    segment_data['images'][e]
                    for e in range(x_index, x_index + n_training, self.freq)
                ]
            )
            y['coords'].append(
                [
                    segment_data['coords'][e]
                    for e in range(
                        y_index,
                        y_index + (self.target_len - 1) * self.freq + 1,
                        self.freq,
                    )
                ]
            )
            y['images'].append(
                [
                    segment_data['images'][e]
                    for e in range(
                        y_index,
                        y_index + (self.target_len - 1) * self.freq + 1,
                        self.freq,
                    )
                ]
            )

        return x, y

    def run(self):
        for segment_data in self.segments:
            xi, yi = self.create_data(segment_data)
            self.x['coords'].extend(xi['coords'])
            self.x['images'].extend(xi['images'])
            self.y['coords'].extend(yi['coords'])
            self.y['images'].extend(yi['images'])

        torch.save({'x': self.x, 'y': self.y}, self.output_file)


if __name__ == '__main__':
    args = get_args()
    data_generator = Formatter(
        input_file=args.input_file,
        output_file=args.output_file,
        seq_length=args.seq_length,
        stride=args.stride,
        pos_predict=args.pos_predict,
        target_len=args.target_len,
        freq=args.freq,
    )
    data_generator.run()
