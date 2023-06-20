from utils import denormalize_data
from models.kalman_filter import KalmanFilter
import torch


def predict_ca(data, target_len=9):
    avg_delta = torch.mean(data['delta_x'], -2).repeat(target_len, 1)
    output = denormalize_data(
        data['x'].squeeze(), avg_delta, data['centroid_nf'], data['size_nf']
    )

    return output


def predict_cv_cs(data, target_len=9):
    data_x = data['x'].squeeze()
    velocity = (data_x[-1] - data_x[0]) / len(data_x)

    output = [data_x[-1] + velocity]
    for i in range(target_len - 1):
        output.append(output[i] + velocity)

    output = torch.stack(output)
    return output


def predict_lkf(data, target_len=9):
    input = data['x'].squeeze()
    data_x = input[:, :2]
    size = torch.mean(input[:, 2:], dim=0).repeat(target_len, 1)

    KF = KalmanFilter(1, data_x[0], 2, 1, 1)

    for i in range(1, len(data_x)):
        pred, _ = KF.predict()
        KF.update(data_x[i].numpy())

    preds = []
    for i in range(target_len):
        pred, _ = KF.predict()
        preds.append([pred[0].item(), pred[1].item()])

    output = torch.tensor([[e[0], e[1]] for e in preds])
    output = torch.cat((output, size), 1)
    return output
