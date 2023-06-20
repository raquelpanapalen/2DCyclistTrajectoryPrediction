from metrics import get_iou
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tensorflow.compat.v1 as tf
import torchvision.transforms as T


class CustomDataset(Dataset):
    def __init__(self, data, crops=True, crop_size=224):
        self.data = data
        self.crops = crops
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        keys = self.keys()
        sample = {key: self.data[key][idx] for key in keys}
        if self.crops:
            sample = self.create_crops(sample)
        return sample

    def keys(self):
        return self.data.keys()
    
    def create_crops(self, sample):
        crops = []
        resize = T.Resize(self.crop_size - 1, max_size=self.crop_size)

        for coords, image in zip(sample['x'], sample['x_images']):
            x1_lim = int((coords[0] - 0.75 * coords[2]) * 1920)
            y1_lim = int((coords[1] - 0.75 * coords[3]) * 1920)
            x1,y1,w,h = (
                x1_lim if x1_lim >= 0 else 0,
                y1_lim if y1_lim >= 0 else 0,
                int(coords[2] * 1.5 * 1920),
                int(coords[3] * 1.5 * 1920)
            )
            crop = tf.image.decode_jpeg(image)
            crop = torch.tensor(crop[y1:y1+h, x1:x1+w, :].numpy())
            crop = torch.permute(crop, (2, 0, 1))
            
            try:
                crop = resize(crop)
            except:
                print(x1,y1,w,h)

            x1 = (self.crop_size - crop.shape[2]) // 2
            x2 = (self.crop_size - crop.shape[2]) - x1
            y1 = (self.crop_size - crop.shape[1]) // 2
            y2 = (self.crop_size - crop.shape[1]) - y1

            crop = F.pad(crop, (x1,x2,y1,y2))
            crop = torch.permute(crop, (1, 2, 0))
            crops.append(crop)
        
        sample['crops'] = torch.stack(crops)
        return sample
    

def create_dataloaders(data, batch_size=10, crops=True):
    ## create custom datasets
    train_dataset = CustomDataset(data['train_data'], crops=crops)
    valid_dataset = CustomDataset(data['valid_data'], crops=crops)
    test_dataset = CustomDataset(data['test_data'], crops=crops)

    ## create dataloaders with SHUFFLE
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )

    return train_loader, valid_loader, test_loader


def denormalize_data(x, pred, centroid_nf, size_nf):
    """
    denormalizing only one sequence --> (seq_length, output_size)
    From predicted motion vectors to predicted BB represented by centroid (x,y) and w,h

    x: the actual inputs points
    pred: seq_length * [∆x, ∆y, ∆w, ∆h]
    """
    # Denormalize motion vector to centroid_nf & size vector to size_nf
    motion_pred = torch.mul(pred[:, :2], centroid_nf)
    size_pred = torch.mul(pred[:, 2:], size_nf)
    pred = torch.cat((motion_pred, size_pred), 1)

    # De-compute relative movement
    new_pred = []
    new_pred.append(x[-1] + pred[0])
    for i in range(1, len(pred)):
        new_pred.append(new_pred[-1] + pred[i])

    # Returns seq_length * [x_c, y_c, w, h]
    return torch.stack(new_pred)


def xywh_to_x1y1x2y2(boxes):
    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(0)
    new_boxes = torch.zeros(boxes.shape)
    new_boxes[:, :, 0] = 1920 * (boxes[:, :, 0] - (boxes[:, :, 2] / 2))
    new_boxes[:, :, 1] = 1920 * (boxes[:, :, 1] - (boxes[:, :, 3] / 2))
    new_boxes[:, :, 2] = 1920 * (boxes[:, :, 0] + (boxes[:, :, 2] / 2))
    new_boxes[:, :, 3] = 1920 * (boxes[:, :, 1] + (boxes[:, :, 3] / 2))
    return new_boxes


def delta_to_xywh_and_x1y1x2y2(data, pred):
    """
    Denormalizing a whole batch of data, one sequence at a time
    pred: (batch_size, seq_length, output_size)
    """
    boxes = torch.stack(
        [
            denormalize_data(
                data['x'][i], pred[i], data['centroid_nf'][i], data['size_nf'][i]
            )
            for i in range(len(pred))
        ]
    )
    return boxes, xywh_to_x1y1x2y2(boxes)


def get_batch_metrics(data, pred):
    metrics = {}
    pred = pred.cpu().detach()

    # Denormalize data to get bounding box [x1,y1,x2,y2]
    target = data['y']

    bb_target = xywh_to_x1y1x2y2(target)
    output, bb_pred = delta_to_xywh_and_x1y1x2y2(data, pred)

    # Get centroid metrics (ADE, FDE)
    ade = torch.mean(
        torch.sqrt(
            ((target[:, :, :2] -  output[:, :, :2]) * 1920)**2
        )
    ).item()
    fde = torch.mean(
        torch.sqrt(
            ((target[:, -1, :2] - output[:, -1, :2]) * 1920)**2
        )
    ).item()
    metrics['ade'] = ade
    metrics['fde'] = fde
    
    # Get bounding box metrics (AIOU, FIOU for every target_len)
    aiou = torch.mean(
        get_iou(bb_target.view(-1, 4), bb_pred.view(-1, 4))
    ).item()
    metrics['aiou'] = aiou * 100

    for i in range(pred.shape[1]):
        fiou = torch.mean(get_iou(bb_target[:, i, :], bb_pred[:, i, :])).item()
        metrics['fiou_{}'.format(i)] = fiou * 100
    
    return metrics


def get_seq_metrics(target, output, bb_target, bb_output):
    metrics = {}
    
    # Get centroid metrics (ADE, FDE)
    ade = torch.mean(
        torch.sqrt(
            ((target[:, :2] -  output[:, :2]) * 1920)**2
        )
    ).item()
    metrics['ade'] = ade
    
    # Get bounding box metrics (AIOU, FIOU for every target_len)
    aiou = torch.mean(
        get_iou(bb_target.view(-1, 4), bb_output.view(-1, 4))
    ).item()
    metrics['aiou'] = aiou * 100

    for i in range(output.shape[0]):
        fde = torch.mean(
            torch.sqrt(
                ((target[i, :2] - output[i, :2]) * 1920)**2
            )
        ).item()
        metrics['fde_{}'.format(i)] = fde
        
        fiou = torch.mean(
            get_iou(bb_target[i, :].unsqueeze(0), bb_output[i, :].unsqueeze(0))
        ).item()
        metrics['fiou_{}'.format(i)] = fiou * 100
    
    return metrics