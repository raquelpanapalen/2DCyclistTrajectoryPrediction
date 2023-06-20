import torch

root_path = '/home/raquelpanadero/Desktop/TFG/waymo_dataset_images'

training = torch.load(f'{root_path}/training.pt')
testing = torch.load(f'{root_path}/testing.pt')
validation = torch.load(f'{root_path}/validation.pt')
data = {'train_data': training, 'test_data': testing, 'valid_data': validation}

torch.save(data, f'{root_path}/dataset.pt')