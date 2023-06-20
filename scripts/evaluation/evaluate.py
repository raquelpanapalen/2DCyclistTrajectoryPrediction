from evaluation.linear_models import save_linear_results
from evaluation.bb_ed import save_bb_ed_results
from evaluation.bb_crops_ed import save_bb_crops_ed_results
from models.linear_models import *
from utils import create_dataloaders

import pandas as pd

def save_results(bb_net, bb_crops_net, data, device, root_path):

    train_loader, valid_loader, test_loader = create_dataloaders(data, batch_size=1)

    # CV & CS
    csv_path = f'{root_path}/cv_cs_%s.csv'
    save_linear_results(train_loader, predict_cv_cs, csv_path % 'train')
    save_linear_results(test_loader, predict_cv_cs, csv_path % 'test')
    save_linear_results(valid_loader, predict_cv_cs, csv_path % 'valid')

    # CA
    csv_path = f'{root_path}/ca_%s.csv'
    save_linear_results(train_loader, predict_ca, csv_path % 'train')
    save_linear_results(test_loader, predict_ca, csv_path % 'test')
    save_linear_results(valid_loader, predict_ca, csv_path % 'valid')

    # LKF
    csv_path = f'{root_path}/lkf_%s.csv'
    save_linear_results(train_loader, predict_lkf, csv_path % 'train')
    save_linear_results(test_loader, predict_lkf, csv_path % 'test')
    save_linear_results(valid_loader, predict_lkf, csv_path % 'valid')

    # BB E-D
    train_loader, valid_loader, test_loader = create_dataloaders(data, batch_size=512)
    csv_path = f'{root_path}/bb_ed_%s.csv'
    save_bb_ed_results(bb_net, train_loader, device, csv_path % 'train')
    save_bb_ed_results(bb_net, test_loader, device, csv_path % 'test')
    save_bb_ed_results(bb_net, valid_loader, device, csv_path % 'valid')


    # BB+Crops E-D
    train_loader, valid_loader, test_loader = create_dataloaders(data, batch_size=16)
    csv_path = f'{root_path}/bb_crops_ed_%s.csv'
    save_bb_crops_ed_results(bb_crops_net, train_loader, device, csv_path % 'train')
    save_bb_crops_ed_results(bb_crops_net, test_loader, device, csv_path % 'test')
    save_bb_crops_ed_results(bb_crops_net, valid_loader, device, csv_path % 'valid')

def get_results(root_path):
    results = {}
    methods = ['ca', 'cv_cs', 'lkf', 'bb_ed', 'bb_crops_ed']
    sets = ['train', 'test', 'valid']
    for method in methods:
        for set in sets:
            csv_path = f'{root_path}/{method}_{set}.csv'
            df = pd.read_csv(csv_path)
            metrics = df.describe()

            res = {
                set: {
                    'fde_0': metrics['fde_0']['mean'],
                    'ade': metrics['ade']['mean'],
                    'fde_8': metrics['fde_8']['mean'],
                    'fiou_0': metrics['fiou_0']['mean'],
                    'aiou': metrics['aiou']['mean'],
                    'fiou_8': metrics['fiou_8']['mean']
                }
            }

            if method in results:
                results[method].update(res)
            else:
                results[method] = res

    return results