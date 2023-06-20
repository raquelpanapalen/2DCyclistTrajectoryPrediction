from models.linear_models import predict_ca, predict_cv_cs, predict_lkf
from utils import delta_to_xywh_and_x1y1x2y2

import torch
import pandas as pd
import seaborn as sns
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_frames(data, bb_net, bb_crops_net, device, index):
    '''plot sequence'''

    # get decoded image
    x_images = list(zip(*data['x_images']))
    decoded_image = tf.image.decode_jpeg(x_images[index][-1])

    # get one sequence for CV & CS, CA, LKF
    linear_data = {
        'x': data['x'][index],
        'delta_x': data['delta_x'][index],
        'centroid_nf': data['centroid_nf'][index],
        'size_nf': data['size_nf'][index],
    }

    # get input with batch_size=512 for BB E-D
    bb_input = (
        torch.cat((data['x'][:, 1:, :], data['delta_x']), dim=2)
        .type(torch.float)
        .to(device)
    )

    # get input with batch_size=16 for BB+Crops E-D
    first_index = index // 16
    batch16_data = {
        key: data[key][16 * first_index : 16 * (first_index + 1)]
        for key in data.keys() if 'images' not in key
    }    
    
    bb_crops_input = (
        torch.cat((batch16_data['x'][:, 1:, :], batch16_data['delta_x']), dim=2)
        .type(torch.float)
        .to(device)
    )
    crops = bb_crops_net.normalize_crops(batch16_data['crops'][:, -1, :]).to(device)

    # get input & target bbs
    input = data['x'][index] * 1920
    target = torch.cat((input[-1].unsqueeze(0), data['y'][index] * 1920))

    # CV & CS, CA, LKF outputs
    cv_cs_out = torch.cat((
        input[-1].unsqueeze(0), 
        predict_cv_cs(linear_data) * 1920
    ))
    ca_out = torch.cat((
        input[-1].unsqueeze(0), 
        predict_ca(linear_data) * 1920
    ))
 
    lkf_out = torch.cat((
        input[-1].unsqueeze(0), 
        predict_lkf(linear_data) *1920
    ))

    # BB E-D output
    bb_out = bb_net(bb_input).cpu().detach()
    bb_out, _ = delta_to_xywh_and_x1y1x2y2(data, bb_out)
    bb_out = torch.cat((input[-1].unsqueeze(0), bb_out[index] * 1920))

    # BB+Crops E-D output
    bb_crops_out = bb_crops_net(bb_crops_input, crops).cpu().detach()
    bb_crops_out, _ = delta_to_xywh_and_x1y1x2y2(batch16_data, bb_crops_out)
    bb_crops_out = torch.cat((input[-1].unsqueeze(0), bb_crops_out[index-first_index*16] * 1920))
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    #Plot trajectory lines
    for coords, label, color in zip(
        [input, target, cv_cs_out, ca_out, lkf_out, bb_out, bb_crops_out],
        ['Input', 'Ground truth', 'CV & CS', 'CA', 'LKF', 'BB E-D', 'BB+Crops E-D'],
        ['blue','lime', 'orange', 'yellow', 'red', 'lightpink', 'darkviolet']
    ):
        ax.plot(coords[:, 0], coords[:, 1], '-', label=label, color=color)  
    
    for (coords, color) in zip(
        [target[-1], cv_cs_out[-1], ca_out[-1], lkf_out[-1], bb_out[-1], bb_crops_out[-1]],
        ['lime', 'orange', 'yellow', 'red', 'lightpink', 'darkviolet']
    ):
        # Plot last centroid
        ax.plot(coords[0], coords[1], 'o', color=color)
        
        # Plot last bounding box
        ax.add_patch(
            patches.Rectangle(
                xy=(
                    (coords[0] - 0.5 * coords[2]),
                    (coords[1] - 0.5 * coords[3]),
                ),
                width=coords[2],
                height=coords[3],
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
        )
    
    xlim_original = plt.gca().get_xlim()
    ylim_original = plt.gca().get_ylim()

    xlim = (
        xlim_original[0] - 30 if xlim_original[0] - 30 > 0 else 0, 
        xlim_original[1] + 30 if xlim_original[1] + 30 < 1920 else 1920
    )
    ylim = (
        ylim_original[1] + 30 if ylim_original[1] + 30 < 1280 else 1280, 
        ylim_original[0] - 30 if ylim_original[0] - 30 > 0 else 0
    )

    plt.imshow(decoded_image)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()


def plot_mean_metric(metric, set, root_path):
    # metric is 'fde' or 'fiou'
    methods = ['ca', 'cv_cs', 'lkf', 'bb_ed', 'bb_crops_ed']
    labels = ['CA', 'CV & CS', 'LKF', 'BB E-D', 'BB+Crops E-D']

    sns.set_style("darkgrid")
    sns.set_context("notebook", rc={"grid.linewidth": 4})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 25))

    for method, label in zip(methods, labels):
        csv_path = f'{root_path}/{method}_{set}.csv'
        df = pd.read_csv(csv_path)
        metrics = df.describe()
        cols = [c for c in df.columns if metric in c]
        res = [metrics[col]['mean'] for col in cols]
        ax = sns.lineplot(x=range(1,10), y=res, label=label, linewidth=7)

    ax.set_xlabel('Target length (# frames)', fontsize=40)
    ax.set_ylabel('FDE (px)' if metric == 'fde' else 'FIOU (%)', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=35, width=10)
    plt.legend(prop={'size': 45})
    plt.show()


def plot_csv_metrics(logs_path):
    lr_list = ['1e-06', '1e-05', '1e-04', '0.001', '0.01']
    sets = ['train', 'valid']
    metrics = ['ade', 'aiou']
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

    ax[0,0].text(19, 36, 'ADE', fontsize=20, fontweight='bold')
    ax[0,0].text(72.5, 36, 'AIOU', fontsize=20, fontweight='bold')


    for j, metric in enumerate(metrics):
        for i, subset in enumerate(sets):
            for lr in lr_list:
                df = pd.read_csv(f'{logs_path}/dinov2_finetuning_e40_lr_{lr}_bsize_16_{metric}_{subset}.csv')
                df['avg'] = df['Value'].rolling(5).sum() / 5
                df['Step'] = df['Step'] / df['Step'][1]
                sns.lineplot(data=df, x='Step', y='avg', label=f'lr={lr}', ax=ax[i,j])

            ax[i,j].set_xlabel('epochs')

    ax[0, 0].set_title('Train')
    ax[0, 1].set_title('Train')
    ax[1, 0].set_title('Validation')
    ax[1, 1].set_title('Validation')
    ax[0, 0].set_ylabel('ADE (px)')
    ax[1, 0].set_ylabel('ADE (px)')
    ax[0, 1].set_ylabel('AIOU (%)')
    ax[1, 1].set_ylabel('AIOU(%)')

    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 0.9), fontsize=11)

    ax[0,0].get_legend().remove()
    ax[0,1].get_legend().remove()
    ax[1,0].get_legend().remove()
    ax[1,1].get_legend().remove()

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.2)

    plt.show()
