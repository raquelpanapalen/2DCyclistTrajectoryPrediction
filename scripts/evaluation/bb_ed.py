from utils import get_seq_metrics, xywh_to_x1y1x2y2, delta_to_xywh_and_x1y1x2y2

import pandas as pd
import torch

def save_bb_ed_results(net, loader, device, csv_path):
    metrics = {
        'x': [],
        'y': [],
        'w': [],
        'h': [],
        'index': []
    }

    net.eval()

    with torch.no_grad():
        
        # init hidden state
        net.encoder.init_hidden(net.batch_size)
        
        # iterate over data
        for batch in loader:
            
            # input & target tensors
            input = torch.cat((batch['x'][:, 1:, :], batch['delta_x']), dim=2).type(torch.float)
            target = batch['delta_y'].type(torch.float)
            input, target = input.to(device), target.to(device)

            # get predicted outputs
            output = net(input).cpu().detach()
            
            # get denormalized LSTM prediction
            output, bb_output = delta_to_xywh_and_x1y1x2y2(batch, output)
            bb_target = xywh_to_x1y1x2y2(batch['y'])
            
            for i in range(len(output)):
                # metrics evaluation
                batch_metrics = get_seq_metrics(
                    batch['y'][i],
                    output[i],
                    bb_target[i],
                    bb_output[i]
                )
                
                metrics['x'].append((torch.mean(batch['x'][i][:, 0]) * 1920).item())
                metrics['y'].append((torch.mean(batch['x'][i][:, 1]) * 1920).item())
                metrics['w'].append((torch.mean(batch['x'][i][:, 2]) * 1920).item())
                metrics['h'].append((torch.mean(batch['x'][i][:, 3]) * 1920).item())
                metrics['index'].append(batch['index'][i].item())
                
                for key in batch_metrics:
                    if key in metrics:
                        metrics[key].append(batch_metrics[key])
                    else:
                        metrics[key] = [batch_metrics[key]]
    
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(csv_path, index=False)