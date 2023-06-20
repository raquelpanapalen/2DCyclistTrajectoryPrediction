from utils import get_seq_metrics, xywh_to_x1y1x2y2
import pandas as pd


def save_linear_results(loader, method, csv_path):
    results = {'index': []}

    for batch in loader:
        output = method(batch)
        target = batch['y'].squeeze(0)
        bb_output = xywh_to_x1y1x2y2(output).squeeze(0)
        bb_target = xywh_to_x1y1x2y2(target).squeeze(0)

        batch_metrics = get_seq_metrics(target, output, bb_target, bb_output)
        results['index'].append(batch['index'].item())

        for key in batch_metrics:
            if key in results:
                results[key].append(batch_metrics[key])
            else:
                results[key] = [batch_metrics[key]]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(csv_path, index=False)