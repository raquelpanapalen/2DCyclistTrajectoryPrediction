import torch
from utils import get_batch_metrics

def train_model(
    net, 
    train_loader,
    valid_loader,
    n_epochs, 
    optimizer, 
    criterion, 
    device,
    writer=None,
):

    net.to(device)
    net.train()

    # train for some number of epochs
    for e in range(n_epochs):
        loss_err = 0
        train_ade = 0
        train_fde = 0
        train_metrics = {}

        # batch loop
        for train in train_loader:
            # input & target tensors
            input = torch.cat((train['x'][:, 1:, :], train['delta_x']), dim=2).type(torch.float)
            target = train['delta_y'].type(torch.float)
            
            # outputs tensor
            outputs = torch.tensor([])
            input, target, outputs = input.to(device), target.to(device), outputs.to(device)
            
            # initialize hidden state
            net.encoder.init_hidden(net.batch_size)

            # zero the gradient
            optimizer.zero_grad()

            # forward step
            outputs = net(input)
            
            # compute the loss & backpropagation
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            loss_err += loss.item()

            # metrics evaluation
            metrics = get_batch_metrics(train, outputs)
            train_ade += metrics['ade']
            train_fde += metrics['fde']
            if not train_metrics:
                train_metrics = metrics
            else:
                train_metrics = {key: value + metrics[key] for key, value in train_metrics.items()}

            if writer:
                writer.add_scalar("Loss/train", loss_err/len(train_loader), e*len(train_loader))
                writer.add_scalar("ADE (points)/train", train_ade/len(train_loader), e*len(train_loader))
                writer.add_scalar("FDE (points)/train", train_fde/len(train_loader), e*len(train_loader))
                writer.add_scalar("AIOU/train", train_metrics['aiou']/len(train_loader), e*len(train_loader))
                for key in [key for key in train_metrics.keys() if 'fiou' in key]:
                    writer.add_scalar(
                        f'{key.upper()}/train', train_metrics[key]/len(train_loader), e*len(train_loader)
                    )


        # Get validation loss
        net.encoder.init_hidden(net.batch_size)
        val_losses = 0
        val_ade = 0
        val_fde = 0
        valid_metrics = {}
        
        net.eval()
        for valid in valid_loader:
            # input & target tensors
            input = torch.cat((valid['x'][:, 1:, :], valid['delta_x']), dim=2).type(torch.float)
            target = valid['delta_y'].type(torch.float)
            input, target = input.to(device), target.to(device)
            
            # forward step & loss computation
            with torch.no_grad():
                outputs = net(input)
                val_loss = criterion(outputs, target)
                val_losses += val_loss.item()

            # evaluation metrics
            metrics = get_batch_metrics(valid, outputs)
            val_ade += metrics['ade']
            val_fde += metrics['fde']
            if not valid_metrics:
                valid_metrics = metrics
            else:
                valid_metrics = {key: value + metrics[key] for key, value in valid_metrics.items()}

        if writer:
            writer.add_scalar("Loss/valid", val_losses/len(valid_loader), e*len(train_loader))
            writer.add_scalar("ADE (points)/valid", val_ade/len(valid_loader), e*len(train_loader))
            writer.add_scalar("FDE (points)/valid", val_fde/len(valid_loader), e*len(train_loader))
            writer.add_scalar("AIOU/valid", valid_metrics['aiou']/len(valid_loader), e*len(train_loader))
            for key in [key for key in valid_metrics.keys() if 'fiou' in key]:
                writer.add_scalar(
                    f'{key.upper()}/valid', valid_metrics[key]/len(valid_loader), e*len(train_loader)
                )

        print(
            "Epoch: {}/{}...".format(e+1, n_epochs),
            "Step: {}...".format(e*len(train_loader)),
            "Loss: {:.6f}...".format(loss_err/len(train_loader)),
            "Val Loss: {:.6f}...".format(val_losses/len(valid_loader)),
            "ADE/T: {:.2f}...".format(train_ade/len(train_loader)),
            "FDE/T: {:.2f}...".format(train_fde/len(train_loader)),
            "IOU-0/T: {:.2f}...".format(train_metrics['fiou_0']/len(train_loader)),
            "AIOU/T: {:.2f}...".format(train_metrics['aiou']/len(train_loader)),
            "FIOU/T: {:.2f}...".format(train_metrics['fiou_8']/len(train_loader)),
            "IOU-0/V: {:.2f}...".format(valid_metrics['fiou_0']/len(valid_loader)),
            "AIOU/V: {:.2f}...".format(valid_metrics['aiou']/len(valid_loader)),
            "FIOU/V: {:.2f}...".format(valid_metrics['fiou_8']/len(valid_loader))
        )
        net.train()

    if writer:
        writer.flush()

