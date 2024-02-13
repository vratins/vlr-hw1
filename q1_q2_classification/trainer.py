from __future__ import print_function

import torch
import numpy as np
import wandb
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # writer = SummaryWriter()
    wandb.init(project="hw1_q1", name=model_name)
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################

            def BinaryCrossEntropyLoss(output, target):
            #Binary cross-entropy loss calculated for multi-label classifiction
            #input: predictions from network, ground truth labels
            #output: loss, single number
                
                output_probability = torch.sigmoid(output)
                #account for underflow
                bce_loss = -wgt * (target * torch.log(output_probability + 1e-10) + (1 - target) * torch.log(1 - output_probability + 1e-10)) 

                return bce_loss.sum(dim=1).mean() #aggregate loss across all labels first
                
            loss = BinaryCrossEntropyLoss(output, target)
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            
            if cnt % args.log_every == 0:
                wandb.log({"Loss/train": loss.item()}, step=cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        wandb.log({f"{tag}/grad": wandb.Histogram(value.grad.cpu().numpy())}, step=cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                wandb.log({"map": map}, step=cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            wandb.log({"learning_rate": scheduler.get_last_lr()[0]}, step=cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)
    
    wandb.finish()
    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
