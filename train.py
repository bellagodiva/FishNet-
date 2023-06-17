import torch
import torch.optim as optim
import torch.nn as nn
from model import FishNet
from sklearn.metrics import accuracy_score, f1_score
import time
import numpy as np
import os

import pickle
from torch.utils.tensorboard import SummaryWriter

def train_model(hyp_params, train_loader, valid_loader, test_loader):
    model = FishNet(num_cls=hyp_params.num_cls, n_stage=hyp_params.n_stage, n_channel=hyp_params.n_channel, 
                    n_res_block=hyp_params.n_res_block, n_trans_block=hyp_params.n_trans_block)
    model = model.cuda()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True)

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        step_loss, step_size = 0, 0
        start_time = time.time()
        for i_batch, samples in enumerate(train_loader):
            image = samples['image']
            label = samples['label'].squeeze(-1)   # if num of labels is 1
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    image, label = image.cuda(), label.cuda()
                    label = label.long()
            
            batch_size = image.size(0)
            preds = model(image)
            loss = criterion(preds, label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            step_loss += loss.item() * batch_size
            step_size += batch_size
            epoch_loss += loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = step_loss / step_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                
                step_loss, step_size = 0, 0
                start_time = time.time()
   
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
             for i_batch, samples in enumerate(loader):
                image = samples['image']
                label = samples['label'].squeeze(-1)   # if num of labels is 1
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        image, label = image.cuda(), label.cuda()
                        label = label.long()
                        
                batch_size = image.size(0)
                preds = model(image)
                total_loss += criterion(preds, label).item() * batch_size

                # Collect the results
                preds_label = torch.argmax(preds, -1)
                results.append(preds_label)
                truths.append(label)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        return avg_loss, results, truths

    #writer = SummaryWriter()
    
    best_valid = 1e8
    '''
    checkpoint = 1
    if checkpoint:
        val_loss, _, _ = evaluate(model, criterion, test=False)
        best_valid = val_loss
    '''
    
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        
        #writer.add_scalar("Loss/train", train_loss, epoch) 
        #writer.add_scalar("Loss/val", val_loss, epoch)
        #writer.add_scalar("Loss/test", test_loss, epoch)   
        
        end = time.time()
        duration = end-start
        scheduler.step()    # Decay learning rate by 10 times every 30 epochs with StepLR
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print("Best validation - Model saved!")
            torch.save(model.state_dict(), '/mnt/hard2/bella/cifar10/pretrained_models/augmentplus32_bs64_adamw1e-4.pt')
            best_valid = val_loss
    
    model.load_state_dict(torch.load('/mnt/hard2/bella/cifar10/pretrained_models/augmentplusn32_bs64_adamw1e-4.pt'))
    _, results, truths = evaluate(model, criterion, test=True)
   
    results = torch.cat(results)
    truths = torch.cat(truths)
    print(results.shape, truths.shape)
    test_preds = results.cpu().detach().numpy()
    test_truth = truths.cpu().detach().numpy()
    f1_weighted = f1_score(test_truth, test_preds, average='weighted')
    f1_perclass = f1_score(test_truth, test_preds, average=None)
    acc = accuracy_score(test_truth, test_preds)
    print("  - F1 weighted: ", f1_weighted)
    print("  - F1 per class: ", f1_perclass)
    print("  - Accuracy: ", acc)


