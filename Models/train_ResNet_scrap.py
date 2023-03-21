#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:19:33 2023

@author: gliu
"""

def train_ResNet(model,loss_fn,optimizer,trainloader,testloader,max_epochs,early_stop=False,verbose=True,
                 reduceLR=False,LRpatience=3):
    """
    inputs:
        model       - Resnet model
        loss_fn     - (torch.nn) loss function
        opt         - tuple of [optimizer_name, learning_rate, weight_decay] for updating the weights
                      currently supports "Adadelta" and "SGD" optimizers
        trainloader - (torch.utils.data.DataLoader) for training datasetmo
        testloader  - (torch.utils.data.DataLoader) for testing dataset
        max_epochs  - number of training epochs
        early_stop  - BOOL or INT, Stop training after N epochs of increasing validation error
                     (set to False to stop at max epoch, or INT for number of epochs)
        verbose     - set to True to display training messages
        reduceLR    - BOOL, set to true to use LR scheduler
        LRpatience  - INT, patience for LR scheduler

    output:

    dependencies:
        from torch import nn,optim

    """
    # Check if there is GPU
    if checkgpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # Get list of params to update
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # if verbose:
            #     print("Params to learn:")
            #     print("\t",name)

    # Set optimizer
    if optimizer[0] == "Adadelta":
        opt = optim.Adadelta(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == "SGD":
        opt = optim.SGD(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    elif optimizer[0] == 'Adam':
        opt = optim.Adam(model.parameters(),lr=optimizer[1],weight_decay=optimizer[2])
    
    # Add Scheduler
    if reduceLR:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LRpatience)
    
    # Set early stopping threshold and counter
    if early_stop is False:
        i_thres = max_epochs
    else:
        i_thres = early_stop
    i_incr    = 0 # Number of epochs for which the validation loss increases
    bestloss  = np.infty

    # Main Loop
    train_acc,test_acc = [],[] # Preallocate tuples to store accuracy
    train_loss,test_loss = [],[]   # Preallocate tuples to store loss
    bestloss = np.infty
    
    for epoch in tqdm(range(max_epochs)): # loop by epoch
        for mode,data_loader in [('train',trainloader),('eval',testloader)]: # train/test for each epoch
            if mode == 'train':  # Training, update weights
                model.train()
            elif mode == 'eval': # Testing, freeze weights
                model.eval()
            
            runningloss = 0
            correct     = 0
            total       = 0
            for i,data in enumerate(data_loader):
                # Get mini batch
                batch_x, batch_y = data
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # Set gradients to zero
                opt.zero_grad()
                
                # Forward pass
                pred_y = model(batch_x)
                
                # Calculate loss
                loss = loss_fn(pred_y,batch_y[:,0])
                
                # Track accuracy
                _,predicted = torch.max(pred_y.data,1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y[:,0]).sum().item()
                #print("Total is now %.2f, Correct is now %.2f" % (total,correct))
                
                # Update weights
                if mode == 'train':
                    loss.backward() # Backward pass to calculate gradients w.r.t. loss
                    opt.step()      # Update weights using optimizer
                elif mode == 'eval':  # update scheduler after 1st epoch
                    if reduceLR:
                        scheduler.step(loss)
                    
                runningloss += float(loss.item())
            
            if verbose: # Print progress message
                print('{} Set: Epoch {:02d}. loss: {:3f}. acc: {:.3f}%'.format(mode, epoch+1, \
                                                runningloss/len(data_loader),correct/total*100))

            # Save model if this is the best loss
            if (runningloss/len(data_loader) < bestloss) and (mode == 'eval'):
                bestloss = runningloss/len(data_loader)
                bestmodel = copy.deepcopy(model)
                if verbose:
                    print("Best Loss of %f at epoch %i"% (bestloss,epoch+1))

            # Save running loss values for the epoch
            if mode == 'train':
                train_loss.append(runningloss/len(data_loader))
                train_acc.append(correct/total)
            else:
                test_loss.append(runningloss/len(data_loader))
                test_acc.append(correct/total)

                # Evaluate if early stopping is needed
                if epoch == 0: # Save previous loss
                    lossprev = runningloss/len(data_loader)
                else: # Add to counter if validation loss increases
                    if runningloss/len(data_loader) > lossprev:
                        i_incr += 1 # Add to counter
                        if verbose:
                            print("Validation loss has increased at epoch %i, count=%i"%(epoch+1,i_incr))
                        
                    else:
                        i_incr = 0 # Zero out counter
                    lossprev = runningloss/len(data_loader)

                if (epoch != 0) and (i_incr >= i_thres):
                    print("\tEarly stop at epoch %i "% (epoch+1))
                    return bestmodel,train_loss,test_loss,train_acc,test_acc

            # Clear some memory
            #print("Before clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))
            del batch_x
            del batch_y
            torch.cuda.empty_cache()
            #print("After clearing in epoch %i mode %s, memory is %i"%(epoch,mode,torch.cuda.memory_allocated(device)))

    #bestmodel.load_state_dict(best_model_wts)
    return bestmodel,train_loss,test_loss,train_acc,test_acc