import numpy as np
import autokeras as ak


ml_type        = 'classification'     # 'regression' or 'classification'
leads          = np.arange(0,2,1)
max_trials     = 1

detrend        = False                # Set to true to use detrended data
varname        = 'ALL'
thresholds     = [-1,1]               # Thresholds (standard deviations, determines number of classes)
num_classes    = len(thresholds)+1    # Set up number of classes for prediction (current supports)
metrics = np.zeros( (leads.shape[0],num_classes) )
nsamples       = 300                  # Number of samples for each class

# Training/Testing Subsets
percent_train = 0.8   # Percentage of data to use for training (remaining for testing)
ens           = 40    # Ensemble members to use
tstep         = 86    # Size of time dimension (in years)

data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]


def calc_metrics(y_val, y_hat, ml_type):
    if ml_type=='regression':
        return np.corrcoef(y_val, y_hat)[0,1]
    elif ml_type=='classification':
        # Calculate accuracy for each class
        class_total   = np.zeros([num_classes])
        class_correct = np.zeros([num_classes])
        val_size = y_val.shape[0]
        for i in range(val_size):
            class_idx  = int(y_val[i])
            check_pred = y_val[i] == y_hat[i]
            class_total[class_idx]   += 1
            class_correct[class_idx] += check_pred 
        class_acc = class_correct/class_total
        return class_acc

def make_classes(y,thresholds,exact_value=False,reverse=False):
    """
    Makes classes based on given thresholds. 
    Parameters
    ----------
    y : ARRAY
        Labels to classify
    thresholds : ARRAY
        1D Array of thresholds to partition the data
    exact_value: BOOL, optional
        Set to True to use the exact value in thresholds (rather than scaling by
                                                          standard deviation)
    Returns
    -------
    y_class : ARRAY [samples,class]
        Classified samples, where the second dimension contains an integer
        representing each threshold
    """
    nthres = len(thresholds)
    if ~exact_value: # Scale thresholds by standard deviation
        y_std = np.std(y) # Get standard deviation
        thresholds = np.array(thresholds) * y_std
    y_class = np.zeros((y.shape[0],1))
    
    if nthres == 1: # For single threshold cases
        thres = thresholds[0]
        y_class[y<=thres] = 0
        y_class[y>thres] = 1
        
        print("Class 0 Threshold is y <= %.2f " % (thres))
        print("Class 0 Threshold is y > %.2f " % (thres))
        return y_class
    
    for t in range(nthres+1):
        if t < nthres:
            thres = thresholds[t]
        else:
            thres = thresholds[-1]
        
        if reverse: # Assign class 0 to largest values
            tassign = nthres-t
        else:
            tassign = t
        
        if t == 0: # First threshold
            y_class[y<=thres] = tassign
            print("Class %i Threshold is y <= %.2f " % (tassign,thres))
        elif t == nthres: # Last threshold
            y_class[y>thres] = tassign
            print("Class %i Threshold is y > %.2f " % (tassign,thres))
        else: # Intermediate values
            thres0 = thresholds[t-1]
            y_class[(y>thres0) * (y<=thres)] = tassign
            print("Class %i Threshold is %.2f < y <= %.2f " % (tassign,thres0,thres))
    return y_class


def select_samples(nsamples,y_class,X):
    """
    Sample even amounts from each class
    Parameters
    ----------
    nsample : INT
        Number of samples to get from each class
    y_class : ARRAY [samples x 1]
        Labels for each sample
    X : ARRAY [samples x channels x height x width]
        Input data for each sample
    
    Returns
    -------
    
    y_class_sel : ARRAY [samples x 1]
        Subsample of labels with equal amounts for each class
    X_sel : ARRAY [samples x channels x height x width]
        Subsample of inputs with equal amounts for each class
    idx_sel : ARRAY [samples x 1]
        Indices of selected arrays
    
    """
    
    allsamples,nchannels,H,W = X.shape
    classes    = np.unique(y_class)
    nclasses   = len(classes)
    

    # Sort input by classes
    label_by_class  = []
    input_by_class  = []
    idx_by_class    = []
    
    y_class_sel = np.zeros([nsamples*nclasses,1])#[]
    X_sel       = np.zeros([nsamples*nclasses,nchannels,H,W])#[]
    idx_sel     = np.zeros([nsamples*nclasses]) 
    for i in range(nclasses):
        
        # Sort by Class
        inclass = classes[i]
        idx = (y_class==inclass).squeeze()
        sel_label = y_class[idx,:]
        sel_input = X[idx,:,:,:]
        sel_idx = np.where(idx)[0]
        
        label_by_class.append(sel_label)
        input_by_class.append(sel_input)
        idx_by_class.append(sel_idx)
        classcount = sel_input.shape[0]
        print("%i samples found for class %i" % (classcount,inclass))
        
        # Shuffle and select first nsamples
        shuffidx = np.arange(0,classcount,1)
        np.random.shuffle(shuffidx)
        shuffidx = shuffidx[0:nsamples]
        
        # Select Shuffled Indices
        y_class_sel[i*nsamples:(i+1)*nsamples,:] = sel_label[shuffidx,:]
        X_sel[i*nsamples:(i+1)*nsamples,...]     = sel_input[shuffidx,...]
        idx_sel[i*nsamples:(i+1)*nsamples]       = sel_idx[shuffidx]
    
    # Shuffle samples again before output (so they arent organized by class)
    shuffidx = np.arange(0,nsamples*nclasses,1)
    np.random.shuffle(shuffidx)
    
    return y_class_sel[shuffidx,...],X_sel[shuffidx,...],idx_sel[shuffidx,...]



##############################################################################
##############################################################################
##############################################################################

for l,lead in enumerate(leads):
    y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
    X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),224,224).transpose(1,0,2,3)
    y_class = make_classes(y,thresholds,reverse=True)
    y_class,X,shuffidx = select_samples(nsamples,y_class,X)
    X = X.transpose(0,2,3,1)
    lead_nsamples      = y_class.shape[0]
    y_train = y_class[0:int(np.floor(percent_train*lead_nsamples)),0]
    y_val = y_class[int(np.floor(percent_train*lead_nsamples)):,0]

    X_train = X[0:int(np.floor(percent_train*lead_nsamples)),:,:,:]
    X_val = X[int(np.floor(percent_train*lead_nsamples)):,:,:,:]


    model = ak.ImageClassifier(num_classes=num_classes, max_trials=max_trials)

    print("start searching")
    
    print(X_train.shape)
    print(y_train.shape)
    # perform the search
    model.fit(X_train, y_train)
    
    model_out = model.export_model()
    model_out.save("autokeras_models/autokeras_lead"+str(l)+".h5")

    # evaluate best model
    y_hat = model.predict(X_val)
    metric = calc_metrics(y_val, y_hat, ml_type)
    
    metrics[l,:] = metric
    print("************************************")
    print("lead:"+str(l)+", metric: "+str(metric))
    print("************************************")

print("**************************")
print(metrics)
np.save("autokeras_accuracy_detrend_"+ml_type+".npy",metrics)

