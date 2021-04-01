"""
autosklearn calc baseline metrics

Apply autosklearn to the AMV prediction problem

Uses data that has been preprocessed by "output_normalized_data.ipynb"
in /Preprocessing
    Assumes data is stored in ../../CESM_data/

Outputs are stored in 
    - ../../CESM_data/Metrics (Experimental Metrics (ex. Acc))
"""

import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier

# ----------------
#%% User Settings
# ----------------

# Setup
ml_type        = 'classification'     # 'regression' or 'classification'
leads          = np.arange(0,25,1)    # Prediction lead times (in years)
nsamples       = 300                  # Number of samples for each class
thresholds     = [-1,1]               # Thresholds (standard deviations, determines number of classes)

# Autosklearn Settings
time_left_for_this_task = 3600 # Number of seconds for autosklearn
per_run_time_limit      = 300  # autosklearn per run time limit

outpath = "../../CESM_data/Metrics/"
# -----------------------------------------
# %% Legacy Variables (Don't edit for now)
# -----------------------------------------

detrend            = False                # Set to true to use detrended data
varname            = 'ALL'                # Use all variables (SST, SSS, PSL)
num_classes        = len(thresholds)+1    # Set up number of classes for prediction (current supports)
percent_train      = 0.8   # Percentage of data to use for training (remaining for testing)
ens                = 40    # Ensemble members to use
tstep              = 86    # Size of time dimension (in years)


# ----------------
# %% Functions
# ----------------

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
    if exact_value is False: # Scale thresholds by standard deviation
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

# ----------------
#%% Script Start
# ----------------

# Preallocate based on problem type
if ml_type=='regression':
    metrics = np.zeros( (leads.shape[0]) )
elif ml_type=='classification':
    metrics = np.zeros( (leads.shape[0],num_classes) )

# Load Data
data   = np.load('../../CESM_data/CESM_data_sst_sss_psl_deseason_normalized_resized_detrend%i.npy'%detrend)
target = np.load('../../CESM_data/CESM_label_amv_index_detrend%i.npy'%detrend)
data   = data[:,0:ens,:,:,:]
target = target[0:ens,:]


##############################################################################
##############################################################################
##############################################################################

# ----------------
#%% Run AutoML
# ----------------
for l,lead in enumerate(leads):
    if ml_type=='regression':
        y = target[:ens,lead:].reshape(ens*(tstep-lead))
        X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),224,224).transpose(1,0,2,3)
        X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
        lead_nsamples      = y.shape[0]
        y_train = y[0:int(np.floor(percent_train*lead_nsamples))]
        y_val   = y[int(np.floor(percent_train*lead_nsamples)):] 

    elif ml_type=='classification':
        y = target[:ens,lead:].reshape(ens*(tstep-lead),1)
        X = (data[:,:ens,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),224,224).transpose(1,0,2,3)
        y_class = make_classes(y,thresholds,reverse=True)
        y_class,X,shuffidx = select_samples(nsamples,y_class,X)
        X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
        lead_nsamples      = y_class.shape[0]
        y_train = y_class[0:int(np.floor(percent_train*lead_nsamples)),0]
        y_val = y_class[int(np.floor(percent_train*lead_nsamples)):,0]

    X_train = X[0:int(np.floor(percent_train*lead_nsamples)),:]
    X_val = X[int(np.floor(percent_train*lead_nsamples)):,:]

    # define search
    if ml_type=='regression':
        model = AutoSklearnRegressor(time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit = per_run_time_limit,
                n_jobs=1,
                memory_limit=1000000,
                tmp_folder=outpath+'log_folder',
                output_folder=outpath+'output_folder',

                )
    elif ml_type=='classification':
        model = AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit = per_run_time_limit,
                n_jobs=1,
                memory_limit=1000000,
                tmp_folder=outpath+'log_folder',
                output_folder=outpath+'output_folder',
                )
    print("start searching")
    
    # perform the search
    model.fit(X_train, y_train, dataset_name=ml_type+'_t'+str(time_left_for_this_task)+'_lead'+str(l))
    
    # summarize
    file = open('log_files/'+ml_type+'_t'+str(time_left_for_this_task)+'_lead'+str(l)+'.txt','w')

    file.write(model.sprint_statistics())
    file.write('\n')
    file.write(model.show_models())
    file.close() 
    
    print(model.sprint_statistics())
    print(model.show_models()) 
    # evaluate best model
    y_hat = model.predict(X_val)
    metric = calc_metrics(y_val, y_hat, ml_type)
    if ml_type=='regression':
        metrics[l] = metric
    elif ml_type=='classification':
        metrics[l,:] = metric
    print("************************************")
    print("lead:"+str(l)+", metric: "+str(metric))
    print("************************************")

print("**************************")
print(metrics)
np.save(outpath+"automl_accuracy_detrend_t"+str(time_left_for_this_task)+"_"+ml_type+".npy",metrics)
