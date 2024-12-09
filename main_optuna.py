import argparse
import numpy as np
import random
import os
import dill
import time
import optuna

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from helpers.gcn_data_perp import build_graph_data_mean, build_graph_data_sd, get_fold_indices1, get_graph_inputs
import pandas as pd

import torch
from torch_geometric.loader import DataLoader
from models import GCN

def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)

def train_model_dual(models, optimizers, train_dataset_loaders, val_dataset_loaders, loss_fn, fold_model_paths, args):
    print("Number of training samples/batches %d" % len(train_dataset_loaders[0]))
    print("Start training...\n")

    early_stopping_patience = args.patience
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) for optimizer in optimizers]
    acc_best, loss_best = 0, np.inf
    tr_ep_loss, val_ep_loss = [], []
    no_improvement_count = 0
    best_epoch = 0

    for epoch in range(args.num_iter):
        # Training phase for both models
        for i, (model, optimizer, train_dataset_loader) in enumerate(zip(models, optimizers, train_dataset_loaders)):
            model.train()
            tr_gtruth = []
            b_loss = []
            tr_preds = []  # We will collect predictions for each model

            # Transfer model to GPU if available
            model.to(args.device)

            for data in train_dataset_loader:
                optimizer.zero_grad()
                features_cuda = data.x.to(args.device)
                edge_index = data.edge_index.to(args.device)
                edge_attr = data.edge_attr
                tr_y = data.y.to(args.device)

                with torch.set_grad_enabled(True):
                    # Forward pass
                    tr_logits, _ = model(features_cuda, edge_index, edge_attr, data.batch.to(args.device))
                    
                    # Calculate loss and perform backward pass
                    loss = loss_fn(tr_logits, tr_y.to(torch.long))
                    loss.backward()
                    optimizer.step()

                tr_gtruth.extend(tr_y.cpu().numpy())
                tr_preds.extend(tr_logits.detach().cpu().numpy())
                b_loss.append(loss.item())

            tr_ep_loss.append(np.mean(b_loss))

            # Store the predictions separately for each model
            if i == 0:
                tr_preds_1 = tr_preds
            else:
                tr_preds_2 = tr_preds

        # Combine outputs of the two models using averaging
        combined_tr_preds = [(pred1 + pred2) / 2 for pred1, pred2 in zip(tr_preds_1, tr_preds_2)]
        correct_train, acc_train = accuracy(np.array(combined_tr_preds), tr_gtruth)

        # Validation phase for both models
        for i, (model, val_dataset_loader, fold_model_path, scheduler) in enumerate(zip(models, val_dataset_loaders, fold_model_paths, schedulers)):
            model.eval()
            val_gtruth = []
            val_preds = []
            b_loss = []

            with torch.no_grad():
                for data in val_dataset_loader:
                    features_cuda = data.x.to(args.device)
                    edge_index = data.edge_index.to(args.device)
                    edge_attr = data.edge_attr
                    val_y = data.y.to(args.device)
                    
                    # Forward pass
                    _, val_logits = model(features_cuda, edge_index, edge_attr, data.batch.to(args.device))
                    
                    # Calculate loss
                    loss = loss_fn(val_logits, val_y.to(torch.long))
                    b_loss.append(loss.item())
                    
                    # Store predictions and ground truth
                    val_gtruth.extend(val_y.cpu().numpy())
                    val_preds.extend(val_logits.detach().cpu().numpy())

            loss_val = np.mean(b_loss)
            val_ep_loss.append(loss_val)

            # Store predictions separately for each model
            if i == 0:
                val_preds_1 = val_preds
            else:
                val_preds_2 = val_preds

        
        # Combine outputs of the two models using averaging
        combined_val_preds = [(pred1 + pred2) / 2 for pred1, pred2 in zip(val_preds_1, val_preds_2)]
        preds = np.argmax(np.array(combined_val_preds), axis=1)
        val_acc = accuracy_score(val_gtruth, preds)
        val_pre = precision_score(val_gtruth, preds, average='binary', zero_division=0)
        val_sen = recall_score(val_gtruth, preds, average='binary', zero_division=0)
        val_spe = recall_score(val_gtruth, preds, average='binary', pos_label=0)
        val_f1 = f1_score(val_gtruth, preds, average='binary', zero_division=0)
        val_roc = roc_auc_score(val_gtruth, preds)

        # Update learning rate scheduler
        for scheduler in schedulers:
            scheduler.step()

        # Save model if required
        if args.ckpt_path != '':
            for model, fold_model_path in zip(models, fold_model_paths):
                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)
                if os.path.exists(fold_model_path):
                    os.remove(fold_model_path)
                torch.save(model.state_dict(), fold_model_path)

        # Track best accuracy
        if val_acc > acc_best:
            acc_best = val_acc
            no_improvement_count = 0
            best_epoch = epoch
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}, best epoch: {best_epoch}")
            break

    print(f"\r\n => val accuracy {val_acc:.5f}")
    return val_acc, val_pre, val_sen, val_spe, val_f1, val_roc

def test_model_dual(models, ts_loaders, fold_model_paths, args):
    # Initialize variables
    test_acc = test_sen = test_spe = test_pre = test_f1 = test_roc = 0.0
    
    # Lists to store combined ground truth and predictions
    combined_ts_gtruth = []
    combined_ts_preds = []
    
    # Loop through each model, its corresponding test loader, and model path
    for model, ts_loader, fold_model_path in zip(models, ts_loaders, fold_model_paths):
        # Load each model's parameters
        model.load_state_dict(torch.load(fold_model_path))
        model.to(args.device)
        model.eval()
        
        ts_gtruth, ts_preds = [], []
        
        # Disable gradient calculations for evaluation
        with torch.no_grad():
            for data in ts_loader:
                features_cuda = data.x.to(args.device)
                edge_index = data.edge_index.to(args.device)
                edge_attr = data.edge_attr
                ts_y = data.y.to(args.device)
                
                # Get model predictions
                _, ts_logits = model(features_cuda, edge_index, edge_attr, data.batch.to(args.device))
                
                # Store ground truth and predictions
                ts_gtruth.extend(ts_y.cpu().numpy())
                ts_preds.extend(ts_logits.detach().cpu().numpy())
        
        # Combine model predictions by averaging
        if combined_ts_preds:
            combined_ts_preds = [(pred1 + pred2) / 2 for pred1, pred2 in zip(combined_ts_preds, ts_preds)]
        else:
            combined_ts_preds = ts_preds
        
        # Combine ground truth
        if not combined_ts_gtruth:
            combined_ts_gtruth = ts_gtruth

    # Convert combined predictions and ground truth to numpy arrays
    y = np.array(combined_ts_gtruth)
    combined_ts_preds_np = np.array(combined_ts_preds)
    
    # Get final predictions from combined outputs
    preds = np.argmax(combined_ts_preds_np, axis=1)
    
    # Calculate performance metrics
    test_acc = accuracy_score(y, preds)
    test_pre = precision_score(y, preds, average='binary', zero_division=0)
    test_sen = recall_score(y, preds, average='binary', zero_division=0)
    test_spe = recall_score(y, preds, average='binary', pos_label=0)
    test_f1 = f1_score(y, preds, average='binary', zero_division=0)
    test_roc = roc_auc_score(y, preds)
    
    # Return performance metrics
    return test_acc, test_sen, test_spe, test_pre, test_f1, test_roc


def main(args):
    # Sets device to GPU if available
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fc_mat_mean, fc_mat_mean_ad, labels, _ = build_graph_data_mean(args)
    fc_mat_sd, fc_mat_sd_ad, labels2, _ = build_graph_data_sd(args)
    fold_indices = get_fold_indices1(args,labels)
    
    dataset = get_graph_inputs(fc_mat_mean, fc_mat_mean_ad, labels, range(len(labels)), 
                                   args.graph_k)
    dataset2 = get_graph_inputs(fc_mat_sd, fc_mat_sd_ad, labels2, range(len(labels)), 
                                   args.graph_k)
        
    seed_everything(args.seed)
    
    train_ind = fold_indices['outer{}'.format(args.outer_loop)]['train']
    val_ind = fold_indices['outer{}'.format(args.outer_loop)]['val']
    test_ind = fold_indices['outer{}'.format(args.outer_loop)]['test']
    
    tr_dataset = [dataset[i] for i in train_ind]
    ts_dataset = [dataset[i] for i in test_ind]
    va_dataset = [dataset[i] for i in val_ind]
    
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, 
                                shuffle = True, drop_last=True)
    va_loader = DataLoader(va_dataset, batch_size=args.batch_size, 
                            shuffle=False, drop_last=True)
    ts_loader = DataLoader(ts_dataset, batch_size=args.batch_size, 
                            shuffle=False, drop_last=True)
    
    train_ind2 = fold_indices['outer{}'.format(args.outer_loop)]['train']
    val_ind2 = fold_indices['outer{}'.format(args.outer_loop)]['val']
    test_ind2 = fold_indices['outer{}'.format(args.outer_loop)]['test']

    tr_dataset2 = [dataset2[i] for i in train_ind2]
    ts_dataset2 = [dataset2[i] for i in test_ind2]
    va_dataset2 = [dataset2[i] for i in val_ind2]

    tr_loader2 = DataLoader(tr_dataset2, batch_size=args.batch_size, 
                                shuffle = True, drop_last=True)
    va_loader2 = DataLoader(va_dataset2, batch_size=args.batch_size, 
                            shuffle=False, drop_last=True)
    ts_loader2 = DataLoader(ts_dataset2, batch_size=args.batch_size, 
                            shuffle=False, drop_last=True)
    
    
    model = GCN(tr_dataset[0].x.shape[1], args.no_classes, args.dropout,
                    hgc=args.hgc, lg=args.lg, edgenet_input_dim=tr_dataset[0].edge_attr,
                    nrois=args.rois, aggr_method=args.aggr_method).to(args.device)
    model = model.to(args.device)

    model2 = GCN(tr_dataset2[0].x.shape[1], args.no_classes, args.dropout,
                    hgc=args.hgc, lg=args.lg, edgenet_input_dim=tr_dataset2[0].edge_attr,
                    nrois=args.rois, aggr_method=args.aggr_method).to(args.device)
    model2 = model.to(args.device)

    # build loss, optimizer, metric 
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.wd)

    fold_model_path = args.ckpt_path + "/model1-trial{}fold{}.pth".format(args.trial_num, args.outer_loop)
    fold_model_path2 = args.ckpt_path + "/model2-trial{}fold{}.pth".format(args.trial_num, args.outer_loop)

    models = [model, model2]
    optimizers = [optimizer, optimizer2]
    tr_loaders = [tr_loader, tr_loader2]
    va_loaders = [va_loader, va_loader2]
    ts_loaders = [ts_loader, ts_loader2]
    fold_model_paths = [fold_model_path, fold_model_path2]
    
    metrics = []
    if args.train==1:
        val_acc, val_sen, val_spe, val_pre, val_f1, val_roc = train_model_dual(models, optimizers, tr_loaders, va_loaders, loss_fn,
                fold_model_paths, args)
        del model
        test_acc, test_sen, test_spe, test_pre, test_f1, test_roc = test_model_dual(models, ts_loaders, fold_model_paths,  args)
        
        metrics = [[val_acc, val_sen, val_spe, val_pre, val_f1, val_roc], [test_acc, test_sen, test_spe,  test_pre, test_f1, test_roc]]

    return metrics


if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='GCN-Classifier')
    parser.add_argument('--PROC_DATA_DIR', default = './data/fc_data_all_good-85_poor-85.pkl', type=str, help='data path')
    parser.add_argument('--LABELS_DATA_DIR', default = './data/eeg_data_all_good-85_poor-85.xlsx', type=str, help='data path')
    parser.add_argument('--pyg_dataset_path', default = './proc_data/gcn_dataset_all_good-85_poor-85.pkl', type=str, help='graph dataset path')
    parser.add_argument('--DATA_DIR', default = './data', type=str, help='data path')
    parser.add_argument('--RES_DIR', default = './Results/gcn/', type=str, help='Results path')
    parser.add_argument('--no_classes', default = 2, type=int, help='number of classes')
    parser.add_argument('--graph_k', default = 10, type=int, help='number of KNN connections to keep in the graph')
    
    # network parameters  
    parser.add_argument('--hgc', type=int, default=16, help='hidden units of gconv layer')
    parser.add_argument('--lg', type=int, default=4, help='number of gconv layers')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
    parser.add_argument('--num_iter', default=300, type=int, help='number of epochs for training')
    parser.add_argument('--patience', default=60, type=int, help='patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=26, help='batch size')
    parser.add_argument('--dropout', default=0.1, type=float, help='ratio of dropout')
       
    parser.add_argument('--aggr_method', type=str, default='flatten', help='graph embedding aggregation methos mean, flatten, max, min, sum')
        
    # other parameters 
    parser.add_argument('--rois', default = 20, type=int, help='regions of interest / num of nodes')
    parser.add_argument('--seed', default = 3407, type=float, help='seed for random generation')
    parser.add_argument('--ckpt_path', type=str, default='./save_models/gcn/', help='checkpoint path to save trained models')
    parser.add_argument('--device', type=str, default='cpu', help='device to run on')
    parser.add_argument('--train', type=int, default=1, help='training value')
    
    args = parser.parse_args()
    os.makedirs(args.RES_DIR, exist_ok=True) # create the results path

    def objective(trial):
        # Sample hyperparameters
        lg = trial.suggest_categorical('lg', [2, 3, 4])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        aggr = trial.suggest_categorical('aggr', ['flatten', 'mean', 'sum', 'max'])
        dropout = trial.suggest_float('dropout', 0.05, 0.5)
        wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
        hgc = trial.suggest_categorical('hgc', [8, 16, 24])

        # Update args with sampled hyperparameters
        args.lg = lg
        args.lr = lr
        args.aggr_method = aggr
        args.dropout = dropout
        args.wd = wd
        args.hgc = hgc
        args.trial_num = trial.number
    
        all_val_acc = []
        all_test_acc = []
        all_val_results = []
        all_test_results = []
        # Train and evaluate model
        for o_fold in range(1, 6):
            print('-------------- Processing data - outer_fold #:{}'.format(o_fold))
            args.outer_loop = o_fold
            val_results, test_results = main(args)
            all_val_acc.append(val_results[0])
            all_test_acc.append(test_results[0])
            all_val_results.append(val_results)
            all_test_results.append(test_results)

        # Convert the list of validation results to a NumPy array
        all_val_acc = np.array(all_val_acc)
        all_test_acc = np.array(all_test_acc)

        # Calculate the mean of the validation results
        mean_val_results = np.mean(all_val_acc)
        mean_test_results = np.mean(all_test_acc)

        trial.set_user_attr('val_results', all_val_results)
        trial.set_user_attr('test_results', all_test_results)

        # Return the mean validation accuracy as the objective value
        return mean_val_results

    # Time taken to optimize hyperparameters
    start_time = time.time()

    # Initialize Optuna study
    study = optuna.create_study(direction='maximize')

     # Optimize hyperparameters
    study.optimize(objective, n_trials=50)
    
    end_time = time.time()
    print(f"Time taken to initialize Optuna study: {end_time - start_time:.2f} seconds")

    best_trial = study.best_trial

    # Print best parameters and best value
    print("Best parameters:", best_trial.params)
    print("Best value:", best_trial.value)

    best_val_results = best_trial.user_attrs['val_results']
    best_test_results = best_trial.user_attrs['test_results']
    # best_trial_num = best_trial.user_attrs['trial_num']

    excel_writer = pd.ExcelWriter(args.RES_DIR + 'best_results.xlsx', engine='xlsxwriter')
    result_row_index = 0
    results_df = pd.DataFrame([
        [best_trial.number,
         best_trial.params['lg'], 
         best_trial.params['lr'], 
         best_trial.params['aggr'],
         best_trial.params['dropout'], 
         best_trial.params['wd'], 
         best_trial.params['hgc'], 
         ]], columns=['TRIAL_NUM', 'lg', 'lr', 'aggr', 'dropout', 'wd', 'hgc'])
    
    results_df.to_excel(excel_writer, sheet_name='Sheet1', startrow=result_row_index, startcol=0, index=False, header=True)
    # Update row index for next write operation
    result_row_index += 2

    # Create DataFrame to hold performance metrics
    foldsSc = pd.DataFrame(index=['fold'+str(m) for m in range(1,6)], columns=['Acc','Sen','Spe', 'Pre','F1','ROC'])
    foldsSc2 = pd.DataFrame(index=['fold'+str(m) for m in range(1,6)], columns=['Acc','Sen','Spe', 'Pre','F1','ROC'])

    for o_fold in range(5):
        foldsSc.iloc[o_fold,:] = best_val_results[o_fold]
        foldsSc2.iloc[o_fold,:] = best_test_results[o_fold]

    foldsSc = foldsSc*100
    mu = np.mean(foldsSc, axis=0).values
    sd = np.std(foldsSc, axis=0).values
    foldsSc.loc['Val Avg'] = np.zeros(6)
    for col_index in range(6):
        foldsSc.iloc[5, col_index] = '{0:.2f} ({1:.2f})'.format(mu[col_index], sd[col_index])

    foldsSc2 = foldsSc2*100
    mu = np.mean(foldsSc2, axis=0).values
    sd = np.std(foldsSc2, axis=0).values
    foldsSc2.loc['Test Avg'] = np.zeros(6)
    for col_index in range(6):
        foldsSc2.iloc[5, col_index] = '{0:.2f} ({1:.2f})'.format(mu[col_index], sd[col_index])

    # Write the DataFrame to the Excel file
    foldsSc.to_excel(excel_writer, sheet_name='Sheet1', startrow=result_row_index, startcol=0, index=True, header=True)
    result_row_index += len(foldsSc) + 2

    foldsSc2.to_excel(excel_writer, sheet_name='Sheet1', startrow=result_row_index, startcol=0, index=True, header=True)
    result_row_index += len(foldsSc2) + 2

    # Save the Excel file
    excel_writer._save()


