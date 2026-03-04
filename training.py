import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from utils import EarlyStopping, ECGDataset, FocalLoss
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold
import mlflow.pytorch


def save_final_model(model, optimizer, PATH: str, losses: list[float], accuracy: list, sensitivity: list, specificity: list, precision: list, auc: list, epoch=20):
    """
        Used to save a model after training and evaluation
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'precision': precision,
        'specificity': specificity,
        'auc': auc,
        'losses': losses
    }, PATH)

def save_checkpoint(model, optimizer, PATH: str, losses: list[float], epoch=20):
    """
        Used to save a checkpoint of a model during training
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, PATH)

# NOTE: Modify it
def fit(model, device, criterion, optimizer, train_data_loader, channels=False, num_epochs=20):

    losses = []
    n_total_steps = len(train_data_loader)
    # Training
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # print(f"Epoch {epoch}")
        for i, (samples, labels) in enumerate(train_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            # print(samples.size())
            # print("Labels =", labels.reshape(1, -1))
            if channels:
                samples = samples.unsqueeze(1)
                labels = labels.reshape(-1, 1)
       
            # forward
            outputs = model(samples)
            # print("Predictions =",(outputs >= 0.5).int().reshape(1, -1))

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)
            # print("Loss =", loss.item())

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            # print("Running loss =", epoch_loss)

        losses.append(epoch_loss / n_total_steps)

        # if (epoch+1) % num_epochs == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step = [{i+1}/{n_total_steps}], loss = {loss.item():.3f}')  

    return model, losses

def fit_evaluate(model, device, criterion, optimizer, train_data_loader, test_data_loader, channels=False, embedding=False, offset=100, num_epochs=20):
    train_losses = []
    test_losses = []
    n_total_steps = len(train_data_loader)
    n_total_test_steps = len(test_data_loader)
    print("Memory allocated:", torch.cuda.memory_allocated())
    print("Memory reserved:", torch.cuda.memory_reserved())
    print("All model params in cuda:", all(p.is_cuda for p in model.parameters()))
    print("Model is on:", next(model.parameters()).device)

    # Training
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        epoch_loss = 0.0
        # print(f"Epoch {epoch+1}")
        model.train()
        # print("Started training")
        for i, (samples, labels) in enumerate(train_data_loader):

            # print("Labels =", labels.reshape(1, -1))
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
                # labels = labels.reshape(-1, 1)
            elif not embedding:
                samples = samples.unsqueeze(2)
                # labels = labels.reshape(-1, 1)

            # samples = samples[:, ::offset, :].contiguous().to(device)   # NOTE: Taking ony 7500
            samples = samples.to(device)
            labels = labels.to(device)

            # print("Samples device:", samples.device)
            # print("Labels device:", labels.device)

            # print("Samples size:", samples.size())
            # print("samples contiguous:", samples.is_contiguous())
            # print("labels size:", labels.size())
            # print("Labels =", labels.reshape(1, -1))
    
            # forward
            
            outputs = model(samples).reshape(-1, 1)
            # print("Predictions =",(outputs >= 0.5).int().reshape(1, -1))
            # outputs = outputs.reshape(-1, 1)

            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / n_total_steps
        train_losses.append(train_loss)

        print(f"Train loss: {train_loss}")

        epoch_loss = 0.0
        model.eval()
        for i, (samples, labels) in enumerate(test_data_loader):
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
                # labels = labels.reshape(-1, 1)
            elif not embedding:
                samples = samples.unsqueeze(2)
                # labels = labels.reshape(-1, 1)

            # samples = samples[:, ::offset, :].to(device)
            samples = samples.to(device)
            labels = labels.to(device)

            outputs = model(samples)

            outputs = outputs.reshape(-1, 1)

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

        test_loss = epoch_loss / n_total_test_steps
        test_losses.append(test_loss)
        print(f"Test loss: {test_loss}")


    return model, train_losses, test_losses

def fit_evaluate_with_recall(model, device, criterion, optimizer, train_data_loader, test_data_loader, channels=False, embedding=False, offset=100, num_epochs=20):
    train_losses = []
    test_losses = []
    recalls = []
    n_total_steps = len(train_data_loader)
    n_total_test_steps = len(test_data_loader)
    print("Memory allocated:", torch.cuda.memory_allocated())
    print("Memory reserved:", torch.cuda.memory_reserved())
    print("All model params in cuda:", all(p.is_cuda for p in model.parameters()))
    print("Model is on:", next(model.parameters()).device)

    # Training
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:

    sig = nn.Sigmoid()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        print(f"Epoch {epoch+1}")
        model.train()
        # print("Started training")
        for i, (samples, labels) in enumerate(train_data_loader):

            # print("Labels =", labels.reshape(1, -1))
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
                # labels = labels.reshape(-1, 1)
            elif not embedding:
                samples = samples.unsqueeze(2)
                # labels = labels.reshape(-1, 1)

            # samples = samples[:, ::offset, :].contiguous().to(device)   # NOTE: Taking ony 7500
            samples = samples.to(device)
            labels = labels.to(device)

            # print("Samples device:", samples.device)
            # print("Labels device:", labels.device)

            # print("Samples size:", samples.size())
            # print("samples contiguous:", samples.is_contiguous())
            # print("labels size:", labels.size())
            # print("Labels =", labels.reshape(1, -1))
    
            # forward
            
            outputs = model(samples).reshape(-1, 1)
            # print("Predictions =",(outputs >= 0.5).int().reshape(1, -1))
            # outputs = outputs.reshape(-1, 1)

            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / n_total_steps)

        epoch_loss = 0.0
        model.eval()
        for i, (samples, labels) in enumerate(test_data_loader):
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
                # labels = labels.reshape(-1, 1)
            elif not embedding:
                samples = samples.unsqueeze(2)
                # labels = labels.reshape(-1, 1)

            # samples = samples[:, ::offset, :].to(device)
            samples = samples.to(device)
            labels = labels.to(device)

            outputs = model(samples)

            outputs = outputs.reshape(-1, 1)

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            predicted = (sig(outputs) >= 0.5).int()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)

        recalls.append(recall_score(all_labels, all_preds, average='binary', zero_division=0.0))
        test_losses.append(epoch_loss / n_total_test_steps)

        # print(prof)

    return model, train_losses, test_losses, recalls

def fit_evaluate_stop(model, device, criterion, optimizer, train_data_loader, test_data_loader, channels=False, embedding=False, num_epochs=20, patience=10, delta=0.01):
    """
        Used to fit a model and evaluate it at the same time + implements early stoping to avoid overfitting
        
        Return: 
            model (nn.Module): the model trained until early stopping
            train_losses (list): records of train losses
            validation_losses (list): records of validation losses
            epoch (int): last epoch reached
    """
    train_losses = []
    test_losses = []
    n_total_steps = len(train_data_loader)
    n_total_test_steps = len(test_data_loader)
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    # Training
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for i, (samples, labels) in enumerate(train_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)
       
            # forward
            outputs = model(samples)

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / n_total_steps
        train_losses.append(train_loss)

        epoch_loss = 0.0
        model.eval()
        for i, (samples, labels) in enumerate(test_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)
       
            outputs = model(samples)

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

        val_loss = epoch_loss / n_total_test_steps
        test_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)

    return model, train_losses, test_losses, epoch

def fit_evaluate_stop_with_recall(model, device, criterion, optimizer, train_data_loader, test_data_loader, channels=False, embedding=False, num_epochs=20, patience=10, delta=0.01):
    """
        Used to fit a model and evaluate it at the same time + implements early stoping over recall (or another metric) to avoid overfitting

        Return: 
            best_model (nn.Module): the best model encountered until early stopping
            train_losses (list): records of train losses
            validation_losses (list): records of validation losses
            epoch (int): where the training stopped
            recalls (list): all recalls recorded during training before stopping
            early_stopping (EarlyStopping): the early stopping object
    """
    train_losses = []
    test_losses = []
    recalls = []
    n_total_steps = len(train_data_loader)
    n_total_test_steps = len(test_data_loader)
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    sig = nn.Sigmoid()
    # Training
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_correct = 0
        all_preds = []
        all_labels = []
        model.train()
        for i, (samples, labels) in enumerate(train_data_loader):
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}/{n_total_steps}")
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)
       
            # forward
            outputs = model(samples)

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / n_total_steps
        train_losses.append(train_loss)

        epoch_loss = 0.0
        n_samples = 0
        model.eval()
        for i, (samples, labels) in enumerate(test_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)
            n_samples += len(samples)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)
       
            outputs = model(samples)

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            predicted = (sig(outputs) >= 0.5).int()
            n_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_labels = torch.tensor(all_labels).flatten()
        all_preds = torch.tensor(all_preds).flatten()
        # all_probs = torch.tensor(all_probs).flatten()

        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0.0)
        recalls.append(recall)
        val_loss = epoch_loss / n_total_test_steps
        test_losses.append(val_loss)
        accuracy = 100 * n_correct / n_samples
        # sensitivity = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_labels == 1).float().sum()
        precision = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_preds == 1).float().sum()
        specificity = ((all_preds == 0) & (all_labels == 0)).float().sum() / (all_labels == 0).float().sum()

        f1 = f1_score(all_labels, all_preds)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Specificity: {specificity:.4f}, f1-score: {f1:.4f}\n')

        if (epoch+1) % 10 == 0:
            torch.save(early_stopping.best_model_state, f'./models/dataset/Segmentation/partial/OneSeqTransfomer 2500-ALL Segmentation 10level separated 50epochs 0.2dropout 0.001lr {epoch+1}epoch best.pth')

        early_stopping(-f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model = early_stopping.load_best_model(model)
    
    return model, train_losses, test_losses, epoch, recalls, early_stopping

def fit_evaluate_stop_with_recall_mlflow(model, device, criterion, optimizer, train_data_loader, test_data_loader, channels=False, embedding=False, num_epochs=20, patience=10, delta=0.01, training_type="mixted", model_name="OneSeqTransformer", aug_technique="GAN"):
    """
        Used to fit a model and evaluate it at the same time + implements early stoping over the recall (or another metric) to avoid overfitting. Stores the results in mlflow for a better experiment tracking.
        
        Return: 
            model (nn.Module): the model trained until early stopping
            best_model (nn.Module): the best model encountered until early stopping
            train_losses (list): records of train losses
            validation_losses (list): records of validation losses
    """
    train_losses = []
    test_losses = []
    recalls = []
    n_total_steps = len(train_data_loader)
    n_total_test_steps = len(test_data_loader)
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    sig = nn.Sigmoid()
    # Training
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_correct = 0
        all_preds = []
        all_labels = []
        model.train()
        for i, (samples, labels) in enumerate(train_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)
       
            # forward
            outputs = model(samples)

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / n_total_steps
        train_losses.append(train_loss)

        epoch_loss = 0.0
        n_samples = 0
        model.eval()
        for i, (samples, labels) in enumerate(test_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)
            n_samples += len(samples)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)
       
            outputs = model(samples)

            if channels:
                outputs = outputs.reshape(-1, 1)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            predicted = (sig(outputs) >= 0.5).int()
            n_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_labels = torch.tensor(all_labels).flatten()
        all_preds = torch.tensor(all_preds).flatten()
        # all_probs = torch.tensor(all_probs).flatten()

        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0.0)
        recalls.append(recall)
        val_loss = epoch_loss / n_total_test_steps
        test_losses.append(val_loss)
        accuracy = 100 * n_correct / n_samples
        # sensitivity = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_labels == 1).float().sum()
        precision = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_preds == 1).float().sum()
        specificity = ((all_preds == 0) & (all_labels == 0)).float().sum() / (all_labels == 0).float().sum()

        f1 = f1_score(all_labels, all_preds)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Specificity: {specificity:.4f}, f1-score: {f1:.4f}\n')

        mlflow.log_metrics({
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1-score": f1
        }, step=epoch+1)

        with open(f'./training_logs/{model_name}/{training_type}_training_log.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')
            f.write(f'Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Specificity: {specificity:.4f}, f1-score: {f1:.4f}\n\n')

        if (epoch+1) % 10 == 0:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"{model_name}-{aug_technique}-{training_type}-{epoch+1}",  # <-- this is required
                registered_model_name=f"{model_name}-{aug_technique}-{training_type}-{epoch+1}",  # optional, for registry
            )
            torch.save(early_stopping.best_model_state, f'./models/dataset/{aug_technique}/partial/{model_name} 2500-HRC90 {aug_technique} 11level {training_type} {num_epochs}epochs 0.2dropout 0.001lr {epoch+1}epoch best.pth')

        early_stopping(-f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model = early_stopping.load_best_model(model)
    mlflow.log_artifact(f"./training_logs/{model_name}/{training_type}_training_log.txt")
    torch.save(early_stopping.best_model_state, f'./models/dataset/{aug_technique}/full/{model_name} 2500-HRC90 {aug_technique} 11level {training_type} {num_epochs}epochs 0.2dropout 0.001lr best.pth')
    mlflow.log_artifact(f"./models/dataset/{aug_technique}/full/{model_name} 2500-HRC90 {aug_technique} 11level {training_type} {num_epochs}epochs 0.2dropout 0.001lr best.pth")
    
    return model, train_losses, test_losses, epoch, recalls, early_stopping


def model_evaluation(model, criterion, data, data_loader, channels=False, embedding=False, sigmoid=False):
    all_labels = []
    all_preds = []
    all_probs = []
    auc = 0.0
    total_loss = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = len(data)

    if not sigmoid:
        sig = nn.Sigmoid()

    with torch.no_grad():
        n_correct = 0

        for samples, labels in data_loader:
            samples = samples.to(device).to(torch.float32)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1)

            if channels:
                samples = samples.unsqueeze(1)
            elif not embedding:
                samples = samples.unsqueeze(2)

            # print(labels.reshape(1, -1))
            
            outputs = model(samples)

            if sigmoid:
                predicted = (outputs >= 0.5).int()
            else:
                predicted = (sig(outputs) >= 0.5).int()
            if channels:
                predicted = predicted.reshape(-1, 1)
                outputs = outputs.reshape(-1, 1)
            n_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels.float())

            total_loss += loss.item()

            # Store labels, predictions, and probabilities for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    # Convert lists to tensors/arrays
    all_labels = torch.tensor(all_labels).flatten()
    all_preds = torch.tensor(all_preds).flatten()
    all_probs = torch.tensor(all_probs).flatten()

    # Calculate metrics
    accuracy = 100 * n_correct / n_samples
    sensitivity = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_labels == 1).float().sum()
    precision = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_preds == 1).float().sum()
    specificity = ((all_preds == 0) & (all_labels == 0)).float().sum() / (all_labels == 0).float().sum()
    try:
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
    except ValueError:
        pass    # This would mean that this set has only one class (not such a good solution)
    total_loss = total_loss / len(data_loader)

    f1 = f1_score(all_labels, all_preds)


    return accuracy, sensitivity, precision, specificity, auc, all_labels, all_preds, total_loss, f1

def cross_validation(model, data, criterion=FocalLoss(alpha=0.6, gamma=2), learning_rate=0.01, target='label', batch_size=64, num_epochs=30, objective='min', n_splits=5, shuffle=True, random_state=123, type='kfold'):
    fold_metrics = []
    accuracies = []
    sensitivities = []
    precisions = []
    specificities = []
    aucs = []
    f1s = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if type != 'stratified':
        if type == 'kfold':
            kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif type == 'temporal':
            kfold = TimeSeriesSplit(n_splits=n_splits)

        for train_index, test_index in kfold.split(data):
            cv_train, cv_test = pd.DataFrame(data.iloc[train_index]), pd.DataFrame(data.iloc[test_index])

            cv_train = ECGDataset(data=cv_train)
            cv_test = ECGDataset(data=cv_test)

            train_data_loader = DataLoader(cv_train, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(cv_test, batch_size=batch_size, shuffle=False)
            
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

            model.train()
            model, _ = fit(model=model, device=device, optimizer=optimizer, train_data_loader=train_data_loader, criterion=criterion, num_epochs=num_epochs)

            model.eval()
            accuracy, sensitivity, precision, specificity, auc, _, _, test_loss, f1 = model_evaluation(model=model, criterion=criterion, data=cv_test, data_loader=test_data_loader, sigmoid=False)

            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            precisions.append(precision)
            specificities.append(specificity)
            aucs.append(auc)
            fold_metrics.append(test_loss)
            f1s.append(f1)
    else:
        kfold = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in kfold.split(data, data[target]):
            cv_train, cv_test = pd.DataFrame(data.iloc[train_index]), pd.DataFrame(data.iloc[test_index])

            cv_train = ECGDataset(data=cv_train)
            cv_test = ECGDataset(data=cv_test)

            train_data_loader = DataLoader(cv_train, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(cv_test, batch_size=batch_size, shuffle=False)
            
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

            model.train()
            model, _ = fit(model=model, device=device, optimizer=optimizer, train_data_loader=train_data_loader, criterion=criterion, num_epochs=num_epochs)

            model.eval()
            accuracy, sensitivity, precision, specificity, auc, _, _, test_loss, f1 = model_evaluation(model=model, criterion=criterion, data=cv_test, data_loader=test_data_loader, sigmoid=False)

            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            precisions.append(precision)
            specificities.append(specificity)
            aucs.append(auc)
            fold_metrics.append(test_loss)
            f1s.append(f1)
    
    mean_result = np.mean(fold_metrics)
    if objective == 'min':
        overall = mean_result + np.std(fold_metrics)
    else:
        overall = mean_result - np.std(fold_metrics)

    results = pd.DataFrame({
        'accuracy': accuracies,
        'sensitivity': sensitivities,
        'precision': precisions,
        'specificity': specificities,
        'auc': aucs,
        'f1-score': f1s,
        'avg_test_loss': fold_metrics,
    })
    return results, fold_metrics, mean_result, overall



# def model_evaluation(model, criterion, data, data_loader, sigmoid=False):
#     """
#         Evaluates a DL model based on a specific criterion on a specific dataset (train or test).

#         Return: 
#             - accuracy (float)
#             - sensitivity (float)
#             - precision (float)
#             - specificity (float)
#             - auc (float)
#             - all_labels (list): all real labels from the dataset
#             - all_preds (list): all predictions made by the model on the dataset
#             - total_loss (float)
#             - f1-score (float)
#     """
#     all_labels = []
#     all_preds = []
#     all_probs = []
#     auc = 0.0
#     total_loss = 0.0

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     n_samples = len(data)

#     if not sigmoid:
#         sig = nn.Sigmoid()

#     with torch.no_grad():
#         n_correct = 0

#         for samples, labels in data_loader:
#             samples.to(device)
#             labels.to(device)

#             samples = samples.unsqueeze(1)
#             labels = labels.reshape(-1, 1)            
#             outputs = model(samples)

#             if sigmoid:
#                 predicted = (outputs >= 0.5).int().reshape(-1, 1)
#             else:
#                 predicted = (sig(outputs) >= 0.5).int().reshape(-1, 1)
#             n_correct += (predicted == labels).sum().item()

#             loss = criterion(outputs.reshape(-1, 1), labels)

#             total_loss += loss.item()

#             # Store labels, predictions, and probabilities for metrics
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#             all_probs.extend(outputs.cpu().numpy())

#     # Convert lists to tensors/arrays
#     all_labels = torch.tensor(all_labels).flatten()
#     all_preds = torch.tensor(all_preds).flatten()
#     all_probs = torch.tensor(all_probs).flatten()

#     # Calculate metrics
#     accuracy = 100 * n_correct / n_samples
#     sensitivity = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_labels == 1).float().sum()
#     precision = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_preds == 1).float().sum()
#     specificity = ((all_preds == 0) & (all_labels == 0)).float().sum() / (all_labels == 0).float().sum()
#     try:
#         auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
#     except ValueError:
#         pass    # This would mean that this set has only one class (not such a good solution)
#     total_loss = total_loss / len(data_loader)

#     f1 = f1_score(all_labels, all_preds)

#     return accuracy, sensitivity, precision, specificity, auc, all_labels, all_preds, total_loss, f1

# def model_evaluation(model, criterion, data, data_loader, channels=False, sigmoid=False):
#     all_labels = []
#     all_preds = []
#     all_probs = []
#     auc = 0.0
#     total_loss = 0.0

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     n_samples = len(data)

#     if not sigmoid:
#         sig = nn.Sigmoid()

#     with torch.no_grad():
#         n_correct = 0

#         for samples, labels in data_loader:
#             samples = samples.to(device).to(torch.float32)
#             labels = labels.to(device)
#             if channels:
#                 samples = samples.unsqueeze(1)
#             labels = labels.reshape(-1, 1)

#             # print(labels.reshape(1, -1))
            
#             outputs = model(samples)

#             if sigmoid:
#                 predicted = (outputs >= 0.5).int()
#             else:
#                 predicted = (sig(outputs) >= 0.5).int()
#             if channels:
#                 predicted = predicted.reshape(-1, 1)
#                 outputs = outputs.reshape(-1, 1)
#             n_correct += (predicted == labels).sum().item()

#             loss = criterion(outputs, labels.float())

#             total_loss += loss.item()

#             # Store labels, predictions, and probabilities for metrics
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#             all_probs.extend(outputs.cpu().numpy())

#     # Convert lists to tensors/arrays
#     all_labels = torch.tensor(all_labels).flatten()
#     all_preds = torch.tensor(all_preds).flatten()
#     all_probs = torch.tensor(all_probs).flatten()

#     # Calculate metrics
#     accuracy = 100 * n_correct / n_samples
#     sensitivity = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_labels == 1).float().sum()
#     precision = ((all_preds == 1) & (all_labels == 1)).float().sum() / (all_preds == 1).float().sum()
#     specificity = ((all_preds == 0) & (all_labels == 0)).float().sum() / (all_labels == 0).float().sum()
#     try:
#         auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
#     except ValueError:
#         pass    # This would mean that this set has only one class (not such a good solution)
#     total_loss = total_loss / len(data_loader)

#     f1 = f1_score(all_labels, all_preds)


#     return accuracy, sensitivity, precision, specificity, auc, all_labels, all_preds, total_loss, f1