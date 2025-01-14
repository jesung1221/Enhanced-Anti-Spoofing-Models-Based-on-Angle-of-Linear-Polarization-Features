# Filename: train_model_with_metrics.py

import torch
import torch.optim as optim
import torch.nn as nn
from model_definition import initialize_model
from dataset_preparation import AOLPDataset, get_subjects, transform
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming class 1 is the positive class
            all_scores.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_scores

# Set the root directory of your dataset
train_root_dir = '../../../final_data/train/data_with_20sparsity'  # Update this path accordingly
test_root_dir = '../../../final_data/test/data_with_20sparsity_only'

#all_subjects = get_subjects(train_root_dir)

# Retrieve subjects from both directories
all_train_subjects = get_subjects(train_root_dir)
all_test_subjects = get_subjects(test_root_dir)

# Ensure subjects in `test_root_dir` are valid for selection
valid_test_subjects = list(set(all_train_subjects) & set(all_test_subjects))

# Path to save iteration-wise weights and results
iteration_weights_folder = 'iteration_weights'
# Ensure the folder exists
os.makedirs(iteration_weights_folder, exist_ok=True)

num_rounds = 100  # Number of rounds
test_accuracies = []
all_rounds_metrics = []
roc_curves = []  # Store ROC curves for averaging
best_test_acc = 0.0  # Initialize the best accuracy
best_model_path = 'best_model_with_metrics.pth'  # Path to save the best model

# Initialize variables to store the best subjects
best_train_subjects = None
best_test_subjects = None

# Define specific FPR thresholds for TPR evaluation
fpr_targets = [0.01, 0.001]  # FPR = 10^-2, 10^-3

# Open the result file to log all data
with open('100iterationResult_with_metrics.txt', 'w') as f:
    f.write('Test Accuracy and Metrics for Each Iteration:\n')

    for round_idx in range(num_rounds):
        print(f'Round {round_idx + 1}/{num_rounds}')

        # Shuffle subjects to ensure randomness in each iteration
        random.shuffle(valid_test_subjects)

        # Randomly select 13 subjects for testing and use the rest for training (51 subject for training) total 64 subjects
        test_subjects = random.sample(valid_test_subjects, 13)
        train_subjects = [s for s in all_train_subjects if s not in test_subjects]

        # Prepare datasets
        train_dataset = AOLPDataset(root_dir=train_root_dir, transform=transform, subjects=train_subjects)
        test_dataset = AOLPDataset(root_dir=test_root_dir, transform=transform, subjects=test_subjects)

        # Create data loaders
        batch_size = 32
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = initialize_model()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 30

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}')

        # Evaluate on the test set after training
        test_labels, test_scores = evaluate(model, test_loader)
        test_predictions = np.array(test_scores) >= 0.5  # Threshold at 0.5
        test_acc = (test_predictions == np.array(test_labels)).mean()
        test_accuracies.append(test_acc)

        # Calculate metrics
        fpr, tpr, thresholds = roc_curve(test_labels, test_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))  # Store for mean ROC curve later

        # Calculate EER
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        idxE = np.nanargmin(abs_diffs)  # Index at which fpr and fnr are closest to each other
        EER = fpr[idxE]

        # EER threshold
        eer_threshold = thresholds[idxE]

        # Calculate HTER
        FAR = fnr[idxE]  # FAR should correspond to FNR in this setup
        FRR = fpr[idxE]  # FRR should correspond to FPR in this setup
        HTER = (FAR + FRR) / 2

        # Calculate TPR @ specific FPRs
        tpr_at_fpr = {}
        for fpr_target in fpr_targets:
            if any(fpr >= fpr_target):
                idx = np.where(fpr >= fpr_target)[0][0]
                tpr_at_fpr[f"TPR@FPR={fpr_target:.0e}"] = tpr[idx]
            else:
                tpr_at_fpr[f"TPR@FPR={fpr_target:.0e}"] = 0.0  # Default if FPR target is not reached


        # Store metrics
        round_metrics = {
            'test_accuracy': test_acc,
            'roc_auc': roc_auc,
            'EER': EER,
            'EER_threshold': eer_threshold,  # Track EER threshold
            'HTER': HTER,
            'FAR': FAR,  # Track FAR
            'FRR': FRR   # Track FRR
        }
        round_metrics.update(tpr_at_fpr)
        all_rounds_metrics.append(round_metrics)

        # Log the metrics for this round
        f.write(f'Round [{round_idx+1}]: Test Accuracy = {test_acc:.4f}, ROC AUC = {roc_auc:.4f}, EER = {EER:.4f}, EER Threshold = {eer_threshold:.4f}, HTER = {HTER:.4f}, FAR = {FAR:.4f}, FRR = {FRR:.4f}, {tpr_at_fpr}\n')
        print(f'Round [{round_idx+1}], Test Accuracy: {test_acc:.4f}, ROC AUC: {roc_auc:.4f}, EER: {EER:.4f}, EER Threshold: {eer_threshold:.4f}, HTER: {HTER:.4f}, FAR: {FAR:.4f}, FRR: {FRR:.4f}, {tpr_at_fpr}')

        # Save the model and subject information if the test accuracy is the best so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save model state dict along with training and test subjects
            state = {
                'model_state_dict': model.state_dict(),
                'train_subjects': train_subjects,
                'test_subjects': test_subjects,
                'test_accuracy': best_test_acc,
                'test_labels': test_labels,
                'test_scores': test_scores
            }
            torch.save(state, best_model_path)
            best_train_subjects = train_subjects
            best_test_subjects = test_subjects
            print(f'New best model saved with accuracy: {best_test_acc:.4f}')
        
        # Save the weights and results for the current iteration
        iteration_path = os.path.join(iteration_weights_folder, f'iteration_{round_idx + 1}.pth')
        iteration_state = {
            'iteration': round_idx + 1,
            'model_state_dict': model.state_dict(),
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
            'metrics': {
                'test_accuracy': test_acc,
                'roc_auc': roc_auc,
                'EER': EER,
                'EER_threshold': eer_threshold,
                'HTER': HTER,
                'FAR': FAR,
                'FRR': FRR,
                'TPR@FPR=0.01': tpr_at_fpr.get('TPR@FPR=1e-02', None),
                'TPR@FPR=0.001': tpr_at_fpr.get('TPR@FPR=1e-03', None),
            },
            'test_labels': test_labels,
            'test_scores': test_scores,
        }
        torch.save(iteration_state, iteration_path)
        print(f'Iteration {round_idx + 1} weights and results saved to {iteration_path}')

    # After all rounds are completed, compute overall statistics
    test_accuracies = [m['test_accuracy'] for m in all_rounds_metrics]
    roc_aucs = [m['roc_auc'] for m in all_rounds_metrics]
    EERs = [m['EER'] for m in all_rounds_metrics]
    eer_thresholds = [m['EER_threshold'] for m in all_rounds_metrics]  # Collect EER thresholds
    HTERs = [m['HTER'] for m in all_rounds_metrics]
    FARs = [m['FAR'] for m in all_rounds_metrics]
    FRRs = [m['FRR'] for m in all_rounds_metrics]

    mean_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    mean_EER = np.mean(EERs)
    std_EER = np.std(EERs)
    mean_EER_threshold = np.mean(eer_thresholds)  # Calculate mean EER threshold
    std_EER_threshold = np.std(eer_thresholds)
    mean_HTER = np.mean(HTERs)
    std_HTER = np.std(HTERs)
    mean_FAR = np.mean(FARs)
    std_FAR = np.std(FARs)
    mean_FRR = np.mean(FRRs)
    std_FRR = np.std(FRRs)

    print(f'\nAggregate Metrics Across All Iterations:')
    print(f'Mean Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}')
    print(f'Mean ROC AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}')
    print(f'Mean EER: {mean_EER:.4f} ± {std_EER:.4f}')
    print(f'Mean EER Threshold: {mean_EER_threshold:.4f} ± {std_EER_threshold:.4f}')
    print(f'Mean HTER: {mean_HTER:.4f} ± {std_HTER:.4f}')
    print(f'Mean FAR: {mean_FAR:.4f} ± {std_FAR:.4f}')
    print(f'Mean FRR: {mean_FRR:.4f} ± {std_FRR:.4f}')

    # Save overall statistics and best subject information to the text file
    f.write('\nAggregate Metrics Across All Iterations:\n')
    f.write(f'Mean Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}\n')
    f.write(f'Mean ROC AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}\n')
    f.write(f'Mean EER: {mean_EER:.4f} ± {std_EER:.4f}\n')
    f.write(f'Mean EER Threshold: {mean_EER_threshold:.4f} ± {std_EER_threshold:.4f}\n')
    f.write(f'Mean HTER: {mean_HTER:.4f} ± {std_HTER:.4f}\n')
    f.write(f'Mean FAR: {mean_FAR:.4f} ± {std_FAR:.4f}\n')
    f.write(f'Mean FRR: {mean_FRR:.4f} ± {std_FRR:.4f}\n')

    # Calculate mean TPR @ FPR
    tpr_at_fpr_summary = {key: [] for key in tpr_at_fpr.keys()}
    for metrics in all_rounds_metrics:
        for key in tpr_at_fpr.keys():
            tpr_at_fpr_summary[key].append(metrics[key])

    for key, values in tpr_at_fpr_summary.items():
        mean_tpr = np.mean(values)
        std_tpr = np.std(values)
        print(f'{key}: {mean_tpr:.4f} ± {std_tpr:.4f}')
        f.write(f'{key}: {mean_tpr:.4f} ± {std_tpr:.4f}\n')

    # Plot and save mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
    mean_roc_auc = auc(mean_fpr, mean_tpr)

    f.write(f'mean_fpr for ROC: {mean_fpr}\n')
    f.write(f'mean_tpr for ROC: {mean_tpr}\n')

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve Across All Iterations')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('mean_roc_curve.png')
    plt.show()

print('Results saved to 100iterationResult_with_metrics.txt')
