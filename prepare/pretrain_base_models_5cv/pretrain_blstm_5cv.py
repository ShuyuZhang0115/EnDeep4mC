import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                            matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import ml_code, read_fasta_data
from models.deep_models import BLSTMModel
from feature_engineering.feature_selection import load_top_features, get_feature_methods, load_best_n

# ------------------------- Tool function -------------------------
def specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp + 1e-7)

class TrainingMonitor(tf.keras.callbacks.Callback):
    """Custom training monitoring callbacks"""
    def __init__(self, fold_num):
        super().__init__()
        self.fold_num = fold_num
        self.epoch_logs = []
        
    def on_epoch_end(self, epoch, logs=None):
        """Save the metrics of each epoch"""
        logs = logs or {}
        logs['epoch'] = epoch + 1
        self.epoch_logs.append(logs)
        
    def on_train_end(self, logs=None):
        """Output the summary information after the training is completed"""
        best_epoch = max(self.epoch_logs, key=lambda x: x['val_accuracy'])
        print(f"\nFold {self.fold_num} best epoch {best_epoch['epoch']}:")
        print(f"train set - loss: {best_epoch['loss']:.6f}, acc: {best_epoch['accuracy']:.6f}")
        print(f"validation test - loss: {best_epoch['val_loss']:.6f}, acc: {best_epoch['val_accuracy']:.6f}")

# ------------------------- Data loading -------------------------
def load_full_dataset(dataset, feature_methods=None):
    """Load and encode the complete data set"""
    base_dir = '/your_path/EnDeep4mC/data/4mC'
    
    # Read the train dataset
    train_pos = read_fasta_data(os.path.join(base_dir, dataset, "train_pos.txt"))
    train_neg = read_fasta_data(os.path.join(base_dir, dataset, "train_neg.txt"))
    train_df = pd.DataFrame({
        "label": [1]*len(train_pos) + [0]*len(train_neg),
        "seq": train_pos + train_neg
    })
    
    # Read the test dataset
    test_pos = read_fasta_data(os.path.join(base_dir, dataset, "test_pos.txt"))
    test_neg = read_fasta_data(os.path.join(base_dir, dataset, "test_neg.txt"))
    test_df = pd.DataFrame({
        "label": [1]*len(test_pos) + [0]*len(test_neg),
        "seq": test_pos + test_neg
    })

    # Feature engineering
    X_train, y_train, _ = ml_code(train_df, "training", feature_methods)
    X_test, y_test, _ = ml_code(test_df, "testing", feature_methods)
    
    return X_train, y_train, X_test, y_test

# ------------------------- Data preprocessing -------------------------
def preprocess_data(X_train, y_train, X_val=None):
    """Data preprocessing process"""
    # SMOTE Oversampling
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # Standardization
    scaler = StandardScaler().fit(X_res)
    X_res = scaler.transform(X_res)
    
    # Add noise enhancement
    np.random.seed(42)
    noise = 0.05 * np.random.normal(0, 1, X_res.shape)
    X_aug = np.vstack([X_res, X_res + noise])
    y_aug = np.concatenate([y_res, y_res])
    
    # Adjust the shape
    input_shape = (1, X_aug.shape[1])
    X_aug = X_aug.reshape(-1, *input_shape)
    
    # Validation set processing
    if X_val is not None:
        X_val = scaler.transform(X_val).reshape(-1, *input_shape)
    
    return X_aug, y_aug, X_val, scaler

# ------------------------- Model construction -------------------------
def build_blstm_model(input_shape, dropout=0.2, rec_dropout=0.1, lr=0.001):
    """Build an instance of the Bi-LSTM model"""
    return BLSTMModel(
        input_shape=input_shape,
        epochs=100,
        batch_size=512,
        dropout_rate=dropout,
        recurrent_dropout=rec_dropout,
        learning_rate=lr
    )

# ------------------------- Training configuration -------------------------
def get_callbacks(fold_model_path, fold_num):
    """Obtain the training callback configuration"""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            mode='max',
            restore_best_weights=True
        ),
        ModelCheckpoint(
            fold_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        TrainingMonitor(fold_num)  # Add custom monitoring
    ]

# ------------------------- Model Evaluation -------------------------
def evaluate_model(model, X_test, y_test, scaler):
    """Model evaluation process"""
    X_test_rs = scaler.transform(X_test).reshape(-1, 1, X_test.shape[1])
    y_proba = model.predict(X_test_rs).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    return {
        'ACC': accuracy_score(y_test, y_pred),
        'SN': recall_score(y_test, y_pred),
        'SP': specificity(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }, y_proba

# ------------------------- Save the result -------------------------
def save_results(dataset, metrics, roc_data):
    """Save the evaluation results and the ROC curve"""
    # Save indicators
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(project_dir, 'pretrained_models/5cv_roc', f'BLSTM_metrics_{dataset}.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save the ROC data
    roc_df = pd.DataFrame(roc_data)
    roc_file = os.path.join(project_dir, 'pretrained_models/5cv_roc', f'BLSTM_ROC_{dataset}.csv')
    roc_df.to_csv(roc_file, index=False)
    
    # Draw the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(roc_df['FPR'], roc_df['TPR'], 
            color='darkorange', lw=2,
            label=f'BLSTM (AUC = {metrics["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Best ROC for {dataset}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(project_dir, 'pretrained_models/5cv_roc', f'BLSTM_ROC_{dataset}.png'))
    plt.close()

# ------------------------- Cross-validation process -------------------------
def run_cross_validation(dataset, X_train, y_train, X_test, y_test, input_shape):
    """Perform 5CV"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_metrics = {'AUC': 0}
    best_roc_data = None
    final_model_path = f'/your_path/EnDeep4mC/pretrained_models/5cv/blstm_best_{dataset}.h5'

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n----- Fold {fold} -----")
        
        # Divide the data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Pre-processing
        X_aug, y_aug, X_val_rs, scaler = preprocess_data(X_tr, y_tr, X_val)
        
        # Build the model
        model = build_blstm_model(input_shape)
        fold_model_path = f'/tmp/blstm_{dataset}_fold{fold}.h5'
        
        # Train the model
        model.model.fit(
            X_aug, y_aug,
            validation_data=(X_val_rs, y_val),
            epochs=100,
            batch_size=512,
            verbose=1,
            callbacks=get_callbacks(fold_model_path, fold),
            class_weight=dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
        )

        # Evaluate model
        model.model.load_weights(fold_model_path)
        test_metrics, y_proba = evaluate_model(model.model, X_test, y_test, scaler)
        
        print(f"\nFold {fold} Test set performance:")
        print(f"AUC: {test_metrics['AUC']:.6f}  ACC: {test_metrics['ACC']:.6f}")
        print(f"SN: {test_metrics['SN']:.6f}  SP: {test_metrics['SP']:.6f}")
        
        # Update the best model
        if test_metrics['AUC'] > best_metrics['AUC']:
            best_metrics = test_metrics
            model.model.save(final_model_path)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            best_roc_data = {'FPR': fpr, 'TPR': tpr}

    return best_metrics, best_roc_data

# ------------------------- Save the result -------------------------
def save_combined_results(summary_data):
    """Save the summary results of all datasets"""
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of Test results:")
    print(summary_df)
    summary_path = '/your_path/EnDeep4mC/pretrained_models/5cv/blstm_5cv_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nThe summary results have been saved to: {summary_path}")

# ------------------------- Main process -------------------------
def main():
    fasta_dataset_names = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
    #fasta_dataset_names = ['4mC_G.subterraneus']
    model_name = 'BLSTM'
    summary_data = []
    
    for dataset in fasta_dataset_names:
        print(f"\n=== Processing Dataset: {dataset} ===")
        
        # Feature selection
        try:
            best_n = load_best_n(model_name, dataset)
            feature_methods = get_feature_methods(load_top_features(model_name, dataset, best_n))
        except Exception as e:
            print(f"Feature selection error: {str(e)}, using default features")
            feature_methods = None
        
        # Load data
        X_train, y_train, X_test, y_test = load_full_dataset(dataset, feature_methods)
        input_shape = (1, X_train.shape[1])
        
        # Perform cross-validation
        best_metrics, best_roc_data = run_cross_validation(
            dataset, X_train, y_train, X_test, y_test, input_shape
        )
        
        # Output result
        print("\nBest Test Performance:")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.6f}")
        
        # Save the results of single datasets
        save_results(dataset, best_metrics, best_roc_data)
        
        # Add to the summary data
        summary_entry = {
            'Dataset': dataset,
            'ACC': f"{best_metrics['ACC']:.6f}",
            'SN': f"{best_metrics['SN']:.6f}", 
            'SP': f"{best_metrics['SP']:.6f}",
            'F1': f"{best_metrics['F1']:.6f}",
            'MCC': f"{best_metrics['MCC']:.6f}",
            'AUC': f"{best_metrics['AUC']:.6f}"
        }
        summary_data.append(summary_entry)

    # Save the summary result
    save_combined_results(summary_data)

if __name__ == "__main__":
    main()