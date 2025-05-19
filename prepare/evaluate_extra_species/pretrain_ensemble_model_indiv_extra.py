import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             f1_score, matthews_corrcoef, roc_curve)
from lightgbm import LGBMClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from prepare.prepare_ml import ml_code, read_fasta_data
from feature_engineering.feature_selection_extra import load_top_features, get_feature_methods

class EnhancedEnsembleIndiv:
    def __init__(self):
        self.base_models = ['CNN', 'BLSTM', 'Transformer']
        self.scalers = {}
        self.meta_model = None
        self.dataset_name = None
        self.feature_configs = {}
        self.best_meta_model = None
        self.full_data_cache = {}  # Cache

    def _get_best_n_features(self, model_name):
        acc_table_path = os.path.join(project_dir, "extra_feature_ranking", f"{model_name}_Feature_Acc_Table.csv")
        acc_table = pd.read_csv(acc_table_path)
        return int(acc_table[acc_table['N'] == 'best_n'][self.dataset_name].values[0])

    def _load_full_dataset(self, model_name):
        """Merge all data for unified feature engineering"""
        if (model_name, self.dataset_name) in self.full_data_cache:
            return self.full_data_cache[(model_name, self.dataset_name)]

        base_dir = '/your_path/EnDeep4mC/data/4mC'
        paths = {
            'train_pos': os.path.join(base_dir, self.dataset_name, "train_pos.txt"),
            'train_neg': os.path.join(base_dir, self.dataset_name, "train_neg.txt"),
            'test_pos': os.path.join(base_dir, self.dataset_name, "test_pos.txt"),
            'test_neg': os.path.join(base_dir, self.dataset_name, "test_neg.txt")
        }

        # Merge all the original data
        full_df = pd.DataFrame({
            "label": ([1]*len(read_fasta_data(paths['train_pos'])) + 
                     [0]*len(read_fasta_data(paths['train_neg'])) +
                     [1]*len(read_fasta_data(paths['test_pos'])) + 
                     [0]*len(read_fasta_data(paths['test_neg']))),
            "seq": (read_fasta_data(paths['train_pos']) + 
                   read_fasta_data(paths['train_neg']) +
                   read_fasta_data(paths['test_pos']) + 
                   read_fasta_data(paths['test_neg']))
        })

        # Unified Feature Engineering
        best_n = self._get_best_n_features(model_name)
        self.feature_configs[model_name] = get_feature_methods(
            load_top_features(model_name, self.dataset_name, best_n)
        )
        X_full, y_full, _ = ml_code(full_df, "training", self.feature_configs[model_name])

        # Cache processing
        self.full_data_cache[(model_name, self.dataset_name)] = (X_full, y_full)
        return X_full, y_full

    def _prepare_data(self, model_name):
        """Optimized data preprocessing process"""
        # Load all the data
        X_full, y_full = self._load_full_dataset(model_name)
        
        # Calculate the sample size of the original training set
        base_dir = '/your_path/EnDeep4mC/data/4mC'
        train_size = (
            len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_pos.txt"))) +
            len(read_fasta_data(os.path.join(base_dir, self.dataset_name, "train_neg.txt")))
        )
        
        # Secure data separation
        X_train = X_full[:train_size]
        y_train = y_full[:train_size]
        X_test = X_full[train_size:]
        y_test = y_full[train_size:]

        # Data augmentation applied to the training set
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        
        # Standardized processing
        self.scalers[model_name] = StandardScaler()
        X_res_scaled = self.scalers[model_name].fit_transform(X_res)
        
        # Noise enhancement
        np.random.seed(42)
        noise = 0.1 * np.random.normal(0, 1, X_res_scaled.shape)
        X_train_aug = np.vstack([X_res_scaled, X_res_scaled + noise])
        y_train_aug = np.concatenate([y_res, y_res])
        
        # Process the test data
        X_test_scaled = self.scalers[model_name].transform(X_test)
        
        return X_train_aug, y_train_aug, X_test_scaled, y_test

    def _load_pretrained_model(self, model_name):
        model_path = f'/your_path/EnDeep4mC/train_data_auto_indiv_extra/{model_name.lower()}_{self.dataset_name}.h5'
        return tf.keras.models.load_model(model_path)

    def _generate_meta_features(self, model_name, X):
        X_reshaped = X.reshape(-1, 1, X.shape[1])
        return self._load_pretrained_model(model_name).predict(X_reshaped, verbose=0).flatten()

    def _prepare_meta_dataset(self):
        meta_train, meta_test = [], []
        y_train_full, y_test_full = None, None

        for model_name in tqdm(self.base_models, desc="Processing base models"):
            X_train, y_train, X_test, y_test = self._prepare_data(model_name)
            
            # Generate meta feature
            train_feat = self._generate_meta_features(model_name, X_train)
            test_feat = self._generate_meta_features(model_name, X_test)
            
            meta_train.append(train_feat)
            meta_test.append(test_feat)
            
            if y_train_full is None:
                y_train_full = y_train
                y_test_full = y_test
            else:
                assert np.array_equal(y_train_full, y_train), "Train labels inconsistent"
                assert np.array_equal(y_test_full, y_test), "Test labels inconsistent"

        return np.column_stack(meta_train), y_train_full, np.column_stack(meta_test), y_test_full

    def train_and_evaluate(self):
        # Prepare the meta dataset
        meta_X_train, y_train, meta_X_test, y_test = self._prepare_meta_dataset()

        base_models = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_samples=20
            ))
        ]

        final_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=0.6,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        )

        # Train meta model
        self.meta_model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_model,
            stack_method='predict_proba',
            passthrough=True,
            n_jobs=-1
        )
        self.meta_model.fit(meta_X_train, y_train)

        # Save model
        model_path = f'/your_path/EnDeep4mC/train_data_auto_indiv_extra/ensemble_indiv_{self.dataset_name}.pkl'
        joblib.dump(self.meta_model, model_path)

        # Evaluate performance
        y_proba = self.meta_model.predict_proba(meta_X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate index
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'sn': recall_score(y_test, y_pred),
            'sp': recall_score(y_test, y_pred, pos_label=0),
            'f1': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        # Save the ROC data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'AUC': [metrics['auc']]*len(fpr)})
        roc_df.to_csv(
            os.path.join(project_dir, 'train_data_auto_indiv_extra', f'Ensemble_roc_{self.dataset_name}.csv'),
            index=False,
            float_format='%.6f'
        )
        
        # Draw the ROC curve
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble (AUC = {metrics["auc"]:.6f})')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({self.dataset_name})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(project_dir, 'train_data_auto_indiv_extra', f'Ensemble_roc_{self.dataset_name}.png'))
        plt.close()

        return metrics

if __name__ == "__main__":
    datasets = [
        '4mC_A.thaliana2', '4mC_C.elegans2', '4mC_C.equisetifolia', '4mC_D.melanogaster2', '4mC_E.coli2',
        '4mC_F.vesca', '4mC_G.pickeringii2', '4mC_G.subterraneus2', '4mC_S.cerevisiae', '4mC_Tolypocladium'
    ]
    #datasets = ['4mC_E.coli2']
    results = []
    
    for dataset in datasets:
        print(f"\n=== Processing {dataset} ===")
        try:
            ensemble = EnhancedEnsembleIndiv()
            ensemble.dataset_name = dataset
            
            metrics = ensemble.train_and_evaluate()
            
            formatted_metrics = {k: f"{v:.6f}" for k, v in metrics.items()}
            results.append({
                'Dataset': dataset,
                **formatted_metrics
            })
            
            print(f"\n{dataset} Test result:")
            for k, v in formatted_metrics.items():
                print(f"{k.upper()}: {v}")
                
        except Exception as e:
            print(f"Failed to process {dataset}: {str(e)}")
            continue
    
    result_df = pd.DataFrame(results)
    save_path = '/your_path/EnDeep4mC/train_data_auto_indiv_extra/ensemble_indiv_results.csv'
    result_df.to_csv(save_path, index=False, float_format='%.6f')
    print("\nThe final result has been saved to:", save_path)