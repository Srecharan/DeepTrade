#new one

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from datetime import datetime
import json
import os
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import pickle

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10):
        """Initialize dataset with sequences"""
        self.sequences = []
        self.targets = []
        
        for i in range(len(X) - sequence_length + 1):
            seq = X[i:(i + sequence_length)]
            target = y[i + sequence_length - 1]
            self.sequences.append(seq)
            self.targets.append(target)
            
        self.sequences = torch.FloatTensor(self.sequences).cuda()  # Move to GPU
        self.targets = torch.FloatTensor(self.targets).cuda()     # Move to GPU
        
        print(f"Dataset shapes - sequences: {self.sequences.shape}, targets: {self.targets.shape}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# In EnhancedLSTM class, modify the architecture
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):  # Changed from dropout=0.4
        super(EnhancedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  
        )

        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # Add sequence length attention
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply batch normalization
        batch_size, seq_len, hidden_dim = lstm_out.shape
        lstm_out = lstm_out.reshape(-1, hidden_dim)
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, seq_len, hidden_dim)
        
        attention_weights = F.softmax(
            self.attention(lstm_out).transpose(1, 2), 
            dim=2
        )
        context_vector = torch.bmm(attention_weights, lstm_out)
        

        context_vector = context_vector.squeeze(1) + lstm_out[:, -1, :]
        
        return self.fc(context_vector)

class ModelTrainer:
    def __init__(self, config_path: str = "config/model_config.json"):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.metrics = {}

    def predict_lstm(self, model: EnhancedLSTM, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM model"""
        model.eval()
        with torch.no_grad():
            sequences = []
            for i in range(len(X) - self.config['lstm']['sequence_length'] + 1):
                seq = X[i:(i + self.config['lstm']['sequence_length'])]
                sequences.append(seq)
            
            if sequences:
                sequences = torch.FloatTensor(sequences).cuda()  # Move to GPU
                predictions = model(sequences).cpu().numpy().flatten()  # Move back to CPU for numpy conversion
                return predictions
            return np.array([])   
        
    def train_lstm(self, X: np.ndarray, y: np.ndarray, symbol: str):
        """Enhanced LSTM training with better regularization and validation"""
        history = {'train_loss': [], 'val_loss': []}
        config = self.config['lstm']
        
        config = {
            'hidden_size': 256,       
            'num_layers': 3,          
            'learning_rate': 0.0003,  
            'batch_size': 32,        
            'epochs': 200,            
            'sequence_length': 30     
        }

        # 1. Improved data splitting
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 2. Create datasets with standard sequence length
        sequence_length = config['sequence_length']
        dataset_train = TimeSeriesDataset(X_train, y_train, sequence_length)
        dataset_val = TimeSeriesDataset(X_val, y_val, sequence_length)
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_dataloader = DataLoader(
            dataset_val,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 3. Create model with standard architecture
        input_size = X.shape[1]
        model = EnhancedLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=0.3  
        ).cuda()
        
        # 4. Standard loss function for all stocks
        criterion = nn.HuberLoss().cuda()
        
        # 5. Standard optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 6. Early stopping with standard parameters
        best_val_loss = float('inf')
        patience = 30 
        min_epochs = 40  
        patience_counter = 0
        best_model_state = None
                
        print("\nStarting training...")
        for epoch in range(config['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            accumulation_steps = 4 
            optimizer.zero_grad()
            
            for batch_idx, (sequences, targets) in enumerate(train_dataloader):
                sequences = sequences.cuda()
                targets = targets.cuda()
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_dataloader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets in val_dataloader:
                    outputs = model(sequences)
                    val_loss += criterion(outputs.squeeze(), targets).item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{config["epochs"]}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if epoch >= min_epochs:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping triggered at epoch {epoch+1}')
                        break

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, f'models/trained/{symbol}_lstm_best.pth')
        
        self.models[f'{symbol}_lstm'] = model
        return model, history
        
    def load_config(self, config_path: str) -> dict:
        return {
            'lstm': {
                'hidden_size': 256,  # Increased
                'num_layers': 3,     # Increased
                'learning_rate': 0.0003,
                'batch_size': 32,    # Decreased for better gradient updates
                'epochs': 200,       # Increased
                'sequence_length': 40  # Increased
            },
            'xgboost': {
                'max_depth': 8,           # Increased
                'learning_rate': 0.01,
                'n_estimators': 300,      # Increased
                'min_child_weight': 3,    # Added
                'subsample': 0.8          # Added
            }
        }

    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data with standardized preprocessing for all stocks"""
        # Calculate log returns
        log_returns = np.log(data[target_col] / data[target_col].shift(1)) * 100
        y = log_returns.values[1:]
        
        # Standard clipping for all stocks
        y = np.clip(y, -5, 5)
        y_scaler = RobustScaler(quantile_range=(10, 90))
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        self.scalers['y'] = y_scaler
        
        # Get feature columns
        feature_cols = [col for col in data.columns if col not in 
                    ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']]
        X = data[feature_cols].values[1:]
        
        # Standard clipping for features
        X = np.clip(X, -10, 10)
        X_scaler = RobustScaler(quantile_range=(10, 90))
        X_scaled = X_scaler.fit_transform(X)
        self.scalers['X'] = X_scaler
        
        return X_scaled, y

        
    def train_xgboost(self, X: np.ndarray, y: np.ndarray, symbol: str):
        """Train XGBoost model"""
        config = self.config['xgboost']
        
        model = xgb.XGBRegressor(
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            n_estimators=config['n_estimators']
        )
        
        model.fit(X, y)
        self.models[f'{symbol}_xgb'] = model
        return model
    
    def train_models(self, data: pd.DataFrame, symbol: str):
        """Train all models with proper cross-validation"""
        # Call prepare_data without symbol argument
        X, y = self.prepare_data(data)
        
        print(f"\nTraining models for {symbol}")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        try:
            # Set up cross-validation with fixed window
            window_size = 30  
            cv_scores_lstm = []
            cv_scores_xgb = []
            

            total_samples = len(X)
            n_splits = 5
            val_size = window_size * 2  
            
            for fold in range(n_splits):
                split_idx = total_samples - (n_splits - fold) * val_size
                X_train = X[:split_idx]
                y_train = y[:split_idx]
                X_val = X[split_idx:split_idx + val_size]
                y_val = y[split_idx:split_idx + val_size]
                
                print(f"\nTraining fold {fold + 1}/{n_splits}")
                print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
                
                # Train LSTM
                print("Training LSTM...")
                lstm_model, fold_history = self.train_lstm(X_train, y_train, symbol)
                
                # Train XGBoost
                print("Training XGBoost...")
                xgb_model = self.train_xgboost(X_train, y_train, symbol)
                
                # Make predictions on validation set
                lstm_preds = []
                xgb_preds = []
                
          
                lstm_model.eval()  # Set to evaluation mode
                with torch.no_grad():
                    for i in range(len(X_val) - window_size + 1):
                        X_window = X_val[i:i + window_size]
                        # LSTM prediction - ensure data is on GPU
                        seq = torch.FloatTensor(X_window).unsqueeze(0).cuda()
                        lstm_pred = lstm_model(seq).cpu().item()
                        lstm_preds.append(lstm_pred)
                        
                        # XGBoost prediction 
                        xgb_pred = xgb_model.predict(X_window[-1:])
                        xgb_preds.append(xgb_pred[0])
                
           
                y_true = y_val[window_size-1:]
                lstm_score = np.mean(np.abs(np.array(lstm_preds) - y_true))
                xgb_score = np.mean(np.abs(np.array(xgb_preds) - y_true))
                
                cv_scores_lstm.append(lstm_score)
                cv_scores_xgb.append(xgb_score)
                
                print(f"Fold {fold + 1} Scores - LSTM: {lstm_score:.4f}, XGB: {xgb_score:.4f}")
            
            print("\nCross-validation Results:")
            print(f"LSTM - Mean: {np.mean(cv_scores_lstm):.4f}, Std: {np.std(cv_scores_lstm):.4f}")
            print(f"XGB  - Mean: {np.mean(cv_scores_xgb):.4f}, Std: {np.std(cv_scores_xgb):.4f}")
            
            # Train final models on full dataset
            final_lstm, final_history = self.train_lstm(X, y, symbol)
            final_xgb = self.train_xgboost(X, y, symbol)
            
            self.models[f'{symbol}_lstm'] = final_lstm
            self.models[f'{symbol}_xgb'] = final_xgb
            
            return {
                'lstm': final_lstm,
                'xgboost': final_xgb
            }, final_history

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise e


    def save_models(self, symbol: str):
        """Save both LSTM and XGBoost models"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save LSTM model
            if f'{symbol}_lstm' in self.models:
                lstm_path = f'models/trained/{symbol}_lstm_best.pth'
                torch.save(self.models[f'{symbol}_lstm'].state_dict(), lstm_path)
                print(f"Saved LSTM model to {lstm_path}")
                
            # Save XGBoost model
            if f'{symbol}_xgb' in self.models:
                xgb_path = f'models/trained/{symbol}_xgb.json'
                self.models[f'{symbol}_xgb'].save_model(xgb_path)
                print(f"Saved XGBoost model to {xgb_path}")
                
            # Save scalers
            if self.scalers:
                scaler_path = f'models/trained/{symbol}_scalers.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers, f)
                print(f"Saved scalers to {scaler_path}")
                
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise e
    
    def save_model_version(self, symbol: str, model_type: str, metrics: Dict):
        """Save model version with metadata"""
        version_info = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': self.config[model_type],
            'version': self._get_next_version(symbol, model_type)
        }
        
        version_path = f'models/trained/versions/{symbol}/{model_type}/'
        os.makedirs(version_path, exist_ok=True)
        
        with open(f'{version_path}/v{version_info["version"]}_info.json', 'w') as f:
            json.dump(version_info, f, indent=4)
        
        # Save model
        if model_type == 'lstm':
            torch.save(
                self.models[f'{symbol}_lstm'].state_dict(),
                f'{version_path}/v{version_info["version"]}_model.pth'
            )
        else:
            self.models[f'{symbol}_xgb'].save_model(
                f'{version_path}/v{version_info["version"]}_model.json'
            )

    def visualize_feature_importance(self, model, feature_names):
        """Add after training XGBoost"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualization/feature_importance.png')
        plt.close()        