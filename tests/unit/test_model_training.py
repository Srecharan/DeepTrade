from utils.stock_manager import StockManager
from utils.model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import os

def plot_training_history(history, symbol):
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{symbol} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs('visualization', exist_ok=True)
    plt.savefig(f'visualization/{symbol}_training_history.png')
    plt.close()

def test_model_training():
    # Initialize managers
    stock_manager = StockManager()
    model_trainer = ModelTrainer()
    os.makedirs('visualization', exist_ok=True)
    
    # Test with a few symbols
    symbols = ['META', 'GOOGL', 'AMZN']
    
    for symbol in symbols:
        print(f"\n===================================== Training models for {symbol} =====================================")
        try:
            # Get data with both technical and sentiment features
            data, _ = stock_manager.prepare_prediction_data(symbol)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Train models and get history
            models, history = model_trainer.train_models(data, symbol)
            
            # Plot training history
            plot_training_history(history, symbol)
            
            # Plot feature importance for XGBoost
            if f'{symbol}_xgb' in model_trainer.models:
                feature_names = [col for col in data.columns if col not in 
                               ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']]
                model_trainer.visualize_feature_importance(
                    model_trainer.models[f'{symbol}_xgb'],
                    feature_names
                )
            
            # Save models
            model_trainer.save_models(symbol)
            print(f"Successfully trained and saved models for {symbol}")
            
        except Exception as e:
            print(f"Error training models for {symbol}: {str(e)}")

if __name__ == "__main__":
    test_model_training()