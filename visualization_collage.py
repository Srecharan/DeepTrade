import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_collage(image_type='prediction'):
    """Create a 3x2 collage of either prediction or training images"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Stock {image_type.title()} Results', fontsize=16)
    
    # Core stocks we want to show
    stocks = ['AAPL', 'AMD', 'GME', 'JNJ', 'MSFT', 'NVDA']
    
    for idx, stock in enumerate(stocks):
        row = idx // 3
        col = idx % 3
        
        # Load image
        img_path = f'visualization/{stock}_{image_type}.png'
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'{stock} {image_type.title()}', pad=20)
        else:
            axes[row, col].text(0.5, 0.5, f'No {image_type} image\nfor {stock}', 
                              ha='center', va='center')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'visualization/stock_{image_type}_collage.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create prediction collage
    create_collage('prediction')
    print("Created prediction collage")
    
    # Create training history collage
    create_collage('training_history')
    print("Created training history collage")

if __name__ == "__main__":
    main()