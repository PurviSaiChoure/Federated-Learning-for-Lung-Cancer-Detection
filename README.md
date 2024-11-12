# Federated Learning for Cancer Detection with Privacy Protection

## Abstract
This project implements a privacy-preserving approach to cancer detection using Federated Learning (FL) combined with Differential Privacy (DP). It enables collaborative learning across distributed medical institutions while protecting patient privacy. The system evaluates different transfer learning architectures (VGG16, ResNet50, MobileNetV2, InceptionV3, and DenseNet121) within a federated learning framework.

## Key Features
- Federated Learning implementation for distributed model training
- Differential Privacy integration for enhanced data protection
- Multiple pre-trained architecture support:
  - InceptionV3
  - ResNet50
  - MobileNetV2
  - VGG16
  - DenseNet121
- Data augmentation and preprocessing pipeline
- Privacy-preserving model updates using DP-enabled optimizers

## Requirements
- TensorFlow 2.12.0
- Keras 2.12.0
- TensorFlow Privacy 0.8.10
- Flower Framework 1.6.0

## Dataset
The project uses a Chest CT scan dataset containing:
- Adenocarcinoma lung images
- Large-cell undifferentiated carcinoma lung images
- Squamous cell lung images
- Normal cell images

Total images: 1000
- Training: 613 images
- Testing: 315 images
- Validation: 72 images

Dataset available at: [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

## Architecture

### Federated Learning Setup
1. **Client Side**:
   - Local model training on private data
   - Data preprocessing and augmentation
   - Differential privacy implementation
   - Model weight updates

2. **Server Side**:
   - Global model coordination
   - FedAvg algorithm for weight aggregation
   - Distribution of updated global model


![Federated Learning Setup](images/fl_setup.png)
*Figure 1: Architecture diagram showing the federated learning setup with multiple clients (hospitals) and central server*

### Model Performance Comparison
![Model Performance](https://github.com/user-attachments/assets/79e872a4-3743-4ca5-b668-bce47120ce50)

*Figure 2: Performance comparison of different architectures on Chest CT scan dataset*

### Transfer Learning Integration
![Transfer Learning Integration](images/transfer_learning.png)
*Figure 3: Integration of Transfer Learning with Federated Learning framework*


### Privacy Protection
- Differential Privacy integration using DP Keras Adam Optimizer
- Noise multiplier settings:
  - Local model: 0.005
  - Global model: 0.001
- Epsilon: 2.0
- Norm Clip: 1.0

## Results
Performance metrics for different architectures in FL setup:

| Model Architecture | Training Accuracy | Testing Accuracy |
|-------------------|-------------------|------------------|
| DenseNet121       | 0.6378           | 0.6984          |
| InceptionV3       | 0.6721           | 0.7746          |
| MobileNetV2       | 0.7031           | 0.7048          |
| VGG16             | 0.6476           | 0.7238          |
| ResNet50          | 0.5073           | 0.5429          |

## Usage
1. Install required dependencies:
```bash
pip install tensorflow==2.12.0 keras==2.12.0 tensorflow-privacy==0.8.10 flwr==1.6.0
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/Federated-Learning-for-Lung-Cancer-Detection.git

cd Federated-Learning-for-Lung-Cancer-Detection
```

3. **Download and prepare the dataset:**
   - Download the Chest CT scan dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images).
   - Extract and organize the data in the following structure:

   ```plaintext
   data/
   ├── train/
   │   ├── adenocarcinoma/
   │   ├── large.cell.carcinoma/
   │   ├── normal/
   │   └── squamous.cell.carcinoma/
   ├── test/
   └── val/

4. Start the FL server:

```bash
python server.py
```


5. Start FL clients (in separate terminals):

```bash 
python client.py --client_id 1
python client.py --client_id 2
```


## Implementation Details

### Data Preprocessing
- Image rescaling to 0-1 range
- Data augmentation techniques:
  - Rotation (range: 8 degrees)
  - Horizontal flip
  - Width and height shift
  - Shear (range: 0.2)
  - Zoom (range: 0.2)

### Model Training
- Optimizer: DP Keras Adam Optimizer
- Loss function: Categorical cross-entropy
- Training epochs: 30 (local) with early stopping
- FL rounds: 15
- Base models: Pre-trained on ImageNet

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Flower framework team for the federated learning implementation
- Kaggle community for providing the Chest CT scan dataset
- TensorFlow team for TensorFlow Privacy implementation

## Contact
For any queries or suggestions, please open an issue in the repository or contact:
- Email: [purvichoure2@gmail.com](mailto:purvichoure2@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/purvi29/](https://www.linkedin.com/in/purvi29/)

## Citation
If you use this code in your research, please cite:

```bibtex
@inproceedings{
  author    = {Choure, P. and Prajapat, S. and Berwal, K.},
  title     = {Federated Learning Approach using Transfer Learning Architectures for Lung Cancer Detection},
  booktitle = {The International Conference on Computing, Communication, Cybersecurity \& AI},
  year      = {2024}
}
```
