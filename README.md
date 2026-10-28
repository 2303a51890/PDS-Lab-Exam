# 🍎 Apple Plant Disease Classification

**Course:** Python for Data Science

A deep learning project that uses computer vision and transfer learning to classify apple plant diseases from leaf images. The project compares two pretrained deep learning architectures, ResNet50 and EfficientNetB4, using the Plant Pathology 2021 (FGVC8) dataset.

## 👥 Team Members

1. **Lalith**
2. **Nakshatra**
3. **Rishi**

## 🎯 Project Objective

The objective of this project is to apply Python-based data science and deep learning techniques to apple plant disease classification. The project involves dataset preparation, image preprocessing, model training, performance evaluation, and comparison of two deep learning models.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Kaggle Notebook

## 🧠 Models Used

- **ResNet50:** A pretrained convolutional neural network using residual connections.
- **EfficientNetB4:** A pretrained convolutional neural network designed to balance accuracy and computational efficiency.

Both models use transfer learning with ImageNet-pretrained weights.

## 📊 Project Workflow

1. Load the Plant Pathology 2021 dataset.
2. Inspect the dataset and identify missing images.
3. Preprocess and resize images to 224 × 224 pixels.
4. Split the dataset into training, validation, and testing sets using a 70:15:15 ratio.
5. Apply data augmentation and class weighting.
6. Train the ResNet50 model.
7. Train the EfficientNetB4 model.
8. Evaluate both models using classification metrics.
9. Compare model performance and inference speed.
10. Generate visualizations and save the results.

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training time
- Inference time
- Inference throughput

The results are used to compare the performance of the two models.

## 📂 Project Outputs

The pipeline generates:

- Trained model files
- Dataset split CSV files
- Class distribution and class weight information
- Training history
- Classification reports
- Confusion matrices
- Accuracy and loss plots
- Model performance comparison CSV
- A ZIP archive containing the project outputs

## 🎓 Course Information

This project demonstrates the application of Python, data preprocessing, data analysis, and deep learning techniques to a real-world agricultural classification problem.

**Course:** Python for Data Science  
**Project Title:** Apple Plant Disease Classification  
**Team Members:** Lalith, Nakshatra, and Rishi

---
