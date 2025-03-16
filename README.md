# Defect Classification and Anomaly Detection

This project focuses on classifying surface defects in metal images and detecting anomalies using machine learning. The dataset used is the [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html).

---

## Models Implemented
- **Convolutional Neural Network (CNN):** A custom CNN architecture for defect classification.
- **Autoencoder:** For anomaly detection based on reconstruction error.

---

## Features

### Data Exploration
- Loaded and analyzed the NEU Surface Defect Database.
- Visualized sample images from each defect class.

### Data Preprocessing
- Resized images to 64x64 pixels.
- Normalized pixel values to the range [0, 1].
- Split data into training and testing sets (80:20 ratio).
- Applied data augmentation (rotation, shifting, flipping) to improve generalization.

### Model Training
- Trained a CNN model for defect classification.
- Used Adam optimizer and categorical cross-entropy loss.
- Trained for 20 epochs with a batch size of 32.

### Model Evaluation
- Evaluated the model using accuracy, precision, recall, and F1-score.
- Generated a confusion matrix to analyze classification errors.

### Visualization
- Plotted training and validation accuracy/loss curves.
- Visualized confusion matrix and classification report.

---

## Results

### Best Model
- **CNN for Defect Classification:**
  - Accuracy: 94%
  - Precision: 0.94 (macro avg)
  - Recall: 0.94 (macro avg)
  - F1-score: 0.94 (macro avg)

### Feature Importance
- The CNN model effectively learned features for distinguishing between defect classes.
- The autoencoder identified anomalies based on high reconstruction error.

---

## Outcome

### Best Performing Model
- The CNN model achieved the highest accuracy (94%) and demonstrated robust performance across all defect classes.

### Future Work
- Experiment with transfer learning using pre-trained models (e.g., ResNet, EfficientNet).
- Improve anomaly detection using advanced techniques like GANs or Isolation Forest.
- Increase dataset size for underrepresented classes to improve recall.

---

## Notes
- The dataset contains 6 defect classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, and scratches.
- Data augmentation significantly improved model generalization.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a concise overview of the project. For detailed implementation, refer to the code and comments in the repository.
