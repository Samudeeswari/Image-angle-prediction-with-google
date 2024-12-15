# Image-Based Angle Prediction Model

### Disclaimer

Due to company policies and restrictions on prosthetic usage for data acquisition, we were only able to gather a limited amount of low-quality image data using a Raspberry Pi camera (referred to as "raspberry 5"). As a result, the model's performance is constrained and may not reflect its full potential under ideal conditions. Future improvements will require higher-quality images and a more extensive dataset to achieve optimal results.

## Overview

This project focuses on developing a convolutional neural network (CNN) to predict prosthetic joint angles (x and y values) directly from image data. The goal is to integrate image-based visual feedback with other sensory modalities, such as electromyography (EMG) signals, to enhance control of prosthetic limb movements. When combined with EMG signal classifications indicating the user's intent, the predicted angles can guide prosthetic systems to adjust motor movements dynamically based on terrain and user actions.

## Workflow

### 1. **Data Acquisition**

The current script simulates data acquisition by fetching:

- Images from a Google Cloud Storage bucket.
- Corresponding angle readings from a Google BigQuery table.

In real-world scenarios, these data points would be collected directly from a Raspberry Pi camera and sensors mounted on the prosthetic device. However, due to data acquisition constraints, the dataset is limited in size and quality, which impacts the model's performance.

### 2. **Data Preprocessing and Augmentation**

- **Image Resizing and Normalization**:

  - Images are resized to a fixed dimension of 128x128 pixels and normalized to the [0, 1] range.
  - This ensures consistency in the data and compatibility with the CNN input requirements.

- **Data Augmentation**:
  - Techniques such as random flips, brightness adjustments, and contrast variations are applied to artificially increase dataset diversity and simulate real-world conditions. While this helps address the small dataset issue to some extent, the results remain limited due to the initial low data quality.

### 3. **Model Architecture**

- **Base Model**:

  - A pre-trained MobileNetV2 is employed as the feature extraction backbone to leverage its ability to capture visual features efficiently from small datasets.

- **Custom Layers**:

  - Dense layers with L2 regularization are added to prevent overfitting.
  - Dropout layers are incorporated to improve generalization.

- **Output**:
  - The model predicts two numeric values corresponding to the prosthetic joint angles (x, y).

This architecture maps visual terrain cues in images to appropriate prosthetic joint angles, enabling adaptive motor adjustments.

### 4. **Training and Validation**

- **Data Splitting**:

  - The dataset is divided into training, validation, and test sets to ensure robust evaluation.

- **Loss and Metrics**:

  - The model is trained to minimize Mean Squared Error (MSE), with Mean Absolute Error (MAE) tracked as a secondary evaluation metric.

- **Callbacks**:
  - Early stopping and learning rate reduction are implemented to optimize training and prevent overfitting.

### 5. **Integration with EMG Signals**

Although the integration is not explicitly implemented in the current code, the ultimate goal is to combine this image-based angle prediction model with classified EMG signals:

- **EMG Signal Classification**:

  - Determines the user's intent (e.g., walking, climbing stairs).

- **Dynamic Motor Adjustments**:
  - The angle predictor fine-tunes motor movements based on visual and EMG feedback, enhancing prosthetic functionality across different terrains and activities.

### 6. **Model Saving and Deployment**

- The trained model is saved as `gcp_angle_predictor.keras` for future inference or fine-tuning.
- This allows seamless integration into downstream systems or additional testing workflows.

### 7. **Limitations**

- **Data Quality**:
  - The low resolution and small dataset size significantly impact the model's ability to generalize.
- **Prototype Nature**:
  - This project serves as a proof of concept rather than a production-ready solution. Future efforts will be required to address these shortcomings.

## Results and Evaluation

The model was evaluated using standard regression metrics:

- **Mean Squared Error (MSE)**: Quantifies the average squared difference between predicted and actual angles.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of prediction errors.
- **Visualization**: Scatter plots of true vs. predicted angles provide a qualitative assessment of model performance.

## Conclusion and Future Work

This project demonstrates a prototype workflow for predicting prosthetic joint angles from image data. Despite its current limitations, it lays the foundation for future enhancements:

1. **Improved Data Acquisition**:

   - Collect higher-quality images and angle measurements under varied conditions.
   - Expand the dataset size to improve model generalization.

2. **Advanced Model Architectures**:

   - Experiment with state-of-the-art CNNs or hybrid models (e.g., combining CNNs with RNNs for temporal dynamics).

3. **Enhanced Integration**:

   - Implement real-time integration with EMG signal classification systems.
   - Test the system on actual prosthetic hardware in real-world conditions.

4. **Fine-Tuning and Optimization**:
   - Explore hyperparameter tuning and advanced data augmentation techniques.
   - Incorporate transfer learning to leverage pre-trained models trained on similar tasks.

By addressing these aspects, this project has the potential to evolve into a robust solution for adaptive prosthetic control, enhancing user mobility and comfort across diverse environments.

## Acknowledgments

Special thanks to the organizers of the Google AI for Impact Hackathon for providing the opportunity to work on this challenge.
