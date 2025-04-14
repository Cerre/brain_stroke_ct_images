# Brain Stroke CT Scan Classification
This project uses a deep learning model to classify brain CT scan images into different categories, Normal, Bleeding, and Ischemia. The classification is performed using PyTorch and leverages transfer learning with a pre-trained convolutional neural network (CNN).

## Dataset

The dataset is available on Kaggle at [ozguraslank/brain-stroke-ct-dataset](https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset).

The project expects a dataset structured similarly to the following:

```
Brain_Stroke_CT_Dataset/
├── Bleeding/
│   ├── DICOM/
│   ├── OVERLAY/
│   └── PNG/
│       └── *.png
├── Ischemia/
│   ├── DICOM/
│   ├── OVERLAY/
│   └── PNG/
│       └── *.png
├── Normal/
│   ├── DICOM/
│   └── PNG/
│       └── *.png
└── External_Test/ (Optional)
    ├── DICOM/
    ├── MASKS/
    ├── labels.csv
    └── ...
```

The primary image format used for training seems to be PNG files located within the respective class subdirectories (Bleeding, Ischemia, Normal) under a `PNG` folder.

## Dependencies

The core dependencies for this project are:

*   Python 3.x
*   PyTorch
*   Torchvision
*   Pandas
*   NumPy
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   Pillow (PIL)
*   Jupyter Notebook or JupyterLab (to run the `.ipynb` file)

You can typically install these using pip:

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow jupyterlab
```

*(Note: Ensure you install the correct PyTorch version for your system/CUDA setup. Refer to the [official PyTorch website](https://pytorch.org/) for specific instructions.)*


## Model

The `main.ipynb` notebook likely utilizes a pre-trained ResNet18 model from `torchvision.models` and fine-tunes it for the specific task of CT scan classification.

## Evaluation

The notebook includes steps to evaluate the trained model's performance using metrics such as:

*   Accuracy
*   Confusion Matrix
*   Classification Report

The model achieved approximately 93% accuracy on the test set for the binary classification task of distinguishing between 'Normal' and 'Not Normal' (Bleeding or Ischemia).