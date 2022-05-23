# EnsembleTorch
Implementation of an Ensemble model as detailed in the book Deep Learning with Pytorch by Vishnu Subramanian.
The Ensemble model is made up of pretained Resnet34, Inception and DenseNet models.

- data_laoder.py loads and preprocesses the data
- utils.py contains neccessary utilities to train and deploy the model
- models.py contains the imported models
- Ensemblemodel.py is where we put together all 3 imported models to generate an Ensemble model

