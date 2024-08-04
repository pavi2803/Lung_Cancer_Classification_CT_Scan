# Lung_Cancer_Classification_CT_Scan
This project classifies Lung cancer Disease by training on the CT scan images of Lungs


### Workflows

1. The Template.py creates the layeout and files, directories required for the project
2. Create config.yaml - consists of things like - connection to API, paths etc
3. Create/update params.yaml - Contains the **hyperparameters** settings
4. Create/update the entity - This usually refers to the entity models or data classes that represent the core data structures of the application, such as user, product, order, etc.
5. Create/update the configuration manager in src config - Configuration file contains the necessary **configuration** and data utilities and functions required for the components of the system
6. Creating the Components - has the components of the project/system - **Data ingestion, Preparing the Base model, Model training and Model Evaluation**
7. Create/update the pipeline - Contains the Workflow Stages - 1,2,3 and 4 which contains the **modular coding**
8. The main.py- Containing the Source Point of the System
9. The dvc.yaml file is the **Data Version Control** file that enables ease of code execution and experimentations by saving the run time

![mlflow1](https://github.com/user-attachments/assets/78b01c2b-78eb-4135-a6ca-39c5faa72aba)

![mlflowapp](https://github.com/user-attachments/assets/f3e09e07-48c6-4624-b327-386eb6d7831a)
