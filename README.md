## Lung Cancer Classification using CT Scans
This project focuses on classifying lung cancer by training on CT scan images of lungs. It involves a structured workflow for data ingestion, model training, evaluation, and more.

### Workflows
1. **Project Initialization**
Template.py: Creates the layout, files, and directories required for the project.
2. **Configuration Files**
config.yaml: Contains essential configurations such as API connections and paths.
params.yaml: Holds the hyperparameters settings for the model.
3. **Entity Models**
Entities: Represents the core data structures of the application, such as user, product, order, etc.
4. **Configuration Manager**
Configuration Manager (src/config): Contains necessary configuration, data utilities, and functions required for the components of the system.
5. **Project Components**
Data Ingestion: Collects and preprocesses the data.
Base Model Preparation: Sets up the initial model structure.
Model Training: Trains the model on the preprocessed data.
Model Evaluation: Evaluates the model's performance.
6. **Pipeline**
Pipeline Creation/Update: Contains the workflow stages 1, 2, 3, and 4 with modular coding for each stage.
7. **Main Entry Point**
main.py: The main entry point of the system, orchestrating the workflow.
8. **Data Version Control**
dvc.yaml: Enables ease of code execution and experimentation by saving the runtime configurations.

### Tech Stack
Python: Core programming language used for developing the application.
DVC (Data Version Control): Manages data and model versioning.
YAML: Used for configuration files.
API Integration: For connecting to external data sources.

### Getting Started
#### Prerequisites
Python 3.x
Git
DVC

### Installation
Clone the Repository:
git clone https://github.com/pavi2803/Lung_Cancer_Classification_CT_Scan.git
cd Lung_Cancer_Classification_CT_Scan

### Install Dependencies:

pip install -r requirements.txt

### Set Up Configuration Files:

Create and configure config.yaml with necessary API connections and paths.
Create and update params.yaml with hyperparameter settings.

Usage
* Initialize the Project Layout:

* python Template.py
* Run the Pipeline:
* python main.py
  
### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
For questions or suggestions, please contact us at pavi2468kuk@gmail.com
