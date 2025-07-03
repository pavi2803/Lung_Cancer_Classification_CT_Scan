from src.lungClassifier.constants import *
from src.lungClassifier.utils.common import read_yaml, create_directories
from src.lungClassifier.entity.config_entity import DataIngestionConfig
import os
import zipfile
import gdown
from src.lungClassifier import logger
from src.lungClassifier.utils.common import get_size
from src.lungClassifier.constants import *
from src.lungClassifier.utils.common import read_yaml, create_directories
from src.lungClassifier.entity.config_entity import PrepareBaseModelConfig
import os
import urllib.request as request
from zipfile import ZipFile
from src.lungClassifier.config.configuration import ConfigurationManager  # adjust if in a different file
from src.lungClassifier.components.prepare_base_model import PrepareBaseModel  # adjust import
import tensorflow as tf
from src.lungClassifier.constants import *
from src.lungClassifier.utils.common import read_yaml, create_directories
import tensorflow as tf
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


try:
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
    print("Data has been extracted!")
except Exception as e:
    raise e


####-------------------------------------------------------------------------------------------------------------###


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


try:
    config_manager = ConfigurationManager()
    prepare_base_model_config = config_manager.get_prepare_base_model_config()

    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    
    prepare_base_model.get_base_model()         # Loads and saves the base model
    prepare_base_model.update_base_model()      # Freezes layers, adds dense head, saves updated model

except Exception as e:
    raise e


###----------------------------------------------------------------------------------------------------------------###


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
        )

        return training_config
    

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):

        self.train_generator = tf.keras.utils.image_dataset_from_directory(
        directory=self.config.training_data,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=self.config.params_image_size[:-1],
        batch_size=self.config.params_batch_size,
        label_mode="categorical"
    )

        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode="categorical"
        )

        # Normalize pixel values (rescale to 0-1)
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        self.train_generator = self.train_generator.map(lambda x, y: (normalization_layer(x), y))
        self.valid_generator = self.valid_generator.map(lambda x, y: (normalization_layer(x), y))

        # Optional: speed up pipeline
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_generator = self.train_generator.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.valid_generator = self.valid_generator.cache().prefetch(buffer_size=AUTOTUNE)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    
    def train(self):
        # self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        # self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.steps_per_epoch = tf.data.experimental.cardinality(self.train_generator).numpy()
        self.validation_steps = tf.data.experimental.cardinality(self.valid_generator).numpy()      

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )



tf.config.run_functions_eagerly(True)

try:
    config = ConfigurationManager()
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train()
    
except Exception as e:
    raise e

##### ---------------------------EVAL Mode -----------------------------------###

import dagshub
import mlflow
from src.lungClassifier.constants import *
from src.lungClassifier.utils.common import read_yaml, create_directories, save_json
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse


dagshub.init(repo_owner='pavi2803', repo_name='Lung_Cancer_Classification_CT_Scan', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


import tensorflow as tf
model = tf.keras.models.load_model('artifacts/training/model.h5')


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int


class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])


    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/Chest-CT-Scan-data",
            mlflow_uri="https://dagshub.com/pavi2803/Lung_Cancer_Classification_CT_Scan.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
    

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    # def log_into_mlflow(self):
    #     mlflow.set_registry_uri(self.config.mlflow_uri)
    #     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
    #     with mlflow.start_run():
    #         mlflow.log_params(self.config.all_params)
    #         mlflow.log_metrics(
    #             {"loss": self.score[0], "accuracy": self.score[1]}
    #         )
    #         # Model registry does not work with file store
    #         mlflow.keras.log_model(self.model, "model")

    import os

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # ✅ Save model locally
            local_model_path = "mlruns_model"
            os.makedirs(local_model_path, exist_ok=True)
            full_model_path = os.path.join(local_model_path, "model.h5")
            self.model.save(full_model_path)

            # ✅ Log manually to MLflow
            mlflow.log_artifact(full_model_path, artifact_path="model")

try:
    config = ConfigurationManager()
    eval_config = config.get_evaluation_config()
    evaluation = Evaluation(eval_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()

except Exception as e:
   raise e


