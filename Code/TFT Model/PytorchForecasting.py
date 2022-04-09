# ---------------------------------------------------#
#
#   File       : PytorchForecasting.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: Assembling and training the model
#                using Pytorch
#
#
# ----------------------------------------------------#


#Imports
#############################

#General
import datetime
import time
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import warnings
#Pytorch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

##############################


warnings.filterwarnings("ignore")  #to avoid printing out absolute paths


class FTSEDataSet:
    """
    FTSE Dataset

    Extracsts the data from the CSV file
    Runs through data loaders
    Null values are removed
    Dataset is split into training, validation and testing datasets
    Converted into an appropriate format for the TFT

    """

    def __init__(self, start=datetime.datetime(2010, 1, 1), stop=datetime.datetime.now()):
        self.df_returns = None
        self.stocks_file_name = "/home/soham/Documents/PycharmProjects/NEA/Code/Data/NEAFTSE2010-21.csv"
        #self.start = start
        #self.stop = stop

    def load(self, binary = True):

        #start = self.start
        #end = self.stop

        df0 = pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True)
        print(df0)


        df0.dropna(axis=1, how='all', inplace=True)
        df0.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:",
              df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)# changed here
        df0 = df0.drop(df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns, 1)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())

        df_returns = pd.DataFrame()
        print(df_returns)
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()
        print(df_returns)


        # split into train and test
        df_returns.dropna(axis=0, how='any', inplace=True)
        if binary:
            df_returns.FTSE = [1 if ftse > 0 else 0 for ftse in df_returns.FTSE]
        self.df_returns = df_returns
        return df_returns

    def get_loaders(self, batch_size=16, n_test=1000, device='cpu'):
        if self.df_returns is None:
            self.load()

        features = self.df_returns.drop('Open', axis=1).values
        labels = self.df_returns.FTSE
        training_data = data_utils.TensorDataset(torch.tensor(features[:-n_test]).float().to(device),
                                                 torch.tensor(labels[:-n_test]).float().to(device))
        test_data = data_utils.TensorDataset(torch.tensor(features[n_test:]).float().to(device),
                                             torch.tensor(labels[n_test:]).float().to(device))
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader



class TFT:

    """
    Temporal Fusion Transformer

    Setting up the model using PyTorch lighting.
    The class determines the main key features of the model, listed below:

    Tuneable Hyperparameters:
        int prediction length
        str   features
        int   max encoder length
        int   training cutoff
        str   time index
        str   group ids
        int   min encoder length
        int   min prediction length
        str   target
        int   max epochs
        int   gpus
        int   learning rate
        int   hidden layer size
        int   drop out
        int   hidden continous size
        int   output size
        int   attention head size
        float loss function

    """

    def __init__(self, prediction_length = 2000):
        self.prediction_length = prediction_length
        self.training = None
        self.validation = None
        self.trainer = None
        self.model = None
        self.batch_size =16

    def load_data(self):
        """
        Load data using the FTSEDataSet class
        Set prediction and encoder lengths
        Set up training data using TimeSeriesDataSet function
        """

        dataset = FTSEDataSet()
        print("Dataset",dataset)
        ftse_df = dataset.load(binary=False)
        print(dataset)
        time_index = "Date"
        target = "Open"
        features = ftse_df.columns.tolist()
        print("Features",features)
        features.remove(target)

        ftse_df[time_index] = pd.to_datetime(ftse_df.index)
        min_date = ftse_df[time_index].min()
        ftse_df[time_index] = (ftse_df[time_index] - min_date).dt.days

        ftse_df["Open_Prediction"] = "Open"
        print("ftse_df",ftse_df)
        max_encoder_length = 4192
        training_cutoff = ftse_df[time_index].max() - self.prediction_length
        print("Training cutoff",training_cutoff)
        print('time_idx',time_index)
        print("ftse_dftp2",ftse_df[lambda x: x[time_index] <= training_cutoff])

        self.training = TimeSeriesDataSet(
            ftse_df[lambda x: x[time_index] <= training_cutoff],
            time_idx=time_index,#changed here
            target="Open",
            categorical_encoders={"Open_Prediction": NaNLabelEncoder().fit(ftse_df.Open_Prediction)},
            group_ids=["Open_Prediction"],#"Open_Prediction"
            min_encoder_length= max_encoder_length // 2 , # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length ,
            min_prediction_length=1,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=features,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        print(self.training.get_parameters())

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        self.validation = TimeSeriesDataSet.from_dataset(self.training, ftse_df, predict=True, stop_randomization=True)

    def create_tft_model(self):
        """
        Create the model
        Define hyperparameters
        Declare input, hidden, drop out, attention head and output size
        Declare epochs


        TFT Design
            1. Variable Selection Network
            2. LSTM Encoder
            3. Normalisation
            4. GRN
            5. MutiHead Attention
            6. Normalisation
            7. GRN
            8. Normalisation
            9. Dense network
            10.Quantile outputs


        """
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        self.trainer = pl.Trainer(
            max_epochs=10,
            gpus=0,
            weights_summary="top",
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.05,
            hidden_size= 4,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=4,  # set to <= hidden_size
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {self.model.size() / 1e3:.1f}k")

    def train(self):
        # create dataloaders for model
        train_dataloader = self.training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)

        # fit network
        self.trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate(self, number_of_examples = 15):
        """
        Evaluate the model
        Load the saved model from the last saved epoch
        Compare predictions against real values
        Create graphs to visualise performance
        """
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)
        raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
        #print('raw_predictions', raw_predictions)
        for idx in range(number_of_examples):  # plot 10 examples
            best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);

        predictions, x = best_tft.predict(val_dataloader, return_x=True)
        #print('predictions2', predictions)
        #print('x values', x)
        predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
        #print('predictions_vs_actuals', predictions_vs_actuals)
        best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);
        #best_tft.plot(predictions,x)
        # print(best_tft)



def tft():
    tft = TFT()
    tft.load_data()
    tft.create_tft_model()
    tft.train()
    #torch.save(tft,"Model.pickle")
    tft.evaluate(number_of_examples=1)
    plt.show()







