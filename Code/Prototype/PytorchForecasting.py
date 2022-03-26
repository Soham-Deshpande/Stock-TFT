from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from homework.dataset.SP500ReturnsDataSet import SP500ReturnsDataSet
import pandas as pd

import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

class TFTSP500:

    def __init__(self, prediction_length = 5):
        self.prediction_length = 5
        self.training = None
        self.validation = None
        self.trainer = None
        self.model = None
        self.batch_size = 128

    def load_data(self):
        dataset = SP500ReturnsDataSet()
        sp_df = dataset.load(binary=False)

        time_index = "date"
        target = "SPY"
        features = sp_df.columns.tolist()
        features.remove(target)

        sp_df[time_index] = pd.to_datetime(sp_df.index)
        min_date = sp_df[time_index].min()
        sp_df[time_index] = (sp_df[time_index] - min_date).dt.days

        sp_df["SPY_Prediction"] = "SPY"

        max_encoder_length = 24
        training_cutoff = sp_df[time_index].max() - self.prediction_length

        self.training = TimeSeriesDataSet(
            sp_df[lambda x: x[time_index] <= training_cutoff],
            time_idx=time_index,
            target=target,
            group_ids=["SPY_Prediction"],
            min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=features,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        self.validation = TimeSeriesDataSet.from_dataset(self.training, sp_df, predict=True, stop_randomization=True)

    def create_tft_model(self):
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        self.trainer = pl.Trainer(
            max_epochs=30,
            gpus=0,
            weights_summary="top",
            gradient_clip_val=0.1,
            limit_train_batches=30,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        self.mdel = TemporalFusionTransformer.from_dataset(
            self.training,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.03,
            hidden_size=16,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=8,  # set to <= hidden_size
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

    def evaluate(self, number_of_examples = 1):
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)
        raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

        for idx in range(number_of_examples):  # plot 10 examples
            best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);


