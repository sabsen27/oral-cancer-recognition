import hydra
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.cae import Autoencoder
from src.models.mae import OralMAEModule
from src.models.dino import OralDinoModule
from src.models.vicreg import OralVICRegModule
from src.models.moco import OralMOCOModule
from src.data1.mae.datamodule import OralMAEDataModule
from src.data1.autoencoder.datamodule import OralAutoencoderDataModule
from src.data1.vicreg.datamodule import OralVICRegDataModule
from src.data1.dino.datamodule import OralDinoDataModule
from src.data1.moco.datamodule import OralMOCODataModule
from src.data1.classification.datamodule import OralClassificationDataModule
from src.data1.contrastive_classification.datamodule import OralContrastiveDataModule
from src.models.classification import *
from src.log import LossLogCallback, get_loggers, HydraTimestampRunCallback
from src.models.contrastive_classification import OralContrastiveClassifierModule
from lightly.transforms.dino_transform import DINOTransform
from torchvision import transforms
from src.utils import *
from test import predict
import os


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    
    if cfg.classification_mode == 'cae':
        torch.set_float32_matmul_precision("high")
    
    callbacks.append(get_early_stopping(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath = 'logs/oral/' + get_current_logging_version('logs/oral') + "/checkpoints/",
        save_on_train_epoch_end=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    callbacks.append(LossLogCallback())
    callbacks.append(HydraTimestampRunCallback())
    callbacks.append(checkpoint_callback)
    loggers = get_loggers(cfg)

    model, data = get_model_and_data(cfg)

    # training
    trainer = pl.Trainer(
        default_root_dir='logs/oral/' + get_current_logging_version('logs/oral') + "/checkpoints/",
        logger=loggers,
        callbacks=callbacks,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',  
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs
    )
    trainer.fit(model, data)

    if cfg.classification_mode == 'cae':
        #output_dir = "reconstructed_images/reconstructed_images"
        #output_dir = "reconstructed_images/reconstructed_images_bbox"
        # predict first batch and save predicted images
        imgs, labels, image_id, image_name = next(iter(data.test_dataloader()))
        y_hat = model(imgs)
        from matplotlib import pyplot as plt
        for i in range(len(imgs)):
            # plot both original and predicted images in the same figure
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(imgs[i].permute(1, 2, 0))
            ax[0].set_title("Original")
            ax[0].axis("off")
            ax[1].imshow(y_hat[i].detach().permute(1, 2, 0))
            ax[1].set_title("Reconstructed")
            ax[1].axis("off")

            # save the figure
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{image_name[i]}.png")
        # Test
        trainer.test(model, data)
        
    elif cfg.classification_mode == 'dino':
        # Test
        trainer.test(model, data)
    
    elif cfg.classification_mode == 'vicreg':
        # Test
        trainer.test(model, data)
    
    elif cfg.classification_mode == 'mae':
        output_dir = "reconstructe_images/reconstructed_images_mae"
        # predict first batch and save predicted images
        imgs, labels, image_id, image_name, imgs2 = next(iter(data.test_dataloader()))
        y_hat = model.reconstructe_images(imgs)
        from matplotlib import pyplot as plt
        for i in range(len(imgs2)):
            # plot both original and predicted images in the same figure
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(imgs2[i].cpu().permute(1, 2, 0))
            ax[0].set_title("Original")
            ax[0].axis("off")
            ax[1].imshow(y_hat[i].cpu().detach().permute(1, 2, 0))
            ax[1].set_title("Reconstructed")
            ax[1].axis("off")

            print(f'Image name: {image_name[i]}')

            # save the figure
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{image_name[i]}.png")
        # Test
        trainer.test(model, data)
    
    elif cfg.classification_mode == 'moco':
        # Test
        trainer.test(model, data)
    
    else:
        # test step
        predict(trainer, model, data, cfg.generate_map, cfg.task, cfg.classification_mode)


def get_model_and_data(cfg):
    ''' 
    This function returns a model and data based on the provided configuration.
    Depending on the task specified in the configuration, it can return either a classifier or a segmenter.
    Args:
        cfg: configuration
    Returns:
        model: model
        data: data
    '''
    model, data = None, None
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)

    # CLASSIFICATION WHOLE
    if cfg.task == 'c' or cfg.task == 'classification':
        if cfg.classification_mode == 'whole':
            # classification model
            model = OralClassifierModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                output_dim=cfg.model.output_dim,
                lr=cfg.train.lr,
                max_epochs = cfg.train.max_epochs
            )
            # whole data
            data = OralClassificationDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )

        # CONTRASTIVE LEARNING
        elif cfg.classification_mode == 'contrastive':
            # classification model
            model = OralContrastiveClassifierModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                output_dim=cfg.model.output_dim,
                lr=cfg.train.lr,
                max_epochs=cfg.train.max_epochs
            )
            # data
            data = OralContrastiveDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )

        # CAE
        elif cfg.classification_mode == 'cae':
            model = Autoencoder(
                cfg.ae,
                cfg.model.output_dim,
                cfg.train.lr,
                cfg.train.max_epochs
            )
            data = OralAutoencoderDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )

        # DINO
        elif cfg.classification_mode == 'dino':
            # classification model
            model = OralDinoModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                output_dim=cfg.model.output_dim,
                lr=cfg.train.lr,
                max_epochs=cfg.train.max_epochs
            )
            # data
            data = OralDinoDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )
        
        # VICREG
        elif cfg.classification_mode == 'vicreg':
            # classification model
            model = OralVICRegModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                output_dim=cfg.model.output_dim,
                lr=cfg.train.lr,
                max_epochs=cfg.train.max_epochs
            )
            # data
            data = OralVICRegDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )
        
        # MAE
        elif cfg.classification_mode == 'mae':
            # classification model
            model = OralMAEModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                output_dim=cfg.model.output_dim,
                lr=cfg.train.lr,
                max_epochs=cfg.train.max_epochs
            )
            # data
            data = OralMAEDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )
        
        # MOCO
        elif cfg.classification_mode == 'moco':
            # classification model
            model = OralMOCOModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                output_dim=cfg.model.output_dim,
                lr=cfg.train.lr,
                max_epochs=cfg.train.max_epochs
            )
            # data
            data = OralMOCODataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )
    
    return model, data

if __name__ == "__main__":
    main()

    