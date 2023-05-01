import torch
import logging
import time
from torcheeg_helpers import set_seed, Trainer
from models import SEED_DEEP
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import SEEDDataset
from torcheeg.model_selection import KFoldGroupbyTrial

device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU: running on ", device)
print("This is a minimal script to recreate best results")

logger = logging.getLogger('seed-deep-cnn-lstm')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
logger.addHandler(console_handler)


set_seed(42)
sample = 200
dataset = SEEDDataset(io_path='./seed_full_200/seed',
                      root_path='./data/Preprocessed_EEG',
                      offline_transform=transforms.Compose([]),
                      online_transform=transforms.Compose([
                          transforms.ToTensor(),
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select('emotion'),
                          transforms.Lambda(lambda x: x + 1)
                      ]),
                      chunk_size=sample,
                      num_worker=24)

k = 5
k_fold = KFoldGroupbyTrial(n_splits=k,
                           split_path='./seed_full_200/split')
bs = 256
dropout1 = 0.5
dropout2 = 0.5
dropoutc = 0.2
num_res_layers = 5
c1 = 50
l1 = 1024
l2 = 1024
ll1 = 1024
ll2 = 768
epochs = 100
lr = 1e-4
wd = 1e-4
do_pool = True
scheduler_gamma = 0.85
step_size = 10

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
  set_seed(42)
  model = SEED_DEEP(do_pool=do_pool,
                    dropoutc=dropoutc,
                    in_channels=1,
                    num_classes=3,
                    out_channels=c1,
                    num_res_layers=num_res_layers,
                    ll1=ll1,
                    ll2=ll2,
                    dropout1=dropout1,
                    dropout2=dropout2)

  # Initialize the trainer and use the 0-th GPU for training
  trainer = Trainer(model=model,
                          lr=lr,
                          device_ids=[0],
                          weight_decay=wd,
                          do_scheduler=True,
                          scheduler_gamma=scheduler_gamma,
                          scheduler_step_size=step_size)
  trainer.logger = logger

  train_loader = DataLoader(train_dataset,
                            batch_size=bs,
                            shuffle=True,
                            num_workers=28,
                            pin_memory=True)
  val_loader = DataLoader(val_dataset,
                          batch_size=bs,
                          shuffle=False,
                          num_workers=28,
                          pin_memory=True)

  trainer.fit(train_loader, val_loader, num_epochs=epochs)
  trainer.test(val_loader)
