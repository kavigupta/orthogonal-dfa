import os
import shutil

# This is meant to be run on kavi's machine; it gets the models from the SAM repo

for which in "donor", "acceptor":
    for seed in range(1, 1 + 5):
        path = f"/mnt/md0/ExpeditionsCommon/spliceai/Canonical/model/splicepoint-model-{which}-{seed}/model"
        path = os.path.join(path, max(os.listdir(path), key=int))
        shutil.copy(path, f"data/pretrained_models/{which}-{seed}.pt")


for which in "400", "10000":
    for seed in range(5):
        path = f"/mnt/md0/ExpeditionsCommon/spliceai/Canonical/model/standard-{which}-{seed}/model"
        path = os.path.join(path, max(os.listdir(path), key=int))
        which_disp = which if which != "10000" else "10k"
        shutil.copy(path, f"data/pretrained_models/spliceai-{which_disp}-{seed}.pt")
