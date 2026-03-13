"""
Trains a model to detect species of interest from Birdset with pytorch_ocl.
This was used to create the acoustic model used for testing in this PA.

Note: This file is not required to run, this is purely for fun and to show how training works in Pytorch (and with openCL)!
You are not expected to learn about machine learning in this course. 

This will download the birdset dataset, which is large. Make sure you have sufficient disk space.

If you are interested in bioacoustics, ML training for audio, and domain shifts, Email shperry@ucsd.edu.

Install:
run `pip install librosa timm datasets=3.6.0 accelerate>=1.1.0'`

"""

import torch
import datasets
import transformers
import pytorch_ocl
import timm
from transformers import Trainer, TrainingArguments
import librosa
import numpy as np
from functools import cached_property

class PeruBirdClassifier(torch.nn.Module):
    def __init__(self, num_classes, device):
        super(PeruBirdClassifier, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes, in_chans=1)
        self.loss = torch.nn.BCEWithLogitsLoss().to(device)
        self.device = device
        self.model.to(device)

    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.loss.to(device)
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.model(input_ids)
        loss = None    
        if labels is not None:
            loss = self.loss(x, labels.float())

        return {"logits": x, "loss": loss}

class OCLTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    Hugging face sets the device automatically
    Which is cool and all but it has no idea about openCL

    So we have to override the device setup to make sure it uses openCL
    """
    @cached_property
    def _setup_devices(self) -> "torch.device":
        self.distributed_state = None
        self._n_gpu = 1
        return torch.device(device) 

def preprocess_function(examples):
    spectrograms = []
    labels = []
    for i in range(len(examples["audio"])):

        row = {k: v[i] for k, v in examples.items()}
        audio = row["audio"]["array"]

        target_length = 32_000 * 5
        if len(audio) >target_length: 
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        sample_rate = row["audio"]["sampling_rate"]
        
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=32_000, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = (spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db)
        spectrograms.append(spectrogram_db[None, ...].astype(np.float32))
        
        label = np.zeros(132, dtype=np.float32)
        label[row["ebird_code_multilabel"]] = 1.0
        labels.append(label)
    return {
        "input_ids": spectrograms,
        "labels": labels
    }

def train():    
    model = PeruBirdClassifier(num_classes=132, device=device)
    dataset = datasets.load_dataset("DBD-research-group/BirdSet", "PER", trust_remote_code=True)
    dataset["train"] = dataset["train"].cast_column( "audio", datasets.Audio(sampling_rate=32_000))
    dataset["test_5s"] = dataset["test_5s"].cast_column( "audio", datasets.Audio(sampling_rate=32_000))

    dataset["train"].set_transform(preprocess_function)
    dataset["test_5s"].set_transform(preprocess_function)
    

    t_args = OCLTrainingArguments(
        label_names=["labels"],
        remove_unused_columns=False,
        output_dir="./output",
        dataloader_pin_memory=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
    )
    model.to(device)
    # for param in model.parameters():
    #     print(param.device)
    # print(t_args.device, next(model.parameters()).device)

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_5s"],
        args=t_args,
    )

    trainer.train()

if __name__ == "__main__":
    train()