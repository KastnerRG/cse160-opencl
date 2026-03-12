from train_audio import preprocess_function
import datasets
from PIL import Image

if __name__ == "__main__":
    dataset = datasets.load_dataset("DBD-research-group/BirdSet", "PER", trust_remote_code=True)
    dataset["train"] = dataset["train"].cast_column( "audio", datasets.Audio(sampling_rate=32_000))

    selected_birds = dataset["train"].select([6, 10, 12, 18, 20, 23])
    selected_birds.save_to_disk("Dataset/audio/selected_bird_datasets")
