from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    data_location: str = Field(..., description="Location of the data")
    extension: str = Field(..., description="File extension")


class PreprocessConfig(BaseModel):
    sampling_rate: int = Field(..., description="Sampling rate")
    signal_length: int = Field(..., description="Signal length")
    block_size: int = Field(..., description="Block size")
    oneshot: bool = Field(..., description="One-shot flag")
    out_dir: str = Field(..., description="Output directory")


class ModelConfig(BaseModel):
    hidden_size: int = Field(..., description="Hidden size")
    n_harmonic: int = Field(..., description="Number of harmonics")
    n_bands: int = Field(..., description="Number of bands")
    sampling_rate: int = Field(..., description="Sampling rate")
    block_size: int = Field(..., description="Block size")


class TrainConfig(BaseModel):
    name: str = Field(..., description="Name of training session / model")
    scales: list[int] = Field(..., description="List of scales")
    overlap: float = Field(..., description="Overlap value")
    steps: int = Field(..., description="Number of steps")
    batch: int = Field(..., description="Batch size")


class Configuration(BaseModel):
    data: DataConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    train: TrainConfig
