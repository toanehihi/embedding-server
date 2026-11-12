from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EMBEDDING_", extra="ignore")

    api_key: SecretStr = Field(min_length=16)
    model_id: str = "BAAI/bge-m3"
    model_revision: str | None = None
    model_name: str = "bge-m3"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    use_fp16: bool = True  # Applied only on CUDA; CPU always uses float32.
    batch_size: int = Field(default=8, ge=1, le=128)
    max_batch_size: int = Field(default=32, ge=1, le=128)
    max_seq_length: int = Field(default=8192, ge=3, le=8192)
    max_total_tokens: int = Field(default=16384, ge=1)
