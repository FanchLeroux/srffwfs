from dataclasses import dataclass

import pathlib


@dataclass
class Config:
    root_dir: pathlib.Path = pathlib.Path(__file__).parents[2]
