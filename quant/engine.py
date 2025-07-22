from typing import List
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from segment_anything import  SamPredictor
from typing import Union
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image





class InferenceStrategy(ABC):

    @abstractmethod
    def inference(self,  *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def build_predictor(self)-> Union[SAM2VideoPredictor, SAM2ImagePredictor, SamPredictor]:
        pass

    @abstractmethod
    def set_image(self, image_dir:str):
        pass

    @abstractmethod
    def set_video(self, video_dir:str):
        pass






class Sam2Engine:
    def __init__(self, strategy:InferenceStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy:InferenceStrategy):
        self._strategy = strategy


    @abstractmethod
    def sample_image_inference(
        self,
        image_dir: torch.Tensor,
        show_image:bool= False
    ) -> torch.Tensor:
        pass


    @abstractmethod
    def sample_video_inference(
        self,
        video_dir:str='./notebooks/videos/bedroom',
        show_video:bool=False
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def hq_video_inference(self):
        pass

    @abstractmethod
    def hq_image_inference(self):
        pass

    @abstractmethod
    def image_inference(self):
        pass

    @abstractmethod
    def video_inference(self):
        pass



class SamEngine:
    def __init__(self, strategy:InferenceStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy:InferenceStrategy):
        self._strategy = strategy


    @abstractmethod
    def sample_image_inference(
        self,
        image_dir: torch.Tensor,
        show_image:bool= False
    ) -> torch.Tensor:
        pass



    @abstractmethod
    def image_inference(self):
        pass



