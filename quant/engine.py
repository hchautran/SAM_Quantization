

from typing import List
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from segment_anything import sam_model_registry,  SamPredictor
from typing import Union
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from torch.data.utils import Dataloader





class InferenceStrategy(ABC):

    @abstractmethod
    def inference(self,  *args, **kwargs) -> torch.Tensor:
        pass


    @abstractmethod
    def build_predictor(self)-> Union[SAM2VideoPredictor, SAM2ImagePredictor, SamPredictor]:
        pass

class DataLoaderStrategy(ABC):
    @abstractmethod
    def get_loader(self,  *args, **kwargs) -> :
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
    def hq_image_inference(self):
        pass

    @abstractmethod
    def image_inference(self):
        pass







if __name__ == '__main__':
    sam_ckt = '../pretrained/sam_vit_b_4b8939.pth'
    model_type = 'vit_b'
    device = 'cuda'
    sam = sam_model(model_type=model_type, checkpoint=sam_ckt, device=device)
class SAMEngine(Engine):
    def __init__(self, predictor):
        super().__init__(SAMInferenceStrategy(predictor))

    def hq_image_inference(self):
        pass

    def image_inference(
        self,
        image_dir: torch.Tensor,
        show_image:bool= False
    ) -> torch.Tensor:
        return self.strategy.inference(image_dir)

    def video_inference(
        self,
        video_dir:str='./notebooks/videos/bedroom',
        show_video:bool=False
    ) -> torch.Tensor:
        raise NotImplementedError("SAM model does not support inference for video")




if __name__ == '__main__':
    sam_ckt = '../pretrained/sam_vit_b_4b8939.pth'
    model_type = 'vit_b'
    device = 'cuda'
    sam = sam_model(model_type=model_type, checkpoint=sam_ckt, device=device)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    for i in range(8):
        print("image:   ",i)
        # hq_token_only: False means use hq output to correct SAM output.
        #                True means use hq output only.
        #                Default: False
        hq_token_only = False
        # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
        # For images contain single object, we suggest to set hq_token_only = True
        # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

        image = cv2.imread('demo/input_imgs/example'+str(i)+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        if i==0:
            input_box = np.array([[4,13,1007,1023]])
            input_point, input_label = None, None
        elif i==1:
            input_box = np.array([[306, 132, 925, 893]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==2:
            input_point = np.array([[495,518],[217,140]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
            hq_token_only = True
        elif i==3:
            input_point = np.array([[221,482],[498,633],[750,379]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==4:
            input_box = np.array([[64,76,940,919]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==5:
            input_point = np.array([[373,363], [452, 575]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==6:
            input_box = np.array([[181, 196, 757, 495]])
            input_point, input_label = None, None
        elif i==7:
            # multi box input
            input_box = torch.tensor([[45,260,515,470], [310,228,424,296]],device=predictor.device)
            transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
            input_point, input_label = None, None

        batch_box = False if input_box is None else len(input_box)>1
        result_path = 'demo/hq_sam_result/'
        os.makedirs(result_path, exist_ok=True)

        if not batch_box:
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box = input_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
            show_res(masks,scores,input_point, input_label, input_box, result_path + 'example'+str(i), image)

        else:
            masks, scores, logits = predictor.predict_torch(
                point_coords=input_point,
                point_labels=input_label,
                boxes=transformed_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
            masks = masks.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            input_box = input_box.cpu().numpy()
            show_res_multi(masks, scores, input_point, input_label, input_box, result_path + 'example'+str(i), image)
