from __future__ import annotations 

from collections import Counter 


from pathlib import Path 

import numpy as np  
from PIL import Image

import torch 
from torch.nn import functional as F 
from transformers import CLIPModel,CLIPProcessor

from src.settings.config import CLIPAdFilterThresholds

from src.constants import CLIP_MODEL_ID,DEVICE

from src.components.base import BaseFilter
from src.entities.crop import Crop 
from src.entities.stage_results import StageResult 

from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger 


class AdFilter(BaseFilter):
    """zero-shot ad/manequin filter"""
    name = "ad_filter"
    
    def __init__(self, 
                 thresholds: CLIPAdFilterThresholds,
                 device: str | None = None):
        self.thresholds = thresholds
        self.device = device or DEVICE

        logger.info(f"Loading {CLIP_MODEL_ID} on {DEVICE}")

        try:
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        except Exception as e:
            raise ModelLoadError(f"Failed to load CLIP from {CLIP_MODEL_ID} on {self.device}:{e}") from e 
        
        # cache prompt embeddings 
        self.real_prompts = thresholds.real_person_prompts
        self.ad_prompts = thresholds.ad_prompts
        self.real_embeddings = self._encode_text(self.real_prompts) 
        self.ad_embeddings = self._encode_text(self.ad_prompts)

        logger.info(f"CLIP model ready | {len(self.real_prompts)} real prompts, {len(self.ad_prompts)} ad prompts cached")

    def _encode_text(self,texts: list[str]) -> torch.Tensor:
        """Embed text prompts. Returns L2-normalized embeddings of shape (N,512)"""
        
        inputs = self.processor(text=texts,
                                return_tensors='pt',
                                padding=True).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        
        if not isinstance(embeddings,torch.Tensor):
            embeddings = embeddings.pooler_output
        
        embeddings = F.normalize(embeddings,dim=1)
        return embeddings 
    
    def _encode_image(self,img_rgb: np.ndarray) -> torch.Tensor:
        """Embed one RGB image. Returns L2-normalized embedding of shape (1, 512)"""
        
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img,return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)

        if not isinstance(embeddings,torch.Tensor):
            embeddings = embeddings.pooler_output
        
        embeddings = F.normalize(embeddings,dim=1)
        return embeddings 
    
    def _apply(self,crop: Crop) -> StageResult:
        img_emb = self._encode_image(crop.image)

        # compute similarities against both prompt sets
        real_scores = (img_emb @ self.real_embeddings.T).squeeze(0).cpu().numpy()
        ad_scores = (img_emb @ self.ad_embeddings.T).squeeze(0).cpu().numpy()

        best_real_score = float(real_scores.max())
        best_ad_score = float(ad_scores.max())
        best_real_idx = int(real_scores.argmax())
        best_ad_idx = int(ad_scores.argmax())
        margin = best_real_score - best_ad_score
        
        metrics: dict[str,float] = {"best_real_score":best_real_score,
                                    "best_ad_score":best_ad_score,
                                    "margin":margin,
                                    "best_real_idx":float(best_real_idx),
                                    "best_ad_idx":float(best_ad_idx)}

        t = self.thresholds
        failures: list[str] = []
        if margin < t.similarity_margin:
            failures.append(f"margin={margin:+.3f}<{t.similarity_margin}"
                            f" (best_ad='{self.ad_prompts[best_ad_idx][:30]}')")

        passed = len(failures) == 0 
        reason = 'ok' if passed else ' | '.join(failures)

        return StageResult(stage_name=self.name,
                           crop_id=crop.crop_id,
                           passed=passed,
                           reason=reason,
                           metrics=metrics)