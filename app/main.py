import json
from typing import List, Optional
import os
from io import BytesIO
import sys
from os import path
import asyncio
from time import time
from starlette.responses import JSONResponse

import torch

sys.setrecursionlimit(10**6)

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image
import numpy as np

from app.model import MainModel

class Setting:
    model = dict(
        checkpoint = '/home/HuyNguyen/Fabrics/PT-Map_Deploy/checkpoints/180.pth',
        samples_root = '/home/HuyNguyen/Fabrics/PT-Map_Deploy/test_base',
        n_shot = 5,
        crop_size = (256, 256),
        net_size = 84,
        device = torch.device('cuda')
    )    

model = MainModel(**Setting.model)

app = FastAPI()

@app.post("/demo-pipeline")
async def demo_pipeline(images: List[UploadFile] = File(...), masks: List[UploadFile] = File(...)):
    if len(images) != len(masks):
        return JSONResponse(jsonable_encoder({'error': f'Number of masks and images must be the same'}), status_code=400)
    
    try:
        
        images = [Image.open(BytesIO(await image.read())).convert('RGB') for image in images]
        masks = [Image.open(BytesIO(await mask.read())) for mask in masks]
        
    except Exception as ex:
        return JSONResponse(jsonable_encoder({'error': f"Upload file error {ex}"}), status_code=400)
    
    color_code = None
    color_name = None
    pattern_dict = dict()
    
    for index, image, mask in zip(range(len(images)), images, masks):
        result = await model(image, mask)
        
        pattern = result['pattern']
        color = result['color']
        pattern_id = pattern['label']
        
        pattern_dict[pattern_id] = pattern_dict.get(pattern_id, 0) + 1
        
        if index == len(images)//2:
            color_code = color['rgb']
            color_name = color['color_name']
    
    max_id = max(pattern_dict.keys(), key=lambda x: pattern_dict[x])
    
    return dict(
        function='pattern_matching',
        results=dict(
            pattern_id=max_id,
            color_code=color_code,
            color_name=color_name
        )
    )