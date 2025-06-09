from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from concrete.ml.onnx.convert import get_equivalent_numpy_forward_from_onnx_tree
import onnx
import os

app = FastAPI()

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/health")
async def health_check():
    return HealthResponse(status="healthy", version="1.0.0")

@app.post("/convert")
async def convert_model(model_path: str):
    try:
        # Load the ONNX model
        onnx_model = onnx.load(model_path)
        
        # Convert to ConcreteML format
        numpy_model = get_equivalent_numpy_forward_from_onnx_tree(onnx_model)
        
        return {"status": "success", "message": "Model converted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 