import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

print('[*] Beginning server startup!')

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3' # Change this ID to an unused GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/api/test')
async def test():
	print("Got a ping to test!")
	return "API ping successful!"