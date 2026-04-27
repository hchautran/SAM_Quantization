import torch
try:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    x = torch.zeros(1024, device='cuda')
    print("Allocation successful")
except Exception as e:
    print(f"Allocation failed: {e}")
