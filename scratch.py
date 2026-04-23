import torch
import cutlass.cute as cute

@cute.kernel
def test_ptr_math(T: cute.Tensor):
    ptr = T.iterator + 2
    pass

try:
    cute.compile(test_ptr_math, cute.make_tensor(None, cute.make_layout((10,))))
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
