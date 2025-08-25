



#TO DO 
# _____ Quantize Model Weights ____ 

# RTN
# GPTQ
# _____ Save quantized 4 bit ____ 
# Pack 4 bit model to 8 bit

# _____Load quantized 4 bit ____ 
# Pack 4 bit model to 8 bit


# ____Implement Online Activation quantization____  

# Unpack weight
# Use sym_quant kernel to quant activation -> A_q, s_q
# A_q @ W_q = Y_q
# s_a_q @ s_w_q = S_q
# Y = S_q @ Y_q
#


# ___