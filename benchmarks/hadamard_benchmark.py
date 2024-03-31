import fast_hadamard_transform
import torch
import time
for i in [1024, 2048, 4096, 4096*2, 4096*3]:
    x = torch.rand(i, i).cuda().to(torch.float16)
    torch.cuda.synchronize()
    fp32_time = 0
    fp16_time = 0
    
    for j in range(10):
        timer = time.time()
        y_had_float = fast_hadamard_transform.hadamard_transform(x.float()).half()
        torch.cuda.synchronize()
        fp32_time += time.time() - timer
    torch.cuda.synchronize()
    print(fp32_time)
    
    for j in range(10):
        timer = time.time()
        y_had = fast_hadamard_transform.hadamard_transform(x)
        torch.cuda.synchronize()
        fp16_time += time.time() - timer
    torch.cuda.synchronize()
    print(fp16_time)
    print(torch.allclose(y_had, y_had_float, atol=1e-7))