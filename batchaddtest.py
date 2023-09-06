import numpy as np
import time
import torch

def compute_lora_out_v2(X, A_list, B_list):
    n = X.shape[0]
    lora_out = torch.zeros((n, 4096), device="cuda")
    for i in range(n):
        x = X[i, :]
        A = A_list[i]
        B = B_list[i]
        lora_out[i, :] = torch.matmul(torch.matmul(x, A.T), B.T)
        
    return lora_out

def compute_lora_out_batched(X, A_list, B_list):
    n = X.shape[0]
    A_batch = torch.stack(A_list).reshape(n, 1, 16, 4096)
    B_batch = torch.stack(B_list).reshape(n, 4096, 16, 1)
    X_reshaped = X[:, None, None, :]
    Y = (A_batch * X_reshaped).sum(dim=-1)
    lora_out = (Y[:, :, :, None] * B_batch).sum(dim=2)
    return lora_out.squeeze(dim=-1)


loop_times = []
batched_times = []

for _ in range(10000):
  n = 3
  X = torch.rand(n, 4096, device="cuda")
  A_list = [torch.rand(16, 4096, device="cuda") for _ in range(n)]
  B_list = [torch.rand(4096, 16, device="cuda") for _ in range(n)]

  start_time_batched = time.time()
  lora_out_batched = compute_lora_out_batched(X, A_list, B_list)
  end_time_batched = time.time()
  batched_time = end_time_batched - start_time_batched

  start_time_loop = time.time()
  lora_out_loop = compute_lora_out_v2(X, A_list, B_list)
  end_time_loop = time.time()
  loop_time = end_time_loop - start_time_loop

  are_close = torch.allclose(lora_out_loop, lora_out_batched)
  loop_times.append(loop_time)
  batched_times.append(batched_time)


# Calculate statistics
loop_mean = np.mean(loop_times)
batched_mean = np.mean(batched_times)
loop_median = np.median(loop_times)
batched_median = np.median(batched_times)

print(f"Loop Mean: {loop_mean}")
print(f"Batched Mean: {batched_mean}")
print(f"Loop Median: {loop_median}")
print(f"Batched Median: {batched_median}")

loop_sum = sum(loop_times)
batched_sum = sum(batched_times)

print(f"loop_sum: {loop_sum}")
print(f"batched_sum: {batched_sum}")