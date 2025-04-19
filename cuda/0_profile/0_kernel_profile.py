import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def trace_handler(prof, log_name):
    prof.export_chrome_trace(log_name)
    

def func_timer(func, inp):
    # CUDA is async
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(10):
        func(inp)
    end.record()
    torch.cuda.synchronize() 
    return start.elapsed_time(end)
   
def torch_square_fp32(a:torch.float32) -> torch.float32:
    return torch.square(a)

def torch_square_int8(a:torch.int8) -> torch.int8:
    return torch.square(a)

def torch_mul_fp32(a:torch.float32) -> torch.float32:
    return a*a

def torch_mul_int8(a:torch.int8) -> torch.int8:
    return a*a
    

def square_fp32(a:torch.float32 ) -> torch.float32: 
    return a**2 

def square_int8(a:torch.int8) -> torch.int8:
    return a**2

def mul_fp32(a:torch.float32) -> torch.float32:
    return a*a

def mul_int8(a:torch.int8) -> torch.int8:
    return a*a

if __name__=="__main__":
    a = torch.tensor([1.0,2.0,3.0, 4.0])
    a = a.to(torch.float32)
    a.to(device)
    b = torch.tensor([1,2,3,4])
    b = a.to(torch.int8)
    b.to(device)
    print("*******warmup*********")
    print("fp32 square : ", func_timer(square_fp32, a))
    print("fp32 mul : ", func_timer(mul_fp32, a))
    print("fp32 torch square : ", func_timer(torch_square_fp32, a))
    print("fp32 torch mul : ", func_timer(torch_mul_fp32, a))
    print("int8 square : ", func_timer(square_int8, b))
    print("int8 mul : ", func_timer(mul_int8, b))
    print("int8 torch square : ", func_timer(torch_square_int8, b))
    print("int8 torch mul : ", func_timer(torch_mul_int8, b))
    print("")
    print("*******warmup*********")
    # torch.cuda.empty_cache()
    # a = torch.tensor([10.0,20.0,30.0, 40.0])
    # a = a.to(torch.float32)
    # a.to(device)
    # b = torch.tensor([10,20,30,40])
    # b = a.to(torch.int8)
    # b.to(device)

    print("fp32 square : ", func_timer(square_fp32, a))
    print("fp32 mul : ", func_timer(mul_fp32, a))
    print("fp32 torch square : ", func_timer(torch_square_fp32, a))
    print("fp32 torch mul : ", func_timer(torch_mul_fp32, a))
    print("")
    
    print("int8 square : ", func_timer(square_int8, b))
    print("int8 mul : ", func_timer(mul_int8, b))
    print("int8 torch square : ", func_timer(torch_square_int8, b))
    print("int8 torch mul : ", func_timer(torch_mul_int8, b))
    print("")
    
    fp32_func_list = [square_fp32, mul_fp32, torch_square_fp32, torch_mul_fp32]
    int8_func_list = [square_int8, mul_int8, torch_square_int8, torch_mul_int8]
    
    
    for func in fp32_func_list:
        with torch.profiler.profile(
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA
                      ],
        schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('log/')##trace_handler(log_name="x")
        ) as prof:
            # print("profiling : ", func.__name__)
            # func_timer(func, a)
            # prof.step()
            
            for iter in range(10):
                #torch.square(torch.randn(10, 10).cuda())
                func_timer(torch_mul_int8, b)
                # send a signal to the profiler that the next iteration has started
                prof.step()
        #print(prof.key_averages().table(sort_by=f"{str(device)}_time_total"))
        
        
    for func in int8_func_list:
        with torch.profiler.profile() as prof:
            print("profiling : ", func.__name__)
            func_timer(func, b)
        print(prof.key_averages().table(sort_by=f"{str(device)}_time_total"))









# fp32 square :  0.04710400104522705
# fp32 mul :  0.03481600061058998
# fp32 torch square :  0.03891199827194214
# fp32 torch mul :  0.03379200026392937

# int8 square :  0.04403200000524521
# int8 mul :  0.03481600061058998
# int8 torch square :  0.03788800165057182
# int8 torch mul :  0.03379200026392937

# profiling :  square_fp32
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        32.26%      32.420us        32.26%      32.420us      16.210us             2  
#                 aten::pow        52.30%      52.567us        56.03%      56.315us       5.632us            10  
#         aten::result_type         2.73%       2.745us         2.73%       2.745us       0.275us            10  
#                  aten::to         1.00%       1.003us         1.00%       1.003us       0.100us            10  
#     cudaDeviceSynchronize         8.84%       8.887us         8.84%       8.887us       4.444us             2  
#      cudaEventElapsedTime         2.87%       2.886us         2.87%       2.886us       2.886us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 100.508us

# profiling :  mul_fp32
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        22.18%       9.918us        22.18%       9.918us       4.959us             2  
#                 aten::mul        54.84%      24.526us        54.84%      24.526us       2.453us            10  
#     cudaDeviceSynchronize        18.10%       8.096us        18.10%       8.096us       4.048us             2  
#      cudaEventElapsedTime         4.88%       2.184us         4.88%       2.184us       2.184us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 44.724us

# profiling :  torch_square_fp32
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        15.36%       9.227us        15.36%       9.227us       4.613us             2  
#              aten::square        12.26%       7.364us        68.19%      40.958us       4.096us            10  
#                 aten::pow        52.36%      31.449us        55.93%      33.594us       3.359us            10  
#         aten::result_type         2.20%       1.324us         2.20%       1.324us       0.132us            10  
#                  aten::to         1.37%       0.821us         1.37%       0.821us       0.082us            10  
#     cudaDeviceSynchronize        12.59%       7.564us        12.59%       7.564us       3.782us             2  
#      cudaEventElapsedTime         3.85%       2.314us         3.85%       2.314us       2.314us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 60.063us

# profiling :  torch_mul_fp32
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        22.60%       8.967us        22.60%       8.967us       4.483us             2  
#                 aten::mul        52.38%      20.781us        52.38%      20.781us       2.078us            10  
#     cudaDeviceSynchronize        18.96%       7.524us        18.96%       7.524us       3.762us             2  
#      cudaEventElapsedTime         6.06%       2.405us         6.06%       2.405us       2.405us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 39.677us

# profiling :  square_int8
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        18.06%       9.398us        18.06%       9.398us       4.699us             2  
#                 aten::pow        59.46%      30.939us        63.90%      33.251us       3.325us            10  
#         aten::result_type         2.79%       1.452us         2.79%       1.452us       0.145us            10  
#                  aten::to         1.65%       0.860us         1.65%       0.860us       0.086us            10  
#     cudaDeviceSynchronize        14.19%       7.384us        14.19%       7.384us       3.692us             2  
#      cudaEventElapsedTime         3.85%       2.004us         3.85%       2.004us       2.004us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 52.037us

# profiling :  mul_int8
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        23.42%       9.108us        23.42%       9.108us       4.554us             2  
#                 aten::mul        51.43%      19.997us        51.43%      19.997us       2.000us            10  
#     cudaDeviceSynchronize        19.66%       7.644us        19.66%       7.644us       3.822us             2  
#      cudaEventElapsedTime         5.49%       2.134us         5.49%       2.134us       2.134us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 38.883us

# profiling :  torch_square_int8
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        13.56%       8.867us        13.56%       8.867us       4.434us             2  
#              aten::square        10.80%       7.064us        72.53%      47.419us       4.742us            10  
#                 aten::pow        58.10%      37.989us        61.72%      40.355us       4.036us            10  
#         aten::result_type         2.22%       1.453us         2.22%       1.453us       0.145us            10  
#                  aten::to         1.40%       0.913us         1.40%       0.913us       0.091us            10  
#     cudaDeviceSynchronize        10.91%       7.133us        10.91%       7.133us       3.567us             2  
#      cudaEventElapsedTime         3.00%       1.963us         3.00%       1.963us       1.963us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 65.382us

# profiling :  torch_mul_int8
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
#           cudaEventRecord        23.54%       8.636us        23.54%       8.636us       4.318us             2  
#                 aten::mul        51.30%      18.815us        51.30%      18.815us       1.881us            10  
#     cudaDeviceSynchronize        20.10%       7.374us        20.10%       7.374us       3.687us             2  
#      cudaEventElapsedTime         5.05%       1.854us         5.05%       1.854us       1.854us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 36.679us