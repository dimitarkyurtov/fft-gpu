import os
import subprocess

def main():
    files_to_check = [
        ("build/Build/Products/Release/FastFourierTransformMetal", "Metal executable detected."),
        ("fft_cuda.exe", "Cuda executable detected."),
        ("fft_opencl.exe", "OpenCL executable detected."),
    ]
    
    metal_thread_group_length = 1024
    metal_cu = [1, 2, 4, 6, 8, 16, 32]

    cuda_thread_group_length = 1024
    cuda_sms = [1, 2, 4, 6, 8, 16, 32]

    opencl_thread_group_length = 1024
    opencl_cu = [1, 2, 4, 6, 8, 16, 32]
    
    for file_path, message in files_to_check:
        if os.path.exists(file_path):
            print(message)
            if file_path == files_to_check[0][0]:
                for cu in metal_cu:
                    try:
                        for i in [1,2,3]:
                            total_threads = cu*metal_thread_group_length
                            print(f"Executing - {i} - {file_path} -gpu {total_threads}:")
                            subprocess.run([file_path, "-gpu", str(total_threads)], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing {file_path} with argument {total_threads}: {e}")
                break
            if file_path == files_to_check[0][1]:
                for sm in cuda_sms:
                    try:
                        for i in [1,2,3]:
                            total_threads = sm*cuda_thread_group_length
                            print(f"Executing - {i} - {file_path} -gpu {total_threads}:")
                            subprocess.run([file_path, "-gpu", str(total_threads)], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing {file_path} with argument {total_threads}: {e}")
                break
            if file_path == files_to_check[0][2]:
                for cu in opencl_cu:
                    try:
                        for i in [1,2,3]:
                            total_threads = cu*opencl_thread_group_length
                            print(f"Executing - {i} - {file_path} -gpu {total_threads}:")
                            subprocess.run([file_path, "-gpu", str(total_threads)], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing {file_path} with argument {total_threads}: {e}")
                break
    else:
        print("No executable file found.")

if __name__ == "__main__":
    main()