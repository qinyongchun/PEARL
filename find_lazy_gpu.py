import GPUtil
def find_lazy():
    # 获取所有显卡的GPU利用率
    gpu_list = GPUtil.getGPUs()

    # 根据显存占用率从低到高排序
    sorted_gpu_list = sorted(gpu_list, key=lambda gpu: gpu.memoryUtil)

    # 获取排名的字符串形式
    ranked_gpu_indices = [str(gpu_list.index(gpu)) for gpu in sorted_gpu_list]
    ranked_gpu_string = ",".join(ranked_gpu_indices)

    return ranked_gpu_string


# print( ranked_gpu_string)
