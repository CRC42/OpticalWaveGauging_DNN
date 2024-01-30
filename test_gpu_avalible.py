import tensorflow as tf
import GPUtil

# 获取所有 GPU 的详细信息
gpus = GPUtil.getGPUs()

# 获取所有物理设备的列表
physical_devices = tf.config.list_physical_devices()

# 打印GPU的编号和名称
for device in physical_devices:
    if device.device_type == 'GPU':
        print("GPU ID:", device.name.split(":")[-1])
        print("GPU Name:", device.name)

# 打印 GPU 的型号
for gpu in gpus:
    print("GPU Name:", gpu.name)
