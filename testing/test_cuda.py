# import torch 

# print(torch.cuda.is_available())
# print("number of gpu" , torch.cuda.device_count())
# print("name of gpu" , torch.cuda.get_device_name())


import tensorflow as tf
print("TF Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))