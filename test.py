import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from models import *
from PIL import Image

"""
想要加载训练模型，并用自己的数据测试
"""
img_path = './a_new_truck.png'
image = Image.open(img_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)

# cifar 默认类型
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

net = SimpleDLA()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)  # 用gpu训练出来的需要用gpu加载，否则报错
net = torch.nn.DataParallel(net)
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.eval()
image = torch.reshape(image, (1, 3, 32, 32))

# img = torchvision.utils.make_grid(image).numpy()
# plt.imshow(np.transpose(img, (1, 2, 0)))
# plt.show()

output = net(image)
print(classes[output.argmax(1)[0].item()])
