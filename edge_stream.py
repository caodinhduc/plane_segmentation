import torch
import torch.nn.functional as F
from network import Resnet
from torch import nn
from network.mynn import initialize_weights, Norm2d
from torch.autograd import Variable

from my_functionals import GatedSpatialConv as gsc


class GSCNN(nn.Module):
    '''
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7
      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    '''

    def __init__(self):
        
        super(GSCNN, self).__init__()
        
        self.interpolate = F.interpolate

        self.dsn1 = nn.Conv2d(512, 1, 1)
        self.dsn2 = nn.Conv2d(1024, 1, 1)
        self.dsn3 = nn.Conv2d(2048, 1, 1)

        self.res1 = Resnet.BasicBlock(256, 256, stride=1, downsample=None)
        self.d1 = nn.Conv2d(256, 32, 1)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
         
        # self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
        #                                                output_stride=8)

        # self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        # self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, features, x_size =(640,480),gts=None):
        """
        features: backbone feature: 
        256 x 120 x 160
        512 x 60 x 80
        1024 x 30 x 40
        2048 x 15 x 20

        output_size = 480 x 640
        """

        x_size = (480, 640)
        # first level
        m1f = F.interpolate(features[0], x_size, mode='bilinear', align_corners=True)
        s1 = F.interpolate(self.dsn1(features[1]), x_size,
                            mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.dsn2(features[2]), x_size,
                            mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.dsn3(features[3]), x_size,
                            mode='bilinear', align_corners=True)
        

        # im_arr = inp.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        # canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        # for i in range(x_size[0]):
        #     canny[i] = cv2.Canny(im_arr[i],10,100)
        # canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(m1f)
        cs = F.interpolate(cs, x_size,
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s1)

        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size,
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s2)
        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size,
                           mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s3)
        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size,
                           mode='bilinear', align_corners=True)

        edge_out = self.sigmoid(cs)
        return edge_out
