# Necessary dependencies
from torchvision import datasets, transforms
import torchvision.models as models
from vit_pytorch import ViT
from vit_pytorch.cvt import CvT
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# Loading datasets
normal_data = datasets.CIFAR10(root = "./data/cifar10", transform = transforms.ToTensor(), train = True, download = False)
corrupted_s1 = datasets.ImageFolder(root="corrupted_images_saturate1",
                           transform=transforms.ToTensor())
corrupted_s2 = datasets.ImageFolder(root="corrupted_images_saturate2",
                           transform=transforms.ToTensor())
corrupted_e1 = datasets.ImageFolder(root="corrupted_images_elastic1",
                           transform=transforms.ToTensor())
corrupted_e2 = datasets.ImageFolder(root="corrupted_images_elastic2",
                           transform=transforms.ToTensor())
corrupted_b1 = datasets.ImageFolder(root="corrupted_images",
                           transform=transforms.ToTensor())
corrupted_b2 = datasets.ImageFolder(root="corrupted_images_blur2",
                           transform=transforms.ToTensor())

# models
model_efb1 = models.efficientnet_b1(pretrained = False)
model_resnet = models.resnet18(pretrained = False)
model_cvt = CvT(
    # init='trunc_norm',
    num_classes = 10,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 6,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
)

model_vit = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 10,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Visualization using AblationCam
target_layer = model.layer4[-1]
input_tensor = normal_data.data[5] # Create an input tensor image for your model..

#Can be GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
cam = GradCAM(model=model_resnet, target_layer=target_layer, use_cuda=True)
grayscale_cam = cam(input_tensor=input_tensor, target_category=5)
rgb_img = Image.open("")
visualization = show_cam_on_image(rgb_img, grayscale_cam)