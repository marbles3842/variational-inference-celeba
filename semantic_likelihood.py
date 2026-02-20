import jax
import jax.numpy as jnp
import numpy as np

class FaceNetLikelihood:
    """
    A lightweight wrapper that acts like a Flax module but actually
    uses a pre-trained PyTorch FaceNet via jax.pure_callback.
    """
    def __init__(self):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            
            # Keep on CPU to avoid conflicting with JAX's GPU preallocation
            self.model = InceptionResnetV1(pretrained='vggface2').eval().cpu()
        except ImportError:
            self.model = None

    def apply(self, variables, x, return_embedding=True):
        if self.model is None:
            raise ImportError(
                "facenet-pytorch is not installed. "
                "Please run: pip install facenet-pytorch torch torchvision"
            )
            
        import torch
        import torch.nn.functional as F

        def pytorch_facenet_forward(jax_images):
            # Convert JAX array back to CPU numpy, then to torch tensor
            imgs_np = np.array(jax_images)
            
            if imgs_np.ndim == 3:
                imgs_np = imgs_np[None, ...]
                
            imgs = torch.from_numpy(imgs_np).float().cpu()
            
            # If shape is [Batch, H, W, C], permute to PyTorch's [Batch, C, H, W]
            if imgs.shape[-1] == 3:
                imgs = imgs.permute(0, 3, 1, 2)
                
            # Upsample the 64x64 CelebA images to the 160x160 that FaceNet expects
            imgs = F.interpolate(imgs, size=(160, 160), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                # FaceNet returns un-normalized embeddings, we typically want standard L2 normalized
                embeddings = self.model(imgs)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                
            return embeddings.numpy()

        # FaceNet (InceptionResnetV1) outputs a 512-dimensional embedding vector
        out_shape = jax.ShapeDtypeStruct((x.shape[0], 512), jnp.float32)
        
        # pure_callback lets JAX call arbitrary python/numpy code inside JIT / map
        return jax.pure_callback(pytorch_facenet_forward, out_shape, x)

def init_semantic_likelihood(rng):
    """
    Returns the semantic likelihood wrapper and an empty params dict 
    since all params are stored inside the PyTorch model directly.
    """
    return FaceNetLikelihood(), {}
