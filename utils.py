import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    
    examples = [
       
        
    ]
    
    for i, (img_name, correct_caption) in enumerate(examples, start=1):
        img_path = f"C:/Users/vietn/Downloads/projectaiselfmade/flickr8k/Images/{img_name}"
        test_img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        output_caption = " ".join(model.caption_image(test_img.to(device), dataset.vocab))
        print(f"Example {i} CORRECT: {correct_caption}")
        print(f"Example {i} OUTPUT: {output_caption}")
    
    model.train()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
