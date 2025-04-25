def predict_image(img_path, model, class_mapping, device):
    from PIL import Image
    from torchvision import transforms
    import torch

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()

    idx_to_class = {v: k for k, v in class_mapping.items()}
    return idx_to_class[predicted_class_idx]
