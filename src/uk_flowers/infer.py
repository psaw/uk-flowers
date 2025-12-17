import torch
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

from uk_flowers.model.module import FlowerClassifier
from uk_flowers.utils.common import seed_everything


def infer(cfg: DictConfig) -> None:
    """
    Inference pipeline.
    """
    # Set seed
    seed_everything(cfg.project.seed)

    # Load model from checkpoint
    inference_cfg = cfg.get("inference", {})
    checkpoint_path_str = inference_cfg.get("checkpoint_path")

    if not checkpoint_path_str:
        # Try to find the best checkpoint in the base save directory
        # We use cfg.train.base_save_dir which is static, unlike save_dir which has timestamps
        base_save_dir = Path(cfg.train.base_save_dir)

        if not base_save_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {base_save_dir} not found.")

        # Search recursively for all checkpoints
        checkpoints = list(base_save_dir.rglob("*.ckpt"))

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {base_save_dir}")

        # Sort by modification time to find the latest
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path_str)

    model = FlowerClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    # Device
    device = torch.device(
        cfg.project.device
        if cfg.project.device != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    # Transforms (same as validation/test)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load image(s)
    image_path_str = inference_cfg.get("image_path")
    if not image_path_str:
        raise ValueError("cfg.inference.image_path must be provided")

    image_path = Path(image_path_str)

    if image_path.is_dir():
        image_paths = list(image_path.rglob("*.jpg"))
    else:
        image_paths = [image_path]

    # Load class mapping
    data_dir = Path(cfg.data.data_dir)
    name_json = data_dir / "cat_to_name.json"
    with name_json.open("r") as f:
        cat_to_name = json.load(f)

    print(f"Starting inference on {len(image_paths)} images...")

    results = {}

    for img_p in image_paths:
        try:
            with img_p.open("rb") as f:
                img_pil = Image.open(f)
                img = img_pil.convert("RGB")

            img_tensor = test_transforms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                top_prob, top_class = probs.topk(1, dim=1)

                predicted_idx = top_class.item()
                predicted_prob = top_prob.item()

                # Map index to class name
                # Note: The model output index corresponds to the sorted class folders (1...102)
                # cat_to_name maps "1" -> "pink primrose"
                # We assume the model's class index i corresponds to folder name str(i+1) if 0-indexed
                # But ImageFolder sorts classes. If folders are 1, 10, 100... sorting might be tricky.
                # Ideally we should save class_to_idx from training.
                # For this dataset, folders are integers.

                # Let's try to map directly if possible, or just output index
                class_name = cat_to_name.get(str(predicted_idx), "Unknown")

                results[str(img_p)] = {
                    "class_idx": predicted_idx,
                    "class_name": class_name,
                    "probability": predicted_prob,
                }

                print(
                    f"Image: {img_p.name}, Class: {class_name} ({predicted_idx}), Prob: {predicted_prob:.4f}"
                )

        except Exception as e:
            print(f"Error processing {img_p}: {e}")

    # Save results
    output_path = Path(inference_cfg.get("output_path", "inference_results.json"))
    with output_path.open("w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")
