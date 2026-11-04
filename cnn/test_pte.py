import json
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from executorch.runtime import Runtime, Verification
from datetime import datetime


# ================= CONFIGURATION =================

IMAGE_PATH = Path(
    r"D:\deep learning\plantvillage dataset\color\Tomato___Bacterial_spot\0b13b997-9957-4029-b2a4-ef4a046eb088___UF.GRC_BS_Lab Leaf 0595.JPG"
)

MODEL_PATH = Path("disease_cnn.pte")
CLASS_NAMES_PATH = Path("checkpoints/class_names.json")


# ================= IMAGE PREPROCESSING =================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return preprocess(image).unsqueeze(0)


# ================= DISEASE DETECTION =================

def main():

    print("\n" + "=" * 60)
    print("           AGRISENSE - AI PLANT DISEASE DETECTION")
    print("=" * 60)

    print("\n[1] Initializing system...")

    if not IMAGE_PATH.is_file():
        print("[ERROR] Image file not found!")
        return

    if not MODEL_PATH.is_file():
        print("[ERROR] ExecuTorch model not found!")
        return

    if not CLASS_NAMES_PATH.is_file():
        print("[ERROR] Class names file not found!")
        return

    with open(CLASS_NAMES_PATH, "r") as f:
        classes = json.load(f)

    print("[OK] Model files verified.")

    # Load image
    print("\n[2] Processing plant image...")

    image_tensor = preprocess_image(IMAGE_PATH)

    print("[OK] Image successfully processed.")
    print(f"[INFO] Image: {IMAGE_PATH.name}")
    print(f"[INFO] Input shape: {tuple(image_tensor.shape)}")

    # Load model
    print("\n[3] Loading AI model...")

    runtime = Runtime.get()

    program = runtime.load_program(
        MODEL_PATH,
        verification=Verification.Minimal
    )

    forward_method = program.load_method("forward")

    print("[OK] ExecuTorch model loaded.")

    # Run inference
    print("\n[4] Running disease detection...")

    start_time = datetime.now()

    outputs = forward_method.execute((image_tensor,))

    inference_time = (datetime.now() - start_time).total_seconds()

    logits = outputs[0]

    probabilities = torch.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(3, len(classes)),
        dim=1
    )

    predicted_idx = top_indices[0][0].item()
    predicted_class = classes[predicted_idx]

    confidence = top_probs[0][0].item() * 100

    # Detection status
    if "healthy" in predicted_class.lower():
        status = "HEALTHY LEAF"
    else:
        status = "DISEASE DETECTED"

    # Format disease name
    readable_name = predicted_class.replace("___", " - ").replace("_", " ")

    # ================= FINAL REPORT =================

    print("\n" + "=" * 60)
    print("                  DETECTION REPORT")
    print("=" * 60)

    print(f"\n  Image Name       : {IMAGE_PATH.name}")

    print(f"  Detection Status: {status}")

    print(f"  Predicted Class  : {readable_name}")

    print(f"  Confidence Score : {confidence:.2f}%")

    print(f"  Inference Time   : {inference_time:.4f} seconds")

    print("\n" + "-" * 60)
    print("                 TOP 3 PREDICTIONS")
    print("-" * 60)

    for rank in range(len(top_indices[0])):

        idx = top_indices[0][rank].item()
        score = top_probs[0][rank].item() * 100

        name = classes[idx].replace("___", " - ").replace("_", " ")

        print(f"\n  {rank + 1}. {name}")
        print(f"     Confidence: {score:.2f}%")

    print("\n" + "=" * 60)
    print("       AI INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nNOTE: Confidence scores are model-generated")
    print("probability estimates, not guaranteed accuracy.\n")


if __name__ == "__main__":
    main()