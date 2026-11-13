"""HF eyeball reference for SmolVLM-256M: prints the HF answer for a given image.

This is NOT a parity gate -- it prints the HuggingFace answer for eyeballing
against v7's dbg_image_recog output on the same image. Automated SmolVLM
coverage lives in test_vlms.bat (process exit codes) and the vision smoke
bats (byte baselines).

Usage: python test_smolvlm_hf.py <hf_model_dir> <image> [question]

  <hf_model_dir>  local HuggingFace dir (config.json + safetensors)
  <image>         a v7 .raw fixture (RGB888, size from the .raw+.meta pair)
                  or any .jpg/.png/.jpeg path
  [question]      defaults to "What is in this image?"
"""
import os
import sys

from PIL import Image

DEFAULT_QUESTION = "What is in this image?"


def load_image(image_path):
    """Load the image as an RGB PIL image.

    A .raw argument is a v7 RGB888 fixture (w*h*3 bytes, dimensions in the
    .meta sidecar "w h") -- exactly what dbg_image_recog.exe feeds the DLL.
    """
    if image_path.endswith(".raw"):
        meta_path = image_path[:-4] + ".meta"
        if not os.path.isfile(meta_path):
            sys.exit("[FATAL] no .meta sidecar next to the .raw fixture: " + meta_path)
        meta = open(meta_path).read().split()
        img_w, img_h = int(meta[0]), int(meta[1])
        raw = open(image_path, "rb").read()
        if len(raw) != img_w * img_h * 3:
            sys.exit(f"[FATAL] .raw is {len(raw)} bytes but .meta says {img_w}x{img_h}x3")
        return Image.frombytes("RGB", (img_w, img_h), raw)
    return Image.open(image_path).convert("RGB")


def main():
    if len(sys.argv) < 3:
        print("usage: test_smolvlm_hf.py <hf_model_dir> <image> [question]", file=sys.stderr)
        return 1
    hf_dir, image_path = sys.argv[1], sys.argv[2]
    question = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_QUESTION

    if image_path.endswith(".raw"):
        img = load_image(image_path)
    else:
        img = load_image(image_path)

    print(f"Model dir: {hf_dir}")
    print(f"Image: {image_path} {img.size[0]}x{img.size[1]}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(hf_dir, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(hf_dir)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    print("Generating...")
    # greedy: do_sample=False alone is deterministic; passing temperature
    # alongside it only produces a transformers warning
    output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"HF OUTPUT: {generated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
