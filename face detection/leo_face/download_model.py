"""
Download the ArcFace w600k_r50.onnx model from GitHub Releases.
"""
import urllib.request
import os
import zipfile

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ONNX_TARGET = os.path.join(MODEL_DIR, "w600k_r50.onnx")

if os.path.isfile(ONNX_TARGET):
    size_mb = os.path.getsize(ONNX_TARGET) / (1024 * 1024)
    print(f"Model already exists ({size_mb:.1f} MB): {ONNX_TARGET}")
    exit(0)

# Direct from GitHub releases (InsightFace v0.7)
ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_PATH = os.path.join(MODEL_DIR, "buffalo_l.zip")

print(f"Downloading buffalo_l.zip (~275 MB) from GitHub...")
print(f"URL: {ZIP_URL}")
print("This may take a few minutes depending on your internet speed...")

try:
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"Download complete! Extracting w600k_r50.onnx...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        for name in zf.namelist():
            if "w600k_r50.onnx" in name:
                data = zf.read(name)
                with open(ONNX_TARGET, "wb") as out:
                    out.write(data)
                size_mb = len(data) / (1024 * 1024)
                print(f"Extracted w600k_r50.onnx ({size_mb:.1f} MB)")
                break
        else:
            print("w600k_r50.onnx not found in zip. Listing contents:")
            for name in zf.namelist():
                print(f"  {name}")
    
    # Clean up zip
    os.remove(ZIP_PATH)
    print(f"Cleaned up zip file.")
    
    if os.path.isfile(ONNX_TARGET):
        print(f"\nDone! Model saved at: {ONNX_TARGET}")
    else:
        print("\nERROR: Model extraction failed.")
        
except Exception as e:
    print(f"\nDownload failed: {e}")
    print("\n=== MANUAL DOWNLOAD ===")
    print("1. Go to: https://github.com/deepinsight/insightface/releases/tag/v0.7")
    print("2. Download 'buffalo_l.zip'")
    print("3. Extract 'w600k_r50.onnx' from the zip")
    print(f"4. Place it at: {ONNX_TARGET}")
