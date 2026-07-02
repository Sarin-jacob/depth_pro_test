from pathlib import Path
from urllib.request import urlopen
from tqdm import tqdm

URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
OUT = Path("checkpoints/depth_pro.pt")

OUT.parent.mkdir(parents=True, exist_ok=True)

with urlopen(URL) as response:
    total = int(response.headers.get("Content-Length", 0))

    with (
        open(OUT, "wb") as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=OUT.name,
        ) as pbar,
    ):
        while True:
            chunk = response.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))

print(f"Downloaded to {OUT}")