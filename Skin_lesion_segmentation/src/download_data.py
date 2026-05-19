import argparse
import subprocess
from pathlib import Path


def download_ph2(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "PH2Dataset.rar"
    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
        return
    subprocess.run(["gdown", "1T_RPkPP0jeWwK8L1UrmBw8V30eD7v6Ql", "-O", str(archive_path)], check=True)
    print(f"Downloaded to {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PH2 dataset archive.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download_ph2(args.output_dir)


if __name__ == "__main__":
    main()
