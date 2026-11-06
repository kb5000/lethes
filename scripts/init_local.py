"""Initialize local deployment files without overwriting existing configuration."""

import argparse
import secrets
from pathlib import Path


def initialize(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        source = (root / ".env.example").read_text()
        source = source.replace(
            "LETHES_API_KEY=\n", "LETHES_API_KEY=" + secrets.token_hex(32) + "\n"
        )
        with env_path.open("x") as output:
            output.write(source)
        env_path.chmod(0o600)
        print("Created .env (secret not printed)")
    else:
        print("Kept existing .env")
    config = root / "deploy" / "lethes.local.json"
    if not config.exists():
        with config.open("x") as output:
            output.write((config.parent / "lethes.example.json").read_text())
        print("Created deploy/lethes.local.json")
    else:
        print("Kept existing deploy/lethes.local.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1])
    initialize(parser.parse_args().directory)
