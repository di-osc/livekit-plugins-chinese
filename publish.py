import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

plugins = Path("livekit-plugins")
for plugin in plugins.iterdir():
    if not plugin.is_dir():
        continue
    if not (plugin / "pyproject.toml").exists():
        continue
    print(f"Building {plugin.name}")
    os.system(f"cd {plugin} && uv build")

pypi_token = os.getenv("PYPI_TOKEN")

if pypi_token:
    os.system(f"uv publish --token {pypi_token}")
