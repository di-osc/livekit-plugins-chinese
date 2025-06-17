import os
from pathlib import Path

from dotenv import load_dotenv
from jsonargparse import auto_cli

load_dotenv()


def run(version: str):
    plugins = Path("livekit-plugins")
    for plugin in plugins.iterdir():
        if not plugin.is_dir():
            continue
        if not (plugin / "pyproject.toml").exists():
            continue
        plugin_name = plugin.stem.replace("livekit-plugins-", "")
        print(f"Building {plugin_name} for version {version}")
        # 修改version.py文件
        with open(plugin / f"livekit/plugins/{plugin_name}" / "version.py", "w") as f:
            f.write(f'''__version__ = "{version}"''')
        print(f"Building {plugin.name} for version {version}")
        os.system(f"cd {plugin} && uv build")

    pypi_token = os.getenv("PYPI_TOKEN")
    if pypi_token:
        os.system(f"uv publish --token {pypi_token}")


if __name__ == "__main__":
    auto_cli(run)
