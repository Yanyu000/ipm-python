from __future__ import annotations

import site
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop


class develop(_develop):
    def run(self):
        src_path = Path(__file__).resolve().parent / "src"
        site_paths = []

        if hasattr(site, "getsitepackages"):
            site_paths.extend(site.getsitepackages())

        user_site = site.getusersitepackages()
        if user_site:
            site_paths.append(user_site)

        for site_path in site_paths:
            try:
                target_dir = Path(site_path)
                target_dir.mkdir(parents=True, exist_ok=True)
                pth_file = target_dir / "hipdlp_proto_editable.pth"
                pth_file.write_text(str(src_path) + "\n", encoding="utf-8")
                print(f"Installed editable path: {pth_file}")
                return
            except OSError:
                continue

        raise RuntimeError("Could not determine a writable site-packages directory")


setup(
    name="hipdlp-proto",
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    cmdclass={"develop": develop},
)