"""Packages the Chrome Extension into a distributable zip archive."""

from pathlib import Path
import zipfile


def package_extension() -> None:
    ext_dir = Path("chrome-extension")
    out_zip = Path("neuromeet-chrome-extension-v1.0.0.zip")
    web_zip = Path("web") / "neuromeet-chrome-extension-v1.0.0.zip"

    print(f"Packaging {ext_dir} into {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in ext_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(ext_dir)
                z.write(file_path, arcname)
                print(f"  + {arcname}")

    # Copy to web directory for 1-click download from Web Studio
    if Path("web").exists():
        import shutil
        shutil.copy(out_zip, web_zip)
        print(f"Copied to {web_zip} for direct download.")

    print(f"Successfully packaged {out_zip} ({out_zip.stat().st_size} bytes).")


if __name__ == "__main__":
    package_extension()
