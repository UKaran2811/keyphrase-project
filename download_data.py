"""Download the Inspec keyphrase-extraction benchmark (Hulth, 2003) in the
CoreNLP-processed form distributed by Boudin's ake-datasets repository.

Source : https://github.com/boudinfl/ake-datasets  (folder datasets/Inspec)
Needs  : git (version >= 2.25 for sparse checkout) and an internet connection.
Output : data/Inspec/{train,dev,test,references}
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/boudinfl/ake-datasets.git"
TMP = os.path.join("data", "_ake_tmp")
DEST = os.path.join("data", "Inspec")


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if os.path.isdir(os.path.join(DEST, "test")) and os.path.isdir(os.path.join(DEST, "references")):
        print("Inspec already present at", DEST)
        return
    os.makedirs("data", exist_ok=True)
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, TMP])
    run(["git", "sparse-checkout", "set", "datasets/Inspec"], cwd=TMP)
    shutil.move(os.path.join(TMP, "datasets", "Inspec"), DEST)
    shutil.rmtree(TMP, ignore_errors=True)
    # the raw archive and preprocessing scripts are not needed to run the project
    shutil.rmtree(os.path.join(DEST, "src"), ignore_errors=True)
    for split in ("train", "dev", "test"):
        n = len([f for f in os.listdir(os.path.join(DEST, split)) if f.endswith(".xml")])
        print(f"{split:5s}: {n} documents")
    print("Done. Dataset stored in", DEST)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print("Download failed:", e)
        print("Manual alternative: clone https://github.com/boudinfl/ake-datasets and copy "
              "datasets/Inspec to data/Inspec")
        sys.exit(1)
