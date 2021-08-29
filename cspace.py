import subprocess


def set_cspace(img_path, cspace_path=None):
    """Given an image path and ICC file path, matches the image's colors to the ICC."""
    if cspace_path is None:
        return

    return subprocess.check_call(['sips', '-m', cspace_path, img_path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
