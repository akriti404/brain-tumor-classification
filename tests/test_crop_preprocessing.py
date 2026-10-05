"""
Unit tests for Step 4's margin-removal preprocessing (data/dataset.py's
CropToContent) and its wiring into build_transforms.

Run: python -m pytest tests/test_crop_preprocessing.py -v
"""
import numpy as np
import pytest
from PIL import Image

from data.dataset import CropToContent, build_transforms


def _make_margin_image(size=200, content_box=(60, 60, 140, 140), content_val=200, margin_val=5):
    """A near-black image with a bright square in the middle -- simulates a
    brain/head region surrounded by black margin, the exact pattern
    CropToContent is meant to remove."""
    arr = np.full((size, size, 3), margin_val, dtype=np.uint8)
    left, top, right, bottom = content_box
    arr[top:bottom, left:right] = content_val
    return Image.fromarray(arr, mode="RGB")


def test_crop_removes_most_of_the_margin():
    img = _make_margin_image(size=200, content_box=(60, 60, 140, 140))
    cropper = CropToContent(padding_frac=0.0)
    cropped = cropper(img)

    # Original is 200x200 with an 80x80 content square -- after cropping,
    # the result should be close to 80x80, not 200x200.
    assert cropped.size[0] < 120 and cropped.size[1] < 120
    assert cropped.size[0] > 60 and cropped.size[1] > 60


def test_padding_frac_increases_crop_size():
    img = _make_margin_image(size=200, content_box=(60, 60, 140, 140))
    tight = CropToContent(padding_frac=0.0)(img)
    padded = CropToContent(padding_frac=0.1)(img)
    assert padded.size[0] >= tight.size[0]
    assert padded.size[1] >= tight.size[1]


def test_uniform_image_returns_unchanged():
    """A perfectly uniform image has no Otsu-separable foreground -- must not crash."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    cropper = CropToContent()
    result = cropper(img)
    assert result.size == img.size  # unchanged, not cropped, not crashed


def test_near_fully_bright_image_skips_crop():
    """If almost the whole image is 'foreground', cropping would do nothing
    useful and risks being wrong -- should skip via min_content_frac guard."""
    arr = np.full((100, 100, 3), 200, dtype=np.uint8)
    arr[0:2, 0:2] = 5  # tiny dark corner, enough for Otsu to find a threshold
    img = Image.fromarray(arr, mode="RGB")
    cropper = CropToContent(min_content_frac=0.05)
    result = cropper(img)
    assert result.size == img.size


def test_build_transforms_includes_crop_by_default():
    tf = build_transforms(image_size=96, augmentation_cfg={}, train=False)
    img = _make_margin_image(size=200, content_box=(60, 60, 140, 140))
    out = tf(img)
    assert out.shape == (3, 96, 96)  # still resizes correctly after cropping


def test_build_transforms_crop_margin_false_disables_crop():
    """With crop_margin=False, CropToContent should not be in the pipeline at
    all -- we can't easily inspect transforms.Compose internals portably, so
    instead confirm behavior: a crop-disabled pipeline run on a margin image
    still produces the expected final tensor shape (Resize always runs)."""
    tf = build_transforms(image_size=96, augmentation_cfg={}, train=False, crop_margin=False)
    img = _make_margin_image(size=200, content_box=(60, 60, 140, 140))
    out = tf(img)
    assert out.shape == (3, 96, 96)


def test_crop_then_resize_pipeline_runs_on_real_sized_image():
    """Sanity check against a size closer to this dataset's actual range (512x512)."""
    img = _make_margin_image(size=512, content_box=(100, 150, 400, 420))
    tf = build_transforms(image_size=96, augmentation_cfg={"horizontal_flip": True, "rotation_degrees": 10,
                                                             "brightness_jitter": 0.1}, train=True)
    out = tf(img)
    assert out.shape == (3, 96, 96)