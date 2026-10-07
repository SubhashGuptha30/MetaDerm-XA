"""
Unit tests for the data pipeline — Phase 3.
Tests dataset shape, dtypes, NaN checks, label validity.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import ISICDataset
from src.data.metadata_encoder import MetadataEncoder
from src.data.transforms import get_train_transforms, get_val_transforms


@pytest.fixture
def dummy_df():
    """Minimal DataFrame mimicking ISIC 2020 CSV structure."""
    return pd.DataFrame({
        "image_name": ["img_001", "img_002", "img_003", "img_004"],
        "target": [0, 1, 0, 0],
        "age_approx": [45.0, 62.0, np.nan, 38.0],
        "sex": ["male", "female", "unknown", "male"],
        "anatom_site_general_challenge": ["torso", "head/neck", "unknown", "lower extremity"],
        "patient_id": ["P001", "P002", "P003", "P004"],
        "age_missing": [0, 0, 1, 0],
    })


@pytest.fixture
def encoder(dummy_df):
    enc = MetadataEncoder()
    dummy_df = dummy_df.copy()
    dummy_df["age_approx"] = dummy_df["age_approx"].fillna(dummy_df["age_approx"].median())
    enc.fit(dummy_df)
    return enc


@pytest.fixture
def encoded_metadata(dummy_df, encoder):
    df = dummy_df.copy()
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())
    cont, sites = encoder.transform(df)
    return cont, sites


class TestMetadataEncoder:
    def test_fit_sets_stats(self, dummy_df, encoder):
        assert encoder._fitted
        assert isinstance(encoder.age_mean, float)
        assert encoder.age_std > 0

    def test_transform_shape(self, dummy_df, encoded_metadata):
        cont, sites = encoded_metadata
        assert cont.shape == (4, 5), f"Expected (4, 5), got {cont.shape}"
        assert sites.shape == (4,)

    def test_no_nans_in_continuous(self, encoded_metadata):
        cont, _ = encoded_metadata
        assert not np.isnan(cont).any(), "NaN found in continuous metadata"

    def test_sex_onehot_sums_to_one(self, encoded_metadata):
        cont, _ = encoded_metadata
        # Columns 1, 2, 3 are sex one-hot
        sex_rows = cont[:, 1:4]
        row_sums = sex_rows.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"Sex one-hot does not sum to 1: {row_sums}"

    def test_site_indices_in_range(self, encoded_metadata, encoder):
        _, sites = encoded_metadata
        assert (sites >= 0).all()
        assert (sites < encoder.num_sites).all()

    def test_save_load_roundtrip(self, encoder, tmp_path):
        save_path = tmp_path / "encoder.json"
        encoder.save(save_path)
        loaded = MetadataEncoder.load(save_path)
        assert abs(loaded.age_mean - encoder.age_mean) < 1e-6
        assert abs(loaded.age_std - encoder.age_std) < 1e-6


class TestISICDataset:
    """Tests dataset with synthetic images (no real files needed)."""

    @pytest.fixture
    def dataset_with_fake_images(self, dummy_df, encoded_metadata, tmp_path):
        # Create fake JPEG images
        from PIL import Image
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for name in dummy_df["image_name"]:
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(images_dir / f"{name}.jpg")

        cont, sites = encoded_metadata
        ds = ISICDataset(
            df=dummy_df,
            images_dir=images_dir,
            metadata_continuous=cont,
            site_indices=sites,
            transform=get_val_transforms(224),
        )
        return ds

    def test_length(self, dataset_with_fake_images, dummy_df):
        assert len(dataset_with_fake_images) == len(dummy_df)

    def test_item_types(self, dataset_with_fake_images):
        img, meta, site, label, name = dataset_with_fake_images[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(meta, torch.Tensor)
        assert isinstance(site, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(name, str)

    def test_image_shape(self, dataset_with_fake_images):
        img, _, _, _, _ = dataset_with_fake_images[0]
        assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"

    def test_label_binary(self, dataset_with_fake_images, dummy_df):
        for i in range(len(dataset_with_fake_images)):
            _, _, _, label, _ = dataset_with_fake_images[i]
            assert label.item() in (0.0, 1.0), f"Non-binary label: {label.item()}"

    def test_no_nans_in_metadata(self, dataset_with_fake_images):
        for i in range(len(dataset_with_fake_images)):
            _, meta, site, _, _ = dataset_with_fake_images[i]
            assert not torch.isnan(meta).any()
            assert not torch.isnan(site.float()).any()

    def test_image_dtype(self, dataset_with_fake_images):
        img, _, _, _, _ = dataset_with_fake_images[0]
        assert img.dtype == torch.float32

    def test_train_transforms_applied(self, dummy_df, encoded_metadata, tmp_path):
        from PIL import Image
        images_dir = tmp_path / "images2"
        images_dir.mkdir()
        for name in dummy_df["image_name"]:
            img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
            img.save(images_dir / f"{name}.jpg")

        cont, sites = encoded_metadata
        ds = ISICDataset(
            df=dummy_df,
            images_dir=images_dir,
            metadata_continuous=cont,
            site_indices=sites,
            transform=get_train_transforms(224),
        )
        img, _, _, _, _ = ds[0]
        assert img.shape == (3, 224, 224)
