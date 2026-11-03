import numpy as np

from recsys24.data import ML_GENRES, generate_world, load_dataset, parse_movielens_1m, temporal_split
from tests_helpers import ML_MOVIES, ML_RATINGS


def test_generator_is_seeded_and_well_formed(small_ds):
    again = generate_world(small_ds.world.cfg)
    assert small_ds.interactions.equals(again.interactions)
    df = small_ds.interactions
    assert set(df.columns) == {"user", "item", "t"}
    assert not df.duplicated(["user", "item"]).any()  # a title is watched at most once
    assert df.groupby("user").size().min() >= small_ds.world.cfg.min_events
    assert df["item"].between(0, small_ds.n_items - 1).all()
    # every event happens after the title's release
    assert (df["t"].to_numpy() >= small_ds.world.item_release[df["item"].to_numpy()]).all()
    assert small_ds.item_genres.sum(1).min() >= 1


def test_popularity_is_skewed(small_ds):
    counts = np.sort(np.bincount(small_ds.interactions["item"], minlength=small_ds.n_items))[::-1]
    top10 = counts[: len(counts) // 10].sum() / counts.sum()
    assert top10 > 0.2


def test_temporal_split_takes_last_events(small_ds):
    sp = temporal_split(small_ds.interactions)
    assert len(sp.test) == 5 * small_ds.n_users and len(sp.val) == 5 * small_ds.n_users
    last_train = sp.train.groupby("user")["t"].max()
    first_val = sp.val.groupby("user")["t"].min()
    first_test = sp.test.groupby("user")["t"].min()
    assert (last_train <= first_val).all() and (sp.val.groupby("user")["t"].max() <= first_test).all()


def test_movielens_parser():
    ds = parse_movielens_1m(ML_RATINGS, ML_MOVIES, min_rating=4, min_events=2)
    assert ds.path_used == "public_movielens_1m"
    assert ds.n_items == 3 and ds.n_users == 2  # user 3 has a single positive and is dropped
    assert ds.item_genres.shape == (3, len(ML_GENRES))
    assert ds.item_genres[0, ML_GENRES.index("Animation")] == 1
    assert ds.interactions["t"].between(0, 1).all()


def test_load_dataset_synthetic_mode(tmp_path):
    from tests_helpers import TINY_WORLD

    ds, info = load_dataset("synthetic", tmp_path, TINY_WORLD)
    assert info["path_used"] == "synthetic_generator" and ds.world is not None
