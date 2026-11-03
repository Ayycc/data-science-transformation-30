from recsys24.data import WorldConfig

TINY_WORLD = WorldConfig(n_users=60, n_items=80, mean_events=20, max_events=30)

ML_MOVIES = (
    "1::Toy Story (1995)::Animation|Children's|Comedy\n2::Jumanji (1995)::Adventure|Children's|Fantasy\n3::Heat (1995)::Action|Crime|Thriller\n"
)
ML_RATINGS = "\n".join(
    [
        "1::1::5::978300760",
        "1::2::4::978300800",
        "1::3::2::978300900",
        "2::3::5::978301000",
        "2::1::4::978302000",
        "3::2::5::978303000",
        "3::1::1::978304000",
    ]
)
