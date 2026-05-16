import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_data(places_path, reviews_path):
    places = pd.read_csv(places_path)
    reviews = pd.read_csv(reviews_path)

    places["rating"] = pd.to_numeric(places["rating"], errors="coerce")
    places["price_level"] = pd.to_numeric(places["price_level"], errors="coerce")

    review_counts = reviews.groupby("place_id").size().reset_index(name="review_count")
    places = places.merge(review_counts, on="place_id", how="left")
    places["review_count"] = places["review_count"].fillna(0).astype(int)

    # Replace empty user_ratings_total with actual review_count
    places["user_ratings_total"] = places["review_count"]

    if "primary_category" not in places.columns:
        places["primary_category"] = places["types"].astype(str).str.split(",").str[0]

    return places, reviews


def add_bayesian_score(places, M=50):
    df = places.copy()
    C = df["rating"].mean()

    v = df["user_ratings_total"].fillna(0)
    R = df["rating"]

    df["bayes_score"] = (v / (v + M)) * R + (M / (v + M)) * C
    df["bayes_score"] = df["bayes_score"].fillna(C)

    return df


def recommend_for_new_visitor(places, area=None, category=None, price_max=None, top_n=5):
    df = add_bayesian_score(places)

    df["price_level_filled"] = df["price_level"].fillna(2)

    if area:
        df = df[df["anchor_area"] == area]

    if category:
        df = df[df["primary_category"] == category]

    if price_max:
        df = df[df["price_level_filled"] <= price_max]

    return df.sort_values("bayes_score", ascending=False).head(top_n)


def recommend_similar_to(places, seed_name, top_n=5):
    df = places.copy()

    df = df.dropna(subset=["name", "rating"])
    df["price_level_filled"] = df["price_level"].fillna(2)

    if seed_name not in df["name"].values:
        return pd.DataFrame()

    features = pd.get_dummies(
        df[["anchor_area", "primary_category"]],
        dummy_na=True
    )

    numeric = df[["rating", "price_level_filled"]].fillna(0)
    feature_matrix = pd.concat([numeric.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    seed_idx = df[df["name"] == seed_name].index[0]
    seed_pos = df.index.get_loc(seed_idx)

    sims = cosine_similarity(
        feature_matrix.iloc[[seed_pos]],
        feature_matrix
    )[0]

    df["similarity"] = sims
    results = df[df["name"] != seed_name].sort_values("similarity", ascending=False)

    return results.head(top_n)