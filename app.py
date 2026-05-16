"""
Galle Cafes & Restaurants Recommender — Streamlit UI

A user-facing inference layer for the Spark-trained recommendation system.
Demonstrates real-world deployment architecture: Spark for batch training,
lightweight Python for online serving.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push this folder to GitHub → connect at https://streamlit.io/cloud → Deploy
"""
import streamlit as st
import base64
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recommender_logic import (
    load_data,
    add_bayesian_score,
    recommend_for_new_visitor,
    recommend_similar_to,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Galle Recommender",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached so the app stays snappy across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    places, reviews = load_data("galle_places_cleaned.csv", "galle_reviews_cleaned.csv")
    return places, reviews

places, reviews = get_data()

# Pre-compute lookups
AREAS = sorted([a for a in places['anchor_area'].dropna().unique()])
CATEGORIES = sorted([c for c in places['primary_category'].dropna().unique()])
PLACE_NAMES = sorted(places['name'].dropna().unique().tolist())

# Approximate LKR ranges for Google's price levels (Sri Lankan context)
PRICE_LKR = {
    1: "💵 Under Rs. 500",
    2: "💵 Rs. 500–1,500",
    3: "💵 Rs. 1,500–4,000",
    4: "💵 Above Rs. 4,000",
}

def price_label(level):
    """Return a human-readable LKR price hint, or 'Not specified' if missing."""
    if pd.isna(level):
        return "💵 Price not listed"
    try:
        return PRICE_LKR.get(int(level), "💵 Price not listed")
    except (ValueError, TypeError):
        return "💵 Price not listed"


def image_to_base64(image_path):
    """Convert a local image into base64 so Streamlit can use it in CSS."""
    path = Path(image_path)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def render_hero_section():
    """Render the top header section with a Galle background image."""
    bg_image = image_to_base64("assets/galle_background.jpeg")
    reviews_text = f"{len(reviews):,}" if reviews is not None else "—"
    avg_rating_text = f"⭐ {places['rating'].mean():.2f}" if places['rating'].notna().any() else "—"

    if bg_image:
        hero_background = (
            "background-image: linear-gradient(rgba(7, 18, 38, 0.62), rgba(7, 18, 38, 0.50)), "
            f"url('data:image/png;base64,{bg_image}');"
        )
    else:
        # Fallback background if the image file is missing
        hero_background = "background: linear-gradient(135deg, #0f172a, #0284c7);"

    st.markdown(
        f"""
        <style>
            .hero-section {{
                {hero_background}
                background-size: cover;
                background-position: center;
                border-radius: 28px;
                padding: 64px 56px 42px 56px;
                margin: 8px 0 28px 0;
                min-height: 430px;
                color: white;
                box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}

            .hero-kicker {{
                display: inline-block;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.18);
                backdrop-filter: blur(6px);
                font-size: 0.92rem;
                font-weight: 700;
                margin-bottom: 16px;
            }}

            .hero-section h1 {{
                font-size: clamp(2.2rem, 5vw, 4.8rem);
                line-height: 1.05;
                margin: 0 0 18px 0;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: white;
                max-width: 980px;
            }}

            .hero-section p {{
                max-width: 980px;
                font-size: 1.08rem;
                line-height: 1.7;
                color: rgba(255, 255, 255, 0.94);
                margin-bottom: 36px;
            }}

            .hero-stats {{
                display: grid;
                grid-template-columns: repeat(4, minmax(150px, 1fr));
                gap: 16px;
            }}

            .hero-stat-card {{
                background: rgba(255, 255, 255, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.30);
                border-radius: 20px;
                padding: 18px 20px;
                backdrop-filter: blur(8px);
            }}

            .hero-stat-card span {{
                display: block;
                font-size: 0.86rem;
                font-weight: 600;
                color: rgba(255, 255, 255, 0.86);
                margin-bottom: 8px;
            }}

            .hero-stat-card strong {{
                display: block;
                font-size: 2rem;
                line-height: 1;
                color: white;
            }}

            @media (max-width: 900px) {{
                .hero-section {{
                    padding: 42px 26px 30px 26px;
                    min-height: auto;
                }}
                .hero-stats {{
                    grid-template-columns: repeat(2, minmax(130px, 1fr));
                }}
            }}

            @media (max-width: 520px) {{
                .hero-stats {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>

        <section class="hero-section">
            <div>
                <div class="hero-kicker">Big Data Analytics Mini-Project</div>
                <h1>🍴 Galle Cafes &amp; Restaurants Recommender</h1>
                <p>
                    A hybrid recommendation system for cafes and restaurants in Galle district, Sri Lanka.
                    Built with Apache Spark for the training pipeline and Streamlit for the serving layer.
                </p>
            </div>
            <div class="hero-stats">
                <div class="hero-stat-card">
                    <span>Establishments</span>
                    <strong>{len(places):,}</strong>
                </div>
                <div class="hero-stat-card">
                    <span>Areas Covered</span>
                    <strong>{len(AREAS)}</strong>
                </div>
                <div class="hero-stat-card">
                    <span>Reviews</span>
                    <strong>{reviews_text}</strong>
                </div>
                <div class="hero-stat-card">
                    <span>Avg Rating</span>
                    <strong>{avg_rating_text}</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header with background image
# ─────────────────────────────────────────────────────────────────────────────
render_hero_section()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🆕 New Visitor",
    "❤️ I Liked This Place",
    "📊 Explore Galle",
])

# =============================================================================
# TAB 1 — New Visitor (Popularity-based, cold-start)
# =============================================================================
with tab1:
    st.subheader("New to Galle? Tell us your preferences.")
    st.caption(
        "Uses Bayesian-weighted popularity scoring. Places with many high ratings "
        "rank above places with one or two perfect-but-unverified scores."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        area_pref = st.selectbox(
            "Where are you staying?",
            options=["Any"] + AREAS,
            index=0,
        )
    with col_b:
        cat_pref = st.selectbox(
            "What are you looking for?",
            options=["Any"] + CATEGORIES,
            index=0,
        )
    with col_c:
        # Map Google price levels to approximate LKR ranges (typical Sri Lankan cafe/restaurant)
        PRICE_LABELS = {
            1: "Under Rs. 500",
            2: "Rs. 500–1,500",
            3: "1,500–4,000",
            4: "Above Rs. 4,000",
        }
        price_pref = st.selectbox(
            "Budget (max price level)",
            options=[1, 2, 3, 4],
            index=3,
            format_func=lambda x: PRICE_LABELS[x],
            help="Approximate cost per person for a meal. Set to your maximum acceptable price.",
        )

        st.caption(f"Selected max budget: **{PRICE_LABELS[price_pref]}**")

    n_recs = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Get Recommendations", type="primary", key="t1_btn"):
        results = recommend_for_new_visitor(
            places,
            area=None if area_pref == "Any" else area_pref,
            category=None if cat_pref == "Any" else cat_pref,
            price_max=price_pref,
            top_n=n_recs,
        )
        if results.empty:
            st.warning("No establishments match those preferences. Try widening your filters.")
        else:
            st.success(f"Found {len(results)} matches")
            for i, (_, row) in enumerate(results.iterrows(), 1):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{i}. {row['name']}**")
                        st.caption(f"📍 {row['anchor_area']} • {row['primary_category']} • {price_label(row.get('price_level'))}")
                        if pd.notna(row.get('address')):
                            st.caption(f"{row['address']}")
                    with c2:
                        st.metric("Rating", f"⭐ {row['rating']:.1f}")
                        st.caption(f"{int(row['user_ratings_total']):,} reviews")
                        st.caption(f"Score: {row['bayes_score']:.2f}")

# =============================================================================
# TAB 2 — I Liked This Place (Content-based)
# =============================================================================
with tab2:
    st.subheader("Found a place you love? We'll find similar ones.")
    st.caption(
        "Content-based filtering using cosine similarity on category, area, "
        "rating, popularity, and price features."
    )

    seed_place = st.selectbox(
        "Pick a place you enjoyed",
        options=PLACE_NAMES,
        index=0,
        help="Start typing to filter the list",
    )

    if seed_place:
        seed_info = places[places['name'] == seed_place].iloc[0]
        with st.container(border=True):
            st.markdown(f"**Selected:** {seed_info['name']}")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.caption(f"📍 {seed_info['anchor_area']}")
            cc2.caption(f"🍴 {seed_info['primary_category']}")
            cc3.caption(f"⭐ {seed_info['rating'] if pd.notna(seed_info['rating']) else 'N/A'}")
            cc4.caption(price_label(seed_info.get('price_level')))

    n_recs = st.slider("Number of similar places", 3, 10, 5, key="t2_slider")

    if st.button("Find Similar Places", type="primary", key="t2_btn"):
        with st.spinner("Computing similarity..."):
            results = recommend_similar_to(places, seed_place, top_n=n_recs)
        if results.empty:
            st.warning("Could not compute similarity for this place — try another.")
        else:
            st.success(f"Top {len(results)} similar places")
            for i, (_, row) in enumerate(results.iterrows(), 1):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{i}. {row['name']}**")
                        st.caption(f"📍 {row['anchor_area']} • {row['primary_category']} • {price_label(row.get('price_level'))}")
                        if pd.notna(row.get('address')):
                            st.caption(f"{row['address']}")
                    with c2:
                        st.metric("Rating", f"⭐ {row['rating']:.1f}")
                        st.caption(f"Similarity: {row['similarity']:.3f}")

# =============================================================================
# TAB 3 — Explore Galle (Analytics dashboard + map)
# =============================================================================
with tab3:
    st.subheader("Galle District at a Glance")
    st.write("")  # spacing

    # ── Map ──
    st.markdown("##### 📍 All establishments on the map")
    map_data = places.dropna(subset=['lat', 'lng'])[['lat', 'lng']].rename(
        columns={'lat': 'latitude', 'lng': 'longitude'}
    )
    if len(map_data):
        st.map(map_data, zoom=10, height=420)
        st.caption(f"Showing {len(map_data)} establishments across Galle district.")
    else:
        st.info("Geographic data unavailable.")

    st.write("")  # buffer space before next section
    st.divider()
    st.write("")

    # ── Side-by-side charts: Areas + Categories ──
    st.markdown("##### Distribution Across the District")
    st.write("")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**Establishments per Area**")
        area_counts = places['anchor_area'].value_counts().reset_index()
        area_counts.columns = ['Area', 'Count']
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=area_counts, y='Area', x='Count',
                    hue='Area', palette='viridis', ax=ax, legend=False)
        ax.set_xlabel("Number of Establishments")
        ax.set_ylabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i, v in enumerate(area_counts['Count']):
            ax.text(v + 1, i, str(int(v)), va='center', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown("**Category Breakdown**")
        cat_counts = places['primary_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']

        cat_counts['Percentage'] = (
        cat_counts['Count'] / cat_counts['Count'].sum() * 100
        ).round(1)

        fig, ax = plt.subplots(figsize=(7, 4.5))

        sns.barplot(
            data=cat_counts,
            y='Category',
            x='Count',
            hue='Category',
            palette='viridis',
            ax=ax,
            legend=False
        )

        ax.set_xlabel("Number of Establishments")
        ax.set_ylabel("")
        ax.set_title("Category Breakdown")

        for i, row in cat_counts.iterrows():
            ax.text(
                row['Count'] + 2,
                i,
                f"{int(row['Count'])} ({row['Percentage']}%)",
                va='center',
                fontsize=9
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.write("")
    st.divider()
    st.write("")

    # ── Top 10 leaderboard ──
    st.markdown("##### 🏆 Top 10 by Bayesian Score")
    scored = add_bayesian_score(places)
    top10 = scored.nlargest(10, 'bayes_score')[
        ['name', 'anchor_area', 'primary_category', 'rating', 'user_ratings_total', 'bayes_score']
    ].reset_index(drop=True)
    top10.index += 1
    top10.columns = ['Name', 'Area', 'Category', 'Rating', 'Reviews', 'Score']
    st.dataframe(top10, use_container_width=True)

    st.write("")
    st.divider()
    st.write("")

    # ── Rating distribution ──
    st.markdown("##### Rating Distribution")
    ratings = places['rating'].dropna()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(ratings, bins=20, color='#1f77b4', edgecolor='white')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Places")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(ratings.mean(), color='red', linestyle='--',
               label=f'Mean: {ratings.mean():.2f}')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built for the Big Data Analytics mini-project. Training pipeline: "
    "Apache Spark MLlib (ALS, FP-Growth, K-Means, Naive Bayes). "
    "Serving layer: Streamlit with pre-computed feature vectors. "
    "Data: custom dataset collected from Google Places API."
)
