# cafe-recommendar-galle
A hybrid recommendation engine for tourism in Galle district, Sri Lanka

How to Run This Project 

Pre-requisites
1. Download the zip folder and open the "galle-recommender-258838M" folder

Steps
1. Run Data Collection (Optional — Skip since the CSVs are Already Exist)
2. Run Part A: Big Data Analytics
   - Upload the Raw CSVs ( galle_places.csv and galle_google_reviews.csv)
   - Download the Cleaned CSVs (Optional)
   - Run All
3. Run Part B: Recommendation System
   - Upload the Cleaned CSVs (galle_places_cleaned.csv and galle_reviews_cleaned.csv)
   - Run All
4. Run the Streamlit App Locally
	(i) Open File Explorer, navigate to  galle-recommender-258838M folder and open the command prompt (For Windows)
	(ii) Create a Virtual Environment (python -m venv venv-> venv\Scripts\activate)
	(iii) Install Dependencies (pip install -r requirements.txt)
	(iv)  Launch the App (streamlit run app.py)
	(v) You should see the 3-tab interface:

		Tab 1 — New Visitor: Pick area + category + budget → click "Get Recommendations"
		Tab 2 — I Liked This Place: Pick a place from the dropdown → click "Find Similar Places"
		Tab 3 — Explore Galle: Browse the interactive map and analytics

