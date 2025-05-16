PROJECT TITLE:
Predictive Analysis of Toss Decisions in IPL Matches Using Big Data Techniques

TEAM MEMBERS:
- Namburi Eshwar Anish Varma
- Sathyapal Reddy Peddakkagari
- Vamshi Krishna Golla
- Paani Narisetty
- Venkata Sainandan Reddy Bhumireddypalli

COURSE:
AIT 614 – Big Data Essentials (Section 001)
Professor: Dr. Ben Duan
George Mason University


PROJECT DESCRIPTION
	This project analyzes the impact of toss decisions (batting or fielding first) on match outcomes in the Indian Premier League (IPL). It uses Big Data techniques including Apache Spark, PySpark, and machine learning (MLPClassifier) to predict optimal toss strategies. We built an interactive decision support system in Databricks, allowing users to choose match context parameters (city, teams, neutral venue, toss winner) via dropdown widgets, and receive a predicted toss strategy to maximize win probability.


PROJECT STRUCTURE

1. datasets/
   - matches.csv
   - deliveries.csv
   - IPL_BallByBall2008_2024(Updated).csv
   - ipl_teams_2024_info.csv
   - team_performance_dataset_2008to2024.csv
   - Players_Info_2024.csv
- Dataset link: IPL Complete Dataset (2008-2024)
(https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)


2. notebook/
   - AIT_614_IPL_Data_Analytics_Team_4.ipynb

3. README.txt (this file)

TECHNOLOGIES USED
- Apache Spark (PySpark)
- Databricks (Community Edition)
- Pandas, Matplotlib, Seaborn
- scikit-learn (MLPClassifier)
- Databricks widgets (UI dropdowns)
- Python 3.9

HOW TO RUN

1. Open Databricks and attach the notebook to your running cluster.
2. Upload all datasets to /FileStore/tables (in Databricks) or mount Google Drive (if in Colab).
3. Run the notebook `AIT_614_IPL_Data_Analytics_Team_4.ipynb` cell-by-cell.
4. In the prediction UI section:
   - Choose the city, teams, neutral venue, and toss winner from dropdowns.
   - The model will display the recommended toss decision (bat or field).


MODEL INFO
- Model: MLPClassifier (scikit-learn)
- Features:
   - City (One-hot encoded)
   - Team 1 and Team 2 (One-hot encoded)
   - Toss winner flag (binary)
   - Neutral venue flag (binary)
- Target: Whether toss winner also won the match
- Accuracy: ~70% on test set


CONTACT & CREDITS
Team 4 – AIT 614 – Spring 2025  
Acknowledgments: Dr.BenDuan for project guidance