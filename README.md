# PES-PDS_EDA-Group05_Project
NIFTY 50 Stock Market Analysis & Prediction

📊 Project Overview
This project involves analyzing historical NIFTY 50 stock market data and predicting daily market movements using machine learning. The project is divided into two parts:
1.	A Jupyter Notebook (PDS_PythonProject.ipynb) for data exploration, cleaning, and model building.
2.	A Streamlit App (app.py) that provides a user-friendly interface for analysis and prediction.
   
________________________________________
📁 Files Included
•	NIFTY_50.csv: Historical stock data.
•	PDS_PythonProject.ipynb: Notebook with EDA and ML model.
•	app.py: Basic version of the Streamlit web app.
•	streamlit_app_improved.py: Enhanced version of the Streamlit app.
•	NIFTY50_Presentation.pptx: Project presentation.

________________________________________
🧪 Jupyter Notebook Highlights (PDS_PythonProject.ipynb)

✅ Steps Performed:
•	Data Loading & Cleaning:
o	Parsed dates
o	Handled missing values with mean/forward fill (no row drops)
•	Feature Engineering:
o	Daily_Movement: Up (1) or Down (0) based on Close vs Open
o	Price_Range, Volatility_Level
•	Exploratory Data Analysis:
o	Summary statistics, trend plots, heatmaps, categorical breakdowns
•	Model Training:
o	Random Forest Classifier
o	Train/Test split with evaluation: Accuracy ~90%, precision, recall

________________________________________

🌐 Streamlit App Highlights (streamlit_app_improved.py)

🔧 Features:
•	Upload your own CSV file
•	Date range selector
•	Interactive visualizations (price trends, volume, heatmaps)
•	Live market movement predictor with user inputs

🧠 Behind the Scenes:
•	The app loads the model trained in the notebook
•	Accepts user input for new prediction
•	Displays prediction (Market Up or Down)
•	Plots trends and statistics for uploaded data

________________________________________
👨‍💻 How to Run the App
1.	Install dependencies:
pip install streamlit pandas scikit-learn matplotlib seaborn
2.	Launch the app:
streamlit run streamlit_app_improved.py
________________________________________

📌 Future Improvements
•	Use advanced models like XGBoost or LSTM
•	Time-series forecasting
•	Integrate real-time stock data using APIs
•	Add model comparison dashboard in app

________________________________________

📬 Project Link
•	GitHub: https://github.com/Prashasti9/PES-PDS_EDA-Group05_Project.git

