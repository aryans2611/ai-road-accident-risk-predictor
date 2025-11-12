 AI Road Accident Risk Predictor
#

 Overview
#
A machine learning project that predicts road accident severity and visualizes risk-prone routes using interactive maps.
Built using the US Accidents (March 2023) dataset.
#
 Objective

To forecast accident risk based on environmental and geographical factors like:

Temperature

Visibility

Wind Speed

Weather Condition

Latitude / Longitude
#
 Technologies Used

Python 3, scikit-learn, pandas, numpy, folium, matplotlib

IDE: VS Code / Google Colab
#
 Model Details

Algorithm: Random Forest Classifier

Accuracy: ~87%

Outputs:

Accident_Hotspots_Map.html — shows accident severity across locations

Lucknow_Patna_Route_Risk.html — predicts risk along Lucknow → Patna route
#
 How to Run
pip install pandas numpy scikit-learn folium matplotlib
python ai_road_accident_predictor.py

#
 Open generated maps in your browser.
#
 Folder Structure
AI Road Accident Risk Predictor/
├── ai_road_accident_predictor.py
├── US_Accidents_March23.csv
├── Accident_Hotspots_Map.html
├── Lucknow_Patna_Route_Risk.html
└── README.md
#
 Future Scope

Live weather integration

Streamlit web app

Real-time route prediction


  ##  Dataset
The dataset used in this project is too large for direct GitHub upload (≈ 2.8 GB).  
You can download it from Google Drive:

[Download US_Accidents_March23.csv](https://drive.google.com/file/d/1EOJc5BfG0HLJZF3xXM9LrLQakcrDlUX-/view?usp=sharing)


