"""
AI Road Accident Risk Predictor
Description:
    This project uses Machine Learning to predict accident severity
    and visualize accident-prone areas using Folium maps.
"""

#.............................................................
# 1. Import Required Libraries
#.............................................................
import pandas as pd
import numpy as np
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


#.............................................................
# 2. Load Dataset
#.............................................................
print("Loading dataset...")
file_path = "US_Accidents_March23.csv"   # Make sure the CSV is in the same folder
data = pd.read_csv(file_path, nrows=100000)
print("Dataset loaded successfully.")
print(data.head())


#.............................................................
# 3. Data Cleaning and Preparation
#.............................................................
print("\nCleaning and preparing data...")

df = data[['Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)',
           'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)', 'Weather_Condition']].dropna()

# Simplify weather conditions (extract only key terms)
df['Weather_Condition'] = df['Weather_Condition'].str.extract('(Rain|Snow|Fog|Clear|Cloud|Storm)', expand=False)
df['Weather_Condition'].fillna('Clear', inplace=True)

print("Sample of cleaned data:")
print(df.head())


#.............................................................
# 4. Encode Categorical Data
#.............................................................
print("\nEncoding weather conditions...")
le = LabelEncoder()
df['Weather_Condition'] = le.fit_transform(df['Weather_Condition'])
print("Encoded classes:", list(le.classes_))


#.............................................................
# 5. Split Data into Features and Target
#.............................................................
X = df[['Start_Lat', 'Start_Lng', 'Temperature(F)', 'Visibility(mi)',
        'Wind_Speed(mph)', 'Distance(mi)', 'Weather_Condition']]
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#.............................................................
# 6. Train the Machine Learning Model
#.............................................................
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model training completed. Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#.............................................................
# 7. Visualize Accident Hotspots (Interactive Map)
#.............................................................
print("\nGenerating accident hotspot map...")

sample_df = df.sample(300)
sample_df['Start_Lat'] = pd.to_numeric(sample_df['Start_Lat'], errors='coerce')
sample_df['Start_Lng'] = pd.to_numeric(sample_df['Start_Lng'], errors='coerce')
sample_df = sample_df.dropna(subset=['Start_Lat', 'Start_Lng'])

center_lat = sample_df['Start_Lat'].mean()
center_lng = sample_df['Start_Lng'].mean()
m = folium.Map(location=[center_lat, center_lng], zoom_start=6, tiles='CartoDB positron')

def get_color(severity):
    if severity == 1:
        return 'green'
    elif severity == 2:
        return 'orange'
    elif severity == 3:
        return 'red'
    else:
        return 'darkred'

for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['Start_Lat'], row['Start_Lng']],
        radius=3,
        color=get_color(int(row['Severity'])),
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

m.save("Accident_Hotspots_Map.html")
print("Hotspot map saved as 'Accident_Hotspots_Map.html'.")


#.............................................................
# 8. Smart Route Risk Prediction (Delhi → Agra)
#.............................................................
print("\nSimulating route risk prediction (Delhi → Agra)...")

route_points = [
    (28.6139, 77.2090),   # Delhi
    (28.3000, 77.3500),
    (27.9000, 77.5500),
    (27.6000, 77.7000),
    (27.2000, 77.9500)    # Agra
]

route_df = pd.DataFrame(route_points, columns=['Start_Lat', 'Start_Lng'])
route_df['Temperature(F)'] = np.random.randint(75, 95, len(route_points))
route_df['Visibility(mi)'] = np.random.randint(5, 10, len(route_points))
route_df['Wind_Speed(mph)'] = np.random.randint(1, 8, len(route_points))
route_df['Distance(mi)'] = np.random.uniform(1.0, 5.0, len(route_points))

# Encode "Clear" condition
clear_code = le.transform(['Clear'])[0]
route_df['Weather_Condition'] = [clear_code] * len(route_points)

# Predict risk levels along the route
route_df['Predicted_Risk'] = model.predict(route_df)

# Generate Folium map for route
m_route = folium.Map(location=[27.9, 77.5], zoom_start=7, tiles='CartoDB positron')

def risk_color(risk):
    if risk == 1:
        return 'green'
    elif risk == 2:
        return 'orange'
    elif risk == 3:
        return 'red'
    else:
        return 'darkred'

for _, row in route_df.iterrows():
    folium.CircleMarker(
        location=[row['Start_Lat'], row['Start_Lng']],
        radius=6,
        color=risk_color(int(row['Predicted_Risk'])),
        fill=True,
        fill_opacity=0.8,
        popup=f"Risk Level: {int(row['Predicted_Risk'])}"
    ).add_to(m_route)

folium.PolyLine(route_points, color='blue', weight=2, opacity=0.5).add_to(m_route)
m_route.save("Delhi_Agra_Route_Risk.html")

print("Route risk map saved as 'Delhi_Agra_Route_Risk.html'.")
print("\nAll processes completed successfully.")
