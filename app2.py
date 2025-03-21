import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    layout="wide"
)

# App title and intro
st.title("Personal Fitness Tracker")
st.write("Predict calories burned based on your physical parameters and activity")

# Sidebar - User inputs
with st.sidebar:
    st.header("User Input Parameters")
    
    age = st.slider("Age:", 10, 100, 30)
    bmi = st.slider("BMI:", 15, 40, 20)
    duration = st.slider("Duration (min):", 0, 35, 15)
    heart_rate = st.slider("Heart Rate:", 60, 130, 80)
    body_temp = st.slider("Body Temperature (°C):", 36, 42, 38)
    gender = st.radio("Gender:", ("Male", "Female"))
    
    gender_value = 1 if gender == "Male" else 0
    
    # Create dataframe from user inputs
    user_data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_value
    }
    
    df = pd.DataFrame(user_data, index=[0])
    
    # Add a predict button
    predict_button = st.button("Predict Calories Burned")

# Load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    
    # Add BMI column
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)
    
    return exercise_df

exercise_df = load_data()

# Model training
@st.cache_resource
def train_model(exercise_df):
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
    
    # Prepare the training and testing sets
    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)
    
    # Separate features and labels
    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]
    
    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]
    
    # Train the model
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)
    
    return random_reg, X_train

model, X_train = train_model(exercise_df)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Your Parameters", "Similar Results", "Comparisons"])

# Make prediction when button is clicked
if predict_button:
    # Align prediction data columns with training data
    df_aligned = df.reindex(columns=X_train.columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(df_aligned)
    predicted_calories = round(prediction[0], 2)
    
    # Find similar results
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                               (exercise_df["Calories"] <= calorie_range[1])]
    
    # Calculate percentiles
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
    
    age_percentile = round(sum(boolean_age) / len(boolean_age), 2) * 100
    duration_percentile = round(sum(boolean_duration) / len(boolean_duration), 2) * 100
    heart_rate_percentile = round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100
    body_temp_percentile = round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100
    
    # Display in tabs
    with tab1:
        st.header("Predicted Calories Burned")
        
        # Display prediction in a more visual way
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric("Calories Burned", f"{predicted_calories} kcal")
            
        with col2:
            # Create a simple gauge chart using matplotlib
            fig, ax = plt.subplots(figsize=(4, 0.8))
            ax.barh([0], [predicted_calories], color='#FF4B4B')
            ax.barh([0], [500], color='#EEE', left=[predicted_calories])
            ax.set_xlim(0, 500)
            ax.set_yticks([])
            ax.set_xticks([0, 100, 200, 300, 400, 500])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            st.pyplot(fig)
    
    with tab2:
        st.header("Your Parameters")
        st.dataframe(df, use_container_width=True)
        
        # Display radar chart of parameters
        # Normalize the data for the radar chart
        categories = ['Age', 'BMI', 'Duration', 'Heart Rate', 'Body Temp']
        
        # Get min and max values for normalization
        min_values = [10, 15, 0, 60, 36]
        max_values = [100, 40, 35, 130, 42]
        
        # Normalize the values
        values = [
            (age - min_values[0]) / (max_values[0] - min_values[0]),
            (bmi - min_values[1]) / (max_values[1] - min_values[1]),
            (duration - min_values[2]) / (max_values[2] - min_values[2]),
            (heart_rate - min_values[3]) / (max_values[3] - min_values[3]),
            (body_temp - min_values[4]) / (max_values[4] - min_values[4])
        ]
        
        # Add the first value again to close the circle
        values.append(values[0])
        categories.append(categories[0])
        
        # Create radar chart
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot the values
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        ax.set_thetagrids(angles * 180/np.pi, categories)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab3:
        st.header("Similar Results")
        st.write("People with similar calorie burns:")
        st.dataframe(similar_data.sample(min(5, len(similar_data))), use_container_width=True)
        
        # Add visualization for similar results
        if not similar_data.empty:
            plt.figure(figsize=(10, 6))
            plt.scatter(similar_data['Age'], similar_data['Calories'], 
                       c=similar_data['Duration'], cmap='viridis', 
                       s=100, alpha=0.7)
            plt.colorbar(label='Duration (min)')
            plt.scatter([age], [predicted_calories], color='red', s=200, marker='*')
            plt.xlabel('Age')
            plt.ylabel('Calories Burned')
            plt.title('You vs Similar Results')
            
            st.pyplot(plt)
        
    with tab4:
        st.header("How You Compare")
        
        # Create columns for the metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Age Percentile", f"{age_percentile}%", 
                     delta="above average" if age_percentile > 50 else "below average")
            st.metric("Exercise Duration Percentile", f"{duration_percentile}%", 
                     delta="above average" if duration_percentile > 50 else "below average")
        
        with col2:
            st.metric("Heart Rate Percentile", f"{heart_rate_percentile}%", 
                     delta="above average" if heart_rate_percentile > 50 else "below average")
            st.metric("Body Temperature Percentile", f"{body_temp_percentile}%", 
                     delta="above average" if body_temp_percentile > 50 else "below average")
        
        # Add a bar chart to visualize percentiles
        percentiles = {
            'Age': age_percentile,
            'Duration': duration_percentile,
            'Heart Rate': heart_rate_percentile,
            'Body Temperature': body_temp_percentile
        }
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(percentiles.keys(), percentiles.values(), color=['#FF9671', '#FFC75F', '#008F7A', '#3498DB'])
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentile')
        ax.set_title('Your Metrics Compared to Others')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        st.pyplot(fig)
else:
    # Display instructions when the app first loads
    with tab1:
        st.info("👈 Adjust your parameters in the sidebar and click 'Predict Calories Burned' to see your results")
        
        # Add a sample image or illustration
        st.image("https://via.placeholder.com/800x400?text=Fitness+Tracker", use_column_width=True)
        
        # Quick guide
        st.subheader("How to use this app:")
        st.write("1. Enter your physical parameters in the sidebar")
        st.write("2. Click the 'Predict Calories Burned' button")
        st.write("3. Explore the different tabs to see your results and how you compare to others")