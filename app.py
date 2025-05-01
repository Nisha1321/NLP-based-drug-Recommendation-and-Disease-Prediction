import streamlit as st
import joblib
import pandas as pd

# Load the trained model and vectorizer
model = joblib.load('passmodel.pkl')
vectorizer = joblib.load('tfidfvectorizer.pkl')

# Load or define the DataFrame `df` for `top_drugs_extractor` function
df = pd.read_csv('drugsComTrain.csv')  # Make sure to replace with your actual file path

# Define the `top_drugs_extractor` function
def top_drugs_extractor(condition):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

# Define the function for prediction
def predict_condition(review):
    # Transform the review using the loaded vectorizer
    review_vector = vectorizer.transform([review])
    # Predict the condition using the loaded model
    condition = model.predict(review_vector)[0]
    return condition

# Streamlit app UI
st.title("Medical Condition Prediction with Drug Recommendation System")

# Display image
image_path = "img.png"  # Replace with the path to your image
st.image(image_path, use_column_width=True)  # Display the image, scaled to the column width

# Input review text
review = st.text_area("Mention about your symptoms here:", height=200)

# Predict condition and recommend drugs on button click
if st.button("Predict Condition"):
    if review:
        # Predict condition
        predicted_condition = predict_condition(review)
        
        # Display the predicted condition with success style (green background, white bold text)
        st.success(f"### Prediction: {predicted_condition}")
        
        # Recommend drugs for the predicted condition
        recommended_drugs = top_drugs_extractor(predicted_condition)
        
        if recommended_drugs:
            # Create a comma-separated string of recommended drugs
            drug_list = ", ".join(recommended_drugs)
            
            # Highlight and style the drug recommendation section with a faded green background
            st.markdown(
                f"""
                <div style="background-color: #004d26; opacity:0.1 ,color: black; padding: 20px; border-radius: 10px;">
                    <h3 style="font-size: 26px; font-weight: bold; margin-bottom: 10px;">
                        Recommended Drugs:
                    </h3>
                    <p style="font-size: 24px; line-height: 1.8;">
                        {drug_list}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No specific drug recommendations available for this condition.")
    else:
        st.write("Please enter a review to get a prediction.")
