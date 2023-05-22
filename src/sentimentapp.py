import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import base64






# Functions
def main():
    st.title("Sentiment Analysis App")
    st.subheader("Reformation Team Project")

    

    st.image("senti.jpg")

    # Define the available models
    models = {
        "ROBERTA": "Adoley/covid-tweets-sentiment-analysis-roberta-model",
        "BERT": "Adoley/covid-tweets-sentiment-analysis",
        "DISTILBERT": "Adoley/covid-tweets-sentiment-analysis-distilbert-model"
    }

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    how_to_use = """
## How to Use

1. Enter your text in the input box.
2. Click the **Analyze Sentiment** button.
3. Wait for the app to process the text and display the sentiment analysis results.
4. Explore the sentiment scores and visualization provided.
"""

    # Add the "How to Use" message to the sidebar
    st.sidebar.markdown(how_to_use)

    if choice == "Home":
        st.subheader("Home")

        # Add a dropdown menu to select the model
        model_name = st.selectbox("Select a model", list(models.keys()))

        with st.form(key="nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

       

        col1, col2 = st.columns(2)
        if submit_button:
            # Display sound-effect
            st.info("🔮 Abracadabra! Your report has been submitted!")
            sound_file = 'C:/Users/viole/OneDrive/Documents/streamlit2/swipe-swoosh.mp3'
            st.audio(sound_file, format='audio/wav')

            with col1:
                st.info("Results")
                tokenizer = AutoTokenizer.from_pretrained(models[model_name])
                model = AutoModelForSequenceClassification.from_pretrained(models[model_name])


                # Tokenize the input text
                inputs = tokenizer(raw_text, return_tensors="pt")

                # Make a forward pass through the model
                outputs = model(**inputs)

                # Get the predicted class and associated score
                predicted_class = outputs.logits.argmax().item()
                score = outputs.logits.softmax(dim=1)[0][predicted_class].item()

                # Compute the scores for all sentiments
                positive_score = outputs.logits.softmax(dim=1)[0][2].item()
                negative_score = outputs.logits.softmax(dim=1)[0][0].item()
                neutral_score = outputs.logits.softmax(dim=1)[0][1].item()

                # Compute the confidence level
                confidence_level = np.max(outputs.logits.detach().numpy())

                # Print the predicted class and associated score
                st.write(f"Predicted class: {predicted_class}, Score: {score:.3f}, Confidence Level: {confidence_level:.2f}")

                # Emoji
                if predicted_class == 2:
                    st.markdown("Sentiment: Positive :smiley:")
                    st.image("positive-smiley-face.png")
                elif predicted_class == 1:
                    st.markdown("Sentiment: Neutral :😐:")
                    st.image("neutral-smiley-face.png")
                else:
                    st.markdown("Sentiment: Negative :angry:")
                    st.image("negative-smiley-face.png")

            

            results_df = pd.DataFrame(columns=["Sentiment Class", "Score"])

            # Create a DataFrame with scores for all sentiments
            all_scores_df = pd.DataFrame({
            'Sentiment Class': ['Positive', 'Negative', 'Neutral'],
            'Score': [positive_score, negative_score, neutral_score]
            })

            # Concatenate the two DataFrames

            results_df = pd.concat([results_df, all_scores_df], ignore_index=True)

           

            # Create the Altair chart
            chart = alt.Chart(results_df).mark_bar(width=50).encode(
                x="Sentiment Class",
                y="Score",
                color="Sentiment Class"
            )

            # Display the chart
            with col2:
                st.altair_chart(chart, use_container_width=True)
                st.write(results_df)

    else:
        st.subheader("About")
        st.write("This marvelous sentiment analysis NLP app, crafted with love by the brilliant minds of Team Reformation, dives into the realm of Covid-19 tweets. Armed with a pre-trained model, it fearlessly predicts the sentiment lurking within the depths of your text. Brace yourself for an adventure of teamwork and collaboration, as we embark on a quest to unravel the sentiments that dwell within the tweetsphere!")



if __name__ == "__main__":
    main()
