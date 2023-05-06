import streamlit as st
import joblib

# load the model and count vectorizer
model = joblib.load("model/spamq.joblib")
cv = joblib.load("model/spamcv.joblib")

# define the app layout
st.subheader("Hello I am :blue[Aashish]")


st.write("<h1 style='color:red'>Welcome TO</h1>", unsafe_allow_html=True)
st.title("Spam Detection App")

st.sidebar.header("Enter your message")
message = st.sidebar.text_input("Type your message here")

# make the prediction and display the result
if st.sidebar.button("Predict"):
    message = cv.transform([message])
    prediction = model.predict(message)[0]
    if prediction == 0:
        st.success("This message is not spam.")
    else:
        st.error("This message is spam.")     
st.balloons()

# set the style of the predict button
predict_button_style = """
    <style>
    #predict_button {
        background-color: #00FFFF;
        color: black;
    }
    </style>
"""
st.markdown(predict_button_style, unsafe_allow_html=True)

