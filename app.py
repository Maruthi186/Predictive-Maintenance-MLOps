import streamlit as st
import pandas as pd
import joblib
import mlflow  
import re      
import os      

mlflow.set_tracking_uri("sqlite:///mlflow.db")


YOUR_RUN_ID = "bca558f391c744829bec5a0b62553545" 


@st.cache_resource
def load_model():
    try:
        model_uri = f"runs:/{YOUR_RUN_ID}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(YOUR_RUN_ID, "model_columns.joblib", ".")
        
        model_cols = joblib.load("model_columns.joblib")
        
        if os.path.exists("model_columns.joblib"):
            os.remove("model_columns.joblib")
            
        return model, model_cols
    
    except Exception as e:
        st.error(f"Error loading model from MLflow Run ID '{YOUR_RUN_ID}':")
        st.error(e)
        st.info("Have you pasted the correct Run ID? Are you running 'streamlit run app.py' from the same folder as your 'mlruns' directory?")
        return None, None

model, model_cols = load_model()


st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title('⚙️ Predictive Maintenance Dashboard')

if model:
    st.markdown(f"Using **Champion Model** from MLflow Run ID: `{YOUR_RUN_ID[:10]}...`")
    st.sidebar.header('Sensor Inputs')

    air_temp = st.sidebar.slider('Air temperature', 295.0, 305.0, 300.1, step=0.1)
    process_temp = st.sidebar.slider('Process temperature', 305.0, 315.0, 310.1, step=0.1)
    rpm = st.sidebar.slider('Rotational speed', 1150, 2900, 1538, step=1)
    torque = st.sidebar.slider('Torque', 3.0, 80.0, 39.9, step=0.1)
    tool_wear = st.sidebar.slider('Tool wear', 0, 270, 107, step=1)
    type_selection = st.sidebar.selectbox('Product Type', ('L', 'M', 'H'))

    
    input_data = {
        'Air_temperature': air_temp,
        'Process_temperature': process_temp,
        'Rotational_speed': rpm,
        'Torque': torque,
        'Tool_wear': tool_wear,
        'Type_L': 1 if type_selection == 'L' else 0,
        'Type_M': 1 if type_selection == 'M' else 0,
    }

    input_df = pd.DataFrame([input_data])
    
    try:
        input_df = input_df[model_cols]
    except KeyError as e:
        st.error(f"Column mismatch error: {e}")
    except TypeError:
        pass
    
    prediction = model.predict(input_df)[0]
    
    st.subheader('Model Prediction')
    if prediction == 0:
        st.success('✅ **System Status: NORMAL**')
    else:
        st.error('🚨 **System Status: FAILURE PREDICTED**')

    with st.expander("Show processed input features"):
        st.write(input_df)
else:
    st.warning("Model not loaded. Please check Run ID and terminal for errors.")