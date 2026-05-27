# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Employee Income Prediction",

    page_icon="💼",

    layout="wide",

    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL & FILES
# ============================================================

model = joblib.load("lightgbm_model.pkl")

label_encoders = joblib.load("label_encoders.pkl")

df = pd.read_csv("dataset_cleaned.csv")

importance_df = pd.read_csv("feature_importance.csv")

# ============================================================
# LOAD BANNER
# ============================================================

banner = Image.open("banner.png")

# ============================================================
# DYNAMIC DROPDOWNS
# ============================================================

countries = sorted(df['native_country'].dropna().unique())

workclasses = sorted(df['workclass'].dropna().unique())

educations = sorted(df['education'].dropna().unique())

marital_statuses = sorted(df['marital_status'].dropna().unique())

occupations = sorted(df['occupation'].dropna().unique())

relationships = sorted(df['relationship'].dropna().unique())

races = sorted(df['race'].dropna().unique())

genders = sorted(df['sex'].dropna().unique())

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #FACC15;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.stButton>button {
    background: linear-gradient(to right, #FACC15, #F59E0B);
    color: black;
    border-radius: 12px;
    height: 3.5em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.02);
}

[data-testid="metric-container"] {
    text-align: center;
}

.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# DISPLAY BANNER
# ============================================================

st.image(banner, use_container_width=True)

# ============================================================
# TITLE
# ============================================================

st.title("💼 Employee Income Prediction using Machine Learning")

st.markdown("""

### Predict Whether Employee Income Exceeds 50K

✅ Final Model: LightGBM Classifier  
✅ Best Test Accuracy: 86.68%  
✅ Baseline Model  

Developed By: **Nakul Gupta**

""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Project Information")

st.sidebar.info("""

### ML Algorithms Used

- Logistic Regression
- Ridge Classifier
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- KNN
- SVM
- Voting Classifier
- Stacking Classifier
- Bagging Classifier

### Final Selected Model

🏆 LightGBM Classifier

### Why LightGBM?

- Best Overall Performance
- Excellent Generalization
- Fast Prediction Speed
- Strong Stability
- Ensemble Learning Power

""")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([

    "🔮 Prediction",

    "📊 Feature Importance",

    "📈 Analytics Dashboard",

    "📘 Project Insights"
])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    st.header("📝 Employee Information")

    col1, col2, col3 = st.columns(3)

    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:

        age = st.slider("Age", 18, 90, 30)

        workclass = st.selectbox("Workclass", workclasses)

        education = st.selectbox("Education", educations)

        marital_status = st.selectbox("Marital Status", marital_statuses)

    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        occupation = st.selectbox(
            "Occupation",
            occupations
        )

        relationship = st.selectbox(
            "Relationship",
            relationships
        )

        race = st.selectbox(
            "Race",
            races
        )

        sex = st.selectbox(
            "Gender",
            genders
        )

    # ========================================================
    # COLUMN 3
    # ========================================================

    with col3:

        hours_per_week = st.slider(
            "Hours per Week",
            1,
            100,
            40
        )

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            value=0
        )

        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            value=0
        )

        native_country = st.selectbox(
            "Native Country",
            countries
        )

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({

        'age': [age],

        'workclass': [workclass],

        'education': [education],

        'marital_status': [marital_status],

        'occupation': [occupation],

        'relationship': [relationship],

        'race': [race],

        'sex': [sex],

        'capital_gain': [capital_gain],

        'capital_loss': [capital_loss],

        'hours_per_week': [hours_per_week],

        'native_country': [native_country]
    })

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    categorical_columns = [

        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country'
    ]

    for column in categorical_columns:

        input_data[column] = label_encoders[column].transform(
            input_data[column]
        )

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button("🚀 Predict Income"):

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1]

        st.success("Prediction Completed Successfully!")

        # ====================================================
        # METRICS
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Prediction",
            ">50K" if prediction[0] == 1 else "<=50K"
        )

        metric2.metric(
            "Confidence",
            f"{probability:.2%}"
        )

        metric3.metric("Model Accuracy", "86.68%")

        st.markdown("")

        # ====================================================
        # FINAL RESULT CARD
        # ====================================================

        if prediction[0] == 1:

            st.success(
                "💰 Predicted Income: Greater than 50K"
            )

        else:

            st.error(
                "📉 Predicted Income: Less than or Equal to 50K"
            )

# ============================================================
# TAB 2 — FEATURE IMPORTANCE
# ============================================================

with tab2:

    st.header("📊 Feature Importance Analysis")

    top_features = importance_df.sort_values(by='Importance', ascending=True).head(12)

    fig = px.bar(top_features, x='Importance', y='Feature', orientation='h', text='Importance',title='Top Important Features')

    fig.update_traces(textposition='outside')

    fig.update_layout(template='plotly_dark', height=600, title_font_size=22, xaxis_title='Feature Importance Score', yaxis_title='Features')

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — ANALYTICS DASHBOARD
# ============================================================

with tab3:

    st.header("📈 Analytics Dashboard")

    # ========================================================
    # INCOME DISTRIBUTION
    # ========================================================

    income_counts = df['income'].value_counts()

    fig1 = px.pie(

        values=income_counts.values,

        names=income_counts.index,

        title="Income Distribution"
    )

    fig1.update_layout(
        template='plotly_dark'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ========================================================
    # AGE DISTRIBUTION
    # ========================================================

    fig2 = px.histogram(

        df,

        x='age',

        color='income',

        title='Age Distribution by Income'
    )

    fig2.update_layout(
        template='plotly_dark'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB 4 — PROJECT INSIGHTS
# ============================================================

with tab4:

    st.header("📘 Project Insights")

    # ========================================================
    # WORKFLOW
    # ========================================================

    with st.expander("📌 Machine Learning Workflow"):

        st.markdown("""

        - Data Cleaning
        - Missing Value Handling
        - Exploratory Data Analysis
        - Label Encoding
        - Baseline Model Building
        - Ensemble Learning
        - Model Evaluation
        - Model Deployment
        """)

    # ========================================================
    # MODELS USED
    # ========================================================

    with st.expander("🤖 Algorithms Compared"):

        st.markdown("""

        - Logistic Regression
        - Ridge Classifier
        - Lasso Regression
        - ElasticNet
        - Decision Tree
        - Random Forest
        - AdaBoost
        - Gradient Boosting
        - XGBoost
        - LightGBM
        - CatBoost
        - KNN
        - SVM
        - Voting Classifier
        - Stacking Classifier
        - Bagging Classifier
        """)

    # ========================================================
    # FINAL MODEL
    # ========================================================

    with st.expander("🏆 Final Model Selection"):

        st.markdown("""

        Final Selected Model:

        ✅ LightGBM Classifier

        Reasons:

        - Highest Test Accuracy
        - Excellent Generalization
        - Fast Prediction Speed
        - Stable Predictions
        - Strong Ensemble Learning
        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""

<div style='text-align: center;'>

### 💼 Employee Income Prediction using Machine Learning

Built with ❤️ using Streamlit | Developed by Nakul Gupta

</div>

""", unsafe_allow_html=True)