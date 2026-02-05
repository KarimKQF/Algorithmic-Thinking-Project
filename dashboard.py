import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import os

# Set page config
st.set_page_config(
    page_title="Credit Risk Control Tower",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Config ---
COLUMN_LABELS = {
    'person_age': 'Age',
    'person_income': 'Annual Income',
    'person_home_ownership': 'Home Ownership',
    'person_emp_length': 'Employment Length (Years)',
    'loan_intent': 'Loan Intent',
    'loan_grade': 'Loan Grade',
    'loan_amnt': 'Loan Amount',
    'loan_int_rate': 'Interest Rate',
    'loan_status': 'Loan Status (Default)',
    'loan_percent_income': 'Debt-to-Income Ratio',
    'cb_person_default_on_file': 'Historical Default',
    'cb_person_cred_hist_length': 'Credit History Length',
    'cb_person_default_on_file': 'Default on File'
}

def get_label(col):
    return COLUMN_LABELS.get(col, col)

# --- functions ---
@st.cache_data
def load_data():
    # Use relative path compatible with any machine
    path = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset(in).csv")
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, sep=';')
    
    # Cleaning: Drop duplicate columns if any exist
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Create a nice label column for plots
    df['Loan Status Label'] = df['loan_status'].map({0: 'Repaid', 1: 'Default'})
    return df

@st.cache_resource
def train_models(df):
    # Preprocessing
    df = df.copy()
    df.drop_duplicates(inplace=True)
    target_col = 'loan_status'
    
    # Separate Features and Target
    # Drop target AND the label we created for plotting (to avoid leakage/100% accuracy)
    drop_cols = [target_col]
    if 'Loan Status Label' in df.columns:
        drop_cols.append('Loan Status Label')
        
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Identify feature types (exclude target from exploration lists later)
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' is crucial for simulation inputs that might be new
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Process
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    models = {}
    
    # XGBoost
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, 
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_processed, y_train)
    models['XGBoost'] = xgb_model
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_processed, y_train)
    models['Logistic Regression'] = lr_model
    
    # MLP
    mlp_model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
    mlp_model.fit(X_train_processed, y_train)
    models['Neural Network'] = mlp_model
    
    return models, preprocessor, X_test_processed, y_test, num_cols, cat_cols

# --- Main App ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Exploratory Analysis (EDA)", "Model Performance", "Risk Simulator"])
    
    df = load_data()
    if df.empty:
        st.stop()
        
    if page == "Overview":
        st.title("🏦 Credit Risk Control Tower")
        st.markdown("### Loan Default Prediction & Analysis Dashboard")
        
        st.image("https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80", use_container_width=True)
        
        st.markdown("""
        This interactive dashboard presents the results of our Credit Risk Analysis project.
        
        **Objectives:**
        1.  Understand key risk factors.
        2.  Visualize data distributions and correlations.
        3.  Compare AI Model performance (XGBoost, Logistic Regression, MLP).
        4.  Simulate loan approval decisions in real-time.
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Borrowers", f"{len(df):,}")
        col2.metric("Global Default Rate", f"{df['loan_status'].mean():.2%}")
        col3.metric("Avg Loan Amount", f"${df['loan_amnt'].mean():,.0f}")
        
    elif page == "Exploratory Analysis (EDA)":
        st.title("📊 Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Categorical Analysis"])
        
        # Exclude target 'loan_status' and label from selection lists
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ['loan_status', 'Loan Status Label']]
        cat_cols_list = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in ['loan_status', 'Loan Status Label']]
        
        with tab1:
            col_dist = st.selectbox("Select Numeric Variable", numeric_cols, format_func=get_label)
            
            fig = px.histogram(df, x=col_dist, color="Loan Status Label", barmode="overlay", 
                               title=f"Distribution of {get_label(col_dist)} by Status",
                               labels={col_dist: get_label(col_dist), "Loan Status Label": "Status"},
                               color_discrete_map={'Repaid': 'green', 'Default': 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.write("Correlation Matrix (Numerical)")
            corr = df[numeric_cols + ['loan_status']].corr()
            # Rename index and columns for the heatmap
            corr.index = [get_label(c) for c in corr.index]
            corr.columns = [get_label(c) for c in corr.columns]
            
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            col_cat = st.selectbox("Select Categorical Variable", cat_cols_list, format_func=get_label)
            fig = px.histogram(df, x=col_cat, color="Loan Status Label", barmode="group", 
                               title=f"Count of {get_label(col_cat)} by Status",
                               labels={col_cat: get_label(col_cat), "Loan Status Label": "Status"},
                               color_discrete_map={'Repaid': 'green', 'Default': 'red'})
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Performance":
        st.title("🧠 AI Model Comparison")
        
        if 'models' not in st.session_state:
            with st.spinner('Training models... (XGBoost, Logistic Regression, MLP)'):
                models, preprocessor, X_test, y_test, _, _ = train_models(df)
                st.session_state['models'] = models
                st.session_state['data'] = (X_test, y_test)
                st.session_state['preprocessor'] = preprocessor
        
        models = st.session_state['models']
        X_test, y_test = st.session_state['data']
        
        perf_data = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            perf_data.append({"Model": name, "Accuracy": acc, "ROC AUC": auc})
            
        perf_df = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Performance Metrics")
            st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'))
            
            best_model = perf_df.loc[perf_df['ROC AUC'].idxmax()]
            st.success(f"🏆 Best Model: **{best_model['Model']}**")
            
        with col2:
            st.markdown("### ROC Curves")
            fig = go.Figure()
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            
            for name, model in models.items():
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={roc_auc_score(y_test, y_prob):.2f})", mode='lines'))
            
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Risk Simulator":
        st.title("🎲 Real-time Risk Simulator")
        st.markdown("Enter applicant details to estimate default probability using the **XGBoost** model.")
        
        if 'models' not in st.session_state:
            st.warning("Please visit the 'Model Performance' tab first to train the models.")
            st.stop()
            
        model = st.session_state['models']['XGBoost'] # Use best model
        preprocessor = st.session_state['preprocessor']
        
        with st.form("simulation_form"):
            st.subheader("Applicant Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 18, 100, 30)
                income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
                emp_len = st.number_input("Employment Length (Years)", 0.0, 50.0, 5.0)
                
            with col2:
                home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
                intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
                grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

            with col3:
                loan_amnt = st.number_input("Loan Amount ($)", 0, 100000, 10000)
                int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 10.0)
                default_hist = st.selectbox("Historical Default?", ["Y", "N"])
                cred_hist_len = st.number_input("Credit History Length (Years)", 0, 50, 5)
            
            submit = st.form_submit_button("Calculate Risk")
            
            if submit:
                # Create DF
                # Note: loan_percent_income must be calculated
                input_data = pd.DataFrame({
                    'person_age': [age],
                    'person_income': [income],
                    'person_home_ownership': [home],
                    'person_emp_length': [emp_len],
                    'loan_intent': [intent],
                    'loan_grade': [grade],
                    'loan_amnt': [loan_amnt],
                    'loan_int_rate': [int_rate],
                    'loan_status': [0], # Dummy
                    'loan_percent_income': [loan_amnt / income if income > 0 else 0],
                    'cb_person_default_on_file': [default_hist],
                    'cb_person_cred_hist_length': [cred_hist_len]
                })
                
                # Preprocess
                try:
                    processed_input = preprocessor.transform(input_data.drop(columns=['loan_status']))
                    prob = model.predict_proba(processed_input)[0][1]
                    
                    st.markdown("---")
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.metric("Default Probability", f"{prob:.1%}")
                    
                    with col_res2:
                        if prob < 0.2:
                            st.success("✅ Low Risk - Approved")
                        elif prob < 0.5:
                            st.warning("⚠️ Moderate Risk - Manual Review")
                        else:
                            st.error("❌ High Risk - Rejected")
                            
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()
