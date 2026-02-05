import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
import warnings
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration & Styling ---
class Config:
    # Use absolute path and handle potential issues
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILENAME = "credit_risk_dataset(in).csv"
    DATA_PATH = os.path.join(BASE_DIR, "data", "credit_risk_dataset(in).csv")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    
    # Stylistic elements
    COLOR_PALETTE = "viridis"
    SNS_STYLE = "whitegrid"
    RANDOM_SEED = 42

# Ensure output directories exist
os.makedirs(Config.PLOTS_DIR, exist_ok=True)

# Setup Rich Console
console = Console()

def print_header(text: str):
    console.print(Panel(Text(text, justify="center", style="bold magenta"), expand=False))

def print_step(text: str):
    console.print(f"\n[bold cyan]➜ {text}[/bold cyan]")

def setup_plotting():
    """Configures seaborn and matplotlib for high-quality outputs."""
    sns.set_theme(style=Config.SNS_STYLE, context="notebook", font_scale=1.1)
    
    # Custom aesthetic tweaks
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.dpi"] = 300  # High DPI for publication quality
    plt.rcParams["savefig.bbox"] = "tight"
    
    # Set palette
    try:
        sns.set_palette(Config.COLOR_PALETTE)
    except:
        pass

# --- Data Loading ---
def load_data(filepath: str) -> pd.DataFrame:
    """Loads data with error handling and rich feedback."""
    print_step(f"Loading data from [green]{filepath}[/green]...")
    
    if not os.path.exists(filepath):
        console.print(f"[bold red]✗ File not found:[/bold red] {filepath}")
        # Debugging info
        console.print(f"Current Working Directory: {os.getcwd()}")
        console.print(f"Directory Contents: {os.listdir(os.path.dirname(filepath) if os.path.dirname(filepath) else '.')}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Detected delimiter as ';' from inspection
        df = pd.read_csv(filepath, sep=';')
        console.print(f"[green]✓ Success![/green] Loaded {len(df):,} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        console.print(f"[bold red]✗ Error loading data:[/bold red] {e}")
        raise

def display_first_rows(df: pd.DataFrame, n: int = 5):
    """Displays the first n rows in a rich table."""
    print_step(f"Previewing first {n} rows:")
    table = Table(show_header=True, header_style="bold magenta", title="Data Preview", border_style="cyan")
    
    # Add columns
    for col in df.columns:
        table.add_column(col, overflow="fold", no_wrap=True)
    
    # Add rows
    for index, row in df.head(n).iterrows():
        table.add_row(*[str(x) for x in row])
    
    console.print(table)

# --- EDA ---
class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def run_all(self):
        print_step("Starting Exploratory Data Analysis (EDA)...")
        self.plot_distributions()
        self.plot_correlation()
        self.plot_categorical_counts()
        console.print(f"\n[green]✓ EDA Completed![/green] Plots saved in [bold]{Config.PLOTS_DIR}[/bold]")

    def plot_distributions(self):
        """Plots distributions for numerical features."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in track(num_cols, description="Plotting distributions..."):
            plt.figure()
            sns.histplot(data=self.df, x=col, kde=True, color="teal")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.savefig(os.path.join(Config.PLOTS_DIR, f"dist_{col}.png"))
            plt.close()

    def plot_categorical_counts(self):
        """Plots counts for categorical features."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in track(cat_cols, description="Plotting categorical counts..."):
            plt.figure()
            sns.countplot(data=self.df, x=col, palette="viridis", order=self.df[col].value_counts().index)
            plt.title(f"Count of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(Config.PLOTS_DIR, f"count_{col}.png"))
            plt.close()

    def plot_correlation(self):
        """Plots correlation matrix heatmap."""
        console.print("Generating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        num_df = self.df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title("Correlation Matrix", fontsize=18)
        plt.savefig(os.path.join(Config.PLOTS_DIR, "correlation_heatmap.png"))
        plt.close()


# --- Preprocessing & Modeling ---
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import xgboost as xgb

class ModelPipeline:
    def __init__(self, df: pd.DataFrame, target_col: str = 'loan_status'):
        self.df = df
        self.target_col = target_col
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.preprocessor = None

    def preprocess(self):
        print_step("Preprocessing data...")
        
        # dropping duplicates
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        if len(self.df) < initial_len:
            console.print(f"[yellow]Removed {initial_len - len(self.df)} duplicate rows.[/yellow]")

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Identify numerical and categorical columns
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        print_step(f"Features detected: [cyan]{len(num_cols)} Numerical[/cyan], [magenta]{len(cat_cols)} Categorical[/magenta]")

        # Create preprocessing pipelines
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        # Split data
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=y
        )
        
        # Fit and transform
        console.print("Fitting preprocessor...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Extract feature names
        try:
            num_names = list(num_cols)
            cat_names = list(self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols))
            self.feature_names = num_names + cat_names
        except:
            self.feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]
            
        return X_train_processed, y_train

    def train_all(self, X_train, y_train):
        print_step("Training all models...")
        
        # 1. XGBoost (Tree/Forest Model)
        print_step("Training XGBoost...")
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        self.models['XGBoost'] = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,
            eval_metric='auc'
        )
        self.models['XGBoost'].fit(X_train, y_train)
        
        # 2. Logistic Regression (Baselines)
        print_step("Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=Config.RANDOM_SEED
        )
        self.models['Logistic Regression'].fit(X_train, y_train)
        
        # 3. MLP Classifier (Deep Learning)
        print_step("Training Deep Learning (MLP)...")
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=Config.RANDOM_SEED
        )
        self.models['Neural Network'].fit(X_train, y_train)
        
        console.print(f"[green]✓ All models trained![/green]")

    def evaluate_all(self):
        print_step("Evaluating Models and Comparing Performance...")
        
        summary_table = Table(title="Model Comparison", border_style="green")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Accuracy", style="magenta")
        summary_table.add_column("ROC-AUC", style="yellow")
        summary_table.add_column("Recall (Class 1)", style="blue")
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test_processed)
            y_prob = model.predict_proba(self.X_test_processed)[:, 1]
            
            acc = accuracy_score(self.y_test, y_pred)
            roc = roc_auc_score(self.y_test, y_prob)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            recall_1 = report['1']['recall']
            
            self.results[name] = {'Accuracy': acc, 'ROC-AUC': roc}
            summary_table.add_row(name, f"{acc:.4f}", f"{roc:.4f}", f"{recall_1:.4f}")
            
            # ROC Curve Plotting
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc:.2f})')
            
            # Save individual confusion matrices
            self.plot_confusion_matrix(self.y_test, y_pred, name)

        console.print(summary_table)
        
        # Finalize ROC Plot
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(Config.PLOTS_DIR, "roc_curve_comparison.png"))
        plt.close()
        
        self.plot_feature_importance()
        self.plot_single_tree()

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        plt.figure()
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(Config.PLOTS_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
        plt.close()

    def plot_feature_importance(self):
        # Only for XGBoost
        if 'XGBoost' in self.models:
            try:
                importances = self.models['XGBoost'].feature_importances_
                indices = np.argsort(importances)[::-1][:20]
                
                plt.figure(figsize=(10, 8))
                # Use feature names if available
                feat_labels = [self.feature_names[i] for i in indices] if self.feature_names else indices
                
                sns.barplot(x=importances[indices], y=feat_labels, palette="viridis")
                plt.title("Top 20 Feature Importances (XGBoost)")
                plt.xlabel("Importance")
                plt.savefig(os.path.join(Config.PLOTS_DIR, "feature_importance_xgboost.png"))
                plt.close()
            except Exception as e:
                console.print(f"[yellow]Could not plot feature importance: {e}[/yellow]")

    def plot_single_tree(self):
        # Visualize the first tree of XGBoost
        if 'XGBoost' in self.models:
            try:
                plt.figure(figsize=(20, 15), dpi=300)
                xgb.plot_tree(self.models['XGBoost'], num_trees=0, rankdir='LR')
                plt.title("XGBoost - First Decision Tree Visualization")
                plt.savefig(os.path.join(Config.PLOTS_DIR, "xgboost_tree_viz.png"), dpi=600)
                plt.close()
                console.print("[green]✓ Decision Tree visualization saved![/green]")
            except Exception as e:
                console.print(f"[yellow]Could not visualize tree (graphviz might be missing): {e}[/yellow]")

# --- Main Execution ---
def main():
    print_header("Credit Risk Analysis - PROJET GROUPE NEOMA")
    setup_plotting()
    
    try:
        df = load_data(Config.DATA_PATH)
        display_first_rows(df)
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            console.print("\n[yellow]! Missing values detected:[/yellow]")
            print_step("Missing Values Summary:")
            table = Table(title="Missing Values", border_style="yellow")
            table.add_column("Column", style="cyan")
            table.add_column("Missing Count", style="magenta")
            table.add_column("Percentage", style="green")
            
            for col, count in missing[missing > 0].items():
                table.add_row(col, str(count), f"{(count/len(df))*100:.2f}%")
            console.print(table)
        else:
            console.print("\n[green]✓ No missing values detected.[/green]")

        # Run EDA? 
        # eda = EDA(df)
        # eda.run_all()

        # Modeling
        if 'loan_status' in df.columns:
            pipeline = ModelPipeline(df, target_col='loan_status')
            X_train, y_train = pipeline.preprocess()
            pipeline.train_all(X_train, y_train)
            pipeline.evaluate_all()
        else:
            console.print("[red]Target column 'loan_status' not found! Cannot train model.[/red]")

    except Exception as e:
        console.print(f"[bold red]Critical Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()
