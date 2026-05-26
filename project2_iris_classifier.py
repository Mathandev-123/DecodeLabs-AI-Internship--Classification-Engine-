import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def run_classification_pipeline():
print("=" * 50)
print("      DECODELABS AI INTERNSHIP - PROJECT 2       ")
print("   DATA CLASSIFICATION ENGINE (PYROID 3 READY)  ")
print("=" * 50)

# --------------------------------------------------  
# STEP 1: INPUT - Load and Understand Dataset  
# --------------------------------------------------  
print("\n[1/5] Loading Iris Benchmark Dataset...")  
iris = load_iris()  
X = iris.data  # Features: Sepal/Petal Length & Width  
y = iris.target  # Classes: Setosa, Versicolor, Virginica  
  
print(f"-> Total Samples Loaded: {len(X)} (Balanced)")  
print(f"-> Number of Target Classes: {len(np.unique(y))} {list(iris.target_names)}")  
print(f"-> Feature Dimensions: {X.shape[1]} (Sepal Length, Sepal Width, Petal Length, Petal Width)")  
  
# --------------------------------------------------  
# STEP 2: PROCESS - Shuffle & Split Data  
# --------------------------------------------------  
print("\n[2/5] Shuffling and Splitting Data (Train/Test Split)...")  
# Shuffling removes order bias before validation splitting  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.3, random_state=42, stratify=y  
)  
print(f"-> Training Set Size: {X_train.shape[0]} samples")  
print(f"-> Testing Set Size: {X_test.shape[0]} samples")  
  
# --------------------------------------------------  
# STEP 3: PROCESS - Feature Scaling  
# --------------------------------------------------  
print("\n[3/5] Applying Feature Scaling (StandardScaler)...")  
# Enforces Mean = 0 and Variance = 1 to remove bias  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  
print("-> Raw data balanced and normalized successfully.")  
  
# --------------------------------------------------  
# STEP 4: PROCESS - Train KNN Algorithm  
# --------------------------------------------------  
k_neighbors = 5  
print(f"\n[4/5] Training Supervised Learning Model (KNN Classifier, K={k_neighbors})...")  
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)  
knn_model.fit(X_train_scaled, y_train)  
print("-> Model training derived logic successfully from history.")  
  
# --------------------------------------------------  
# STEP 5: OUTPUT - Model Evaluation  
# --------------------------------------------------  
print("\n[5/5] Generating Evaluation Metrics...")  
y_pred = knn_model.predict(X_test_scaled)  
  
# Calculate Core Metrics  
macro_f1 = f1_score(y_test, y_pred, average='macro')  
cm = confusion_matrix(y_test, y_pred)  
  
print("-" * 40)  
print("                OUTPUT REPORT           ")  
print("-" * 40)  
print("\n■ CONFUSION MATRIX:")  
print(cm)  
print("\n■ GLOBAL F1-SCORE (Macro Target):")  
print(f"-> F1-Score: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")  
  
print("\n■ DETAILED CLASSIFICATION REPORT:")  
print(classification_report(y_test, y_pred, target_names=iris.target_names))  
print("=" * 50)  

# --------------------------------------------------  
# INTERACTIVE PREDICTION ENGINE FOR TESTS  
# --------------------------------------------------  
print("\n--- TEST YOUR EXPERIMENTAL DATA ---")  
print("Provide custom dimensions to make a live prediction:")  
try:  
    sepal_len = float(input("Enter Sepal Length (cm) [e.g., 5.1]: ") or 5.1)  
    sepal_wid = float(input("Enter Sepal Width (cm)  [e.g., 3.5]: ") or 3.5)  
    petal_len = float(input("Enter Petal Length (cm) [e.g., 1.4]: ") or 1.4)  
    petal_wid = float(input("Enter Petal Width (cm)  [e.g., 0.2]: ") or 0.2)  
      
    # Structure raw input list  
    custom_input = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])  
    # Pass through the exact same scaling layer  
    custom_input_scaled = scaler.transform(custom_input)  
      
    # Predict class  
    prediction = knn_model.predict(custom_input_scaled)[0]  
    predicted_class = iris.target_names[prediction]  
      
    print(f"\n[Result]: The AI categorized this flower as: **{predicted_class.upper()}**")  
except ValueError:  
    print("\n[Error]: Invalid numeric format entered. Skipping live test.")  
  
print("=" * 50)  
print("Portfolio project completed. Ready for DecodeLabs evaluation!")

if name == "main":
run_classification_pipeline().
