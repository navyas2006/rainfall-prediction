
import os                                      # Files aur folders ke paths handle karne ke liye
import sys                                     # Python system/path settings ke liye
import json                                    # JSON file create/save karne ke liye
import joblib                                  # Model, encoder aur scaler save/load karne ke liye
import numpy as np                             # Numerical operations ke liye

from sklearn.model_selection import train_test_split
# Dataset ko training aur testing parts mein divide karne ke liye

from sklearn.ensemble import RandomForestClassifier
# Random Forest classification model import kar rahe hain

from sklearn.preprocessing import StandardScaler
# Numerical features ko standard scale par convert karne ke liye

from sklearn.metrics import (
    accuracy_score,                            # Model ki accuracy calculate karne ke liye
    classification_report,                     # Precision, recall aur F1-score ke liye
    confusion_matrix,                          # Actual vs predicted values ka table banane ke liye
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Current Python file kis folder mein hai, uska complete path nikal raha hai

ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# SCRIPT_DIR ke parent folder ko project ka main/root folder maan rahe hain

sys.path.insert(0, SCRIPT_DIR)
# Python ko current script wale folder mein modules search karne ke liye bata rahe hain

import preprocess as pp
# Apni preprocess.py file ko import kar rahe hain
# pp iska short name hai


DATA_PATH = os.path.join(ROOT_DIR, "data", "rainfall_dataset.csv")
# Dataset file ka complete path bana rahe hain

MODELS_DIR = os.path.join(ROOT_DIR, "models")
# Trained model aur preprocessing files ko models folder mein save karenge


def main():
    # Program ka main training process is function ke andar hoga

    print("=" * 70)
    print("RAINFALL PREDICTION - MODEL TRAINING")
    print("=" * 70)
    # Program ka title screen par print kar rahe hain


    # 1. Dataset load karna

    if not os.path.exists(DATA_PATH):
        # Check kar rahe hain ki dataset file exist karti hai ya nahi

        print(f"ERROR: Dataset not found at {DATA_PATH}")
        # Agar dataset nahi mili to error message print karega

        sys.exit(1)
        # Error ki wajah se program ko stop kar dega


    print(f"\nLoading dataset from: {DATA_PATH}")
    # User ko bata raha hai ki dataset kis location se load ho rahi hai

    df = pp.load_raw_dataset(DATA_PATH)
    # preprocess.py ke function se CSV ko DataFrame mein load kar rahe hain


    print("\n--- DATASET INFO (RAW) ---")
    # Raw dataset ki information ka heading

    print("Shape (rows, columns):", df.shape)
    # Dataset mein kitni rows aur columns hain, ye batayega

    print("\nColumn names:")
    # Column names ka heading

    print(list(df.columns))
    # Dataset ke saare column names print karega

    print("\nData types:")
    # Data types ka heading

    print(df.dtypes)
    # Har column ka data type batayega

    print("\nMissing values per column:")
    # Missing values ka heading

    print(df.isnull().sum())
    # Har column mein kitni missing values hain, ye count karega


    # 2. Binary target banana aur data clean karna

    print("\n--- CLEANING DATA ---")
    # Cleaning process ka heading

    df = pp.create_binary_target(df)
    # rainfall column se rain_occurred target banayega
    # rainfall > 0 → 1
    # rainfall = 0 → 0

    print(
        f"Rows remaining after dropping missing "
        f"'{pp.RAW_TARGET_COLUMN}': {df.shape[0]}"
    )
    # Missing rainfall wali rows remove hone ke baad remaining rows batayega

    df = pp.clean_dataset(df)
    # Unwanted columns remove karega
    # Numerical missing values ko median se fill karega
    # Categorical missing values ko mode se fill karega

    print("Missing values filled (numerical -> median, categorical -> mode).")
    # Cleaning complete hone ka message


    print("\nTarget column distribution ('rain_occurred'):")
    # Target values ki distribution ka heading

    print(df[pp.TARGET_COLUMN].value_counts())
    # Target mein kitne 0 aur kitne 1 hain, ye count karega

    print("  1 = Rainfall occurred, 0 = No rainfall")
    # Target values ka meaning batayega


    # 3. Categorical encoding aur feature matrix banana

    print("\n--- ENCODING CATEGORICAL COLUMNS ---")
    # Encoding process ka heading

    encoders = {}
    # Empty dictionary bana rahe hain
    # Ismein trained LabelEncoder objects save honge

    X = pp.build_feature_matrix(df, encoders, fit=True)
    # Categorical columns ko numbers mein convert karega
    # Aur final feature matrix X banayega
    # fit=True ka matlab encoder categories ko learn karega

    y = df[pp.TARGET_COLUMN]
    # y target/output hai
    # Model ko rain_occurred predict karna hai

    print("Categorical columns encoded:", pp.CATEGORICAL_COLUMNS)
    # Kaunse categorical columns encode hue, ye print karega

    print("Final feature columns (in order):", pp.FEATURE_COLUMNS)
    # Model ko diye jaane wale features ki list print karega

    print("Feature matrix shape:", X.shape)
    # X mein kitni rows aur features hain, ye batayega


    # 4. Numerical features scale karna

    scaler = StandardScaler()
    # StandardScaler ka object bana rahe hain

    X_scaled = X.copy()
    # X ki copy bana rahe hain taaki original X safe rahe

    X_scaled[pp.NUMERICAL_COLUMNS] = scaler.fit_transform(
        X[pp.NUMERICAL_COLUMNS]
    )
    # Sirf numerical columns ko scale kar rahe hain
    # StandardScaler values ko approximately mean=0 aur standard deviation=1 ke scale par laata hai
    # Random Forest ke liye scaling required nahi hoti, lekin yahan apply ki gayi hai


    # 5. Train / test split

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # Dataset ko training aur testing data mein divide kar raha hai
    #
    # 80% → training
    # 20% → testing
    #
    # random_state=42 → same split ko reproduce karne ke liye
    # stratify=y → 0 aur 1 ka proportion maintain karne ke liye

    print(
        f"\nTrain set size: {X_train.shape[0]}   "
        f"Test set size: {X_test.shape[0]}"
    )
    # Training aur testing rows ki quantity print karega


    # 6. Random Forest model train karna

    print("\n--- TRAINING MODEL (RandomForestClassifier) ---")
    # Model training ka heading

    model = RandomForestClassifier(
        n_estimators=100,
        # Random Forest mein total 100 decision trees banenge

        max_depth=12,
        # Har decision tree ki maximum depth 12 hogi

        min_samples_leaf=5,
        # Har leaf mein minimum 5 samples hone chahiye

        n_jobs=-1,
        # Available CPU cores ko use karke training faster karne ki koshish karega

        random_state=42,
        # Random process ko fixed rakhta hai taaki results reproduce ho sakein

        class_weight="balanced",
        # Agar classes unequal hain to minority class ko relatively zyada importance dega
    )

    model.fit(X_train, y_train)
    # IMPORTANT: YAHAN ACTUAL MODEL TRAIN HO RAHA HAI
    # X_train = input features
    # y_train = correct target values

    print("Model training complete.")
    # Training complete hone ka message


    # 7. Model evaluate karna

    y_pred = model.predict(X_test)
    # Test data ko model mein dekar predictions generate kar rahe hain
    # 0 = No Rainfall
    # 1 = Rainfall

    accuracy = accuracy_score(y_test, y_pred)
    # Actual values aur predicted values compare karke accuracy calculate karta hai

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Rainfall", "Rainfall"]
    )
    # Precision, recall, F1-score aur support ki detailed report banata hai

    cm = confusion_matrix(y_test, y_pred)
    # Actual aur predicted values ka confusion matrix banata hai


    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    # Evaluation results ka heading print kar raha hai

    print(f"\nAccuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    # Accuracy ko decimal aur percentage dono format mein print karega

    print("\nClassification Report:")
    # Classification report ka heading

    print(report)
    # Classification report print karega

    print("Confusion Matrix:")
    # Confusion matrix ka heading

    print("                 Predicted No   Predicted Yes")
    # Confusion matrix ke predicted columns ke labels

    print(f"Actual No        {cm[0][0]:<14} {cm[0][1]}")
    # Actual No Rainfall wali row print karega
    # cm[0][0] = Actual 0, Predicted 0
    # cm[0][1] = Actual 0, Predicted 1

    print(f"Actual Yes       {cm[1][0]:<14} {cm[1][1]}")
    # Actual Rainfall wali row print karega
    # cm[1][0] = Actual 1, Predicted 0
    # cm[1][1] = Actual 1, Predicted 1


    # Feature importance

    importances = model.feature_importances_
    # Random Forest se har feature ki importance nikal rahe hain

    print("\nFeature importances:")
    # Feature importance ka heading

    for feat, imp in sorted(
        zip(pp.FEATURE_COLUMNS, importances),
        key=lambda x: -x[1]
    ):
        # Har feature ko uski importance ke saath pair kar rahe hain
        # Phir importance ke highest se lowest order mein sort kar rahe hain
        # lambda x: -x[1] importance ko descending order mein sort karta hai

        print(f"  {feat:15s} {imp:.4f}")
        # Feature ka naam aur importance value print kar raha hai


    # 8. Model aur preprocessing objects save karna

    os.makedirs(MODELS_DIR, exist_ok=True)
    # models folder create karega
    # Agar folder already exist karta hai to error nahi aayega

    joblib.dump(
        model,
        os.path.join(MODELS_DIR, "rainfall_model.pkl")
    )
    # Trained Random Forest model ko rainfall_model.pkl mein save kar raha hai

    joblib.dump(
        encoders,
        os.path.join(MODELS_DIR, "encoders.pkl")
    )
    # Trained LabelEncoders ko encoders.pkl mein save kar raha hai

    joblib.dump(
        scaler,
        os.path.join(MODELS_DIR, "scaler.pkl")
    )
    # Trained StandardScaler ko scaler.pkl mein save kar raha hai


    metadata = {
        "feature_columns": pp.FEATURE_COLUMNS,
        # Model mein use hone wale features

        "categorical_columns": pp.CATEGORICAL_COLUMNS,
        # Categorical columns ki list

        "numerical_columns": pp.NUMERICAL_COLUMNS,
        # Numerical columns ki list

        "target_column": pp.TARGET_COLUMN,
        # Target column ka naam

        "accuracy": accuracy,
        # Model ki accuracy save kar rahe hain

        "categorical_options": {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in pp.CATEGORICAL_COLUMNS
        },
        # Har categorical column ki unique values save kar raha hai
        # Example: season → Summer, Winter, Monsoon

        "numerical_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            for col in pp.NUMERICAL_COLUMNS
        },
        # Har numerical column ki minimum aur maximum value save kar raha hai
    }


    with open(
        os.path.join(MODELS_DIR, "metadata.json"),
        "w"
    ) as f:
        # metadata.json ko write mode mein open kar rahe hain

        json.dump(metadata, f, indent=2)
        # Metadata ko JSON format mein save kar rahe hain
        # indent=2 JSON ko readable format mein rakhta hai


    print(
        f"\nSaved model to:      "
        f"{os.path.join(MODELS_DIR, 'rainfall_model.pkl')}"
    )
    # Saved model ka path print karega

    print(
        f"Saved encoders to:   "
        f"{os.path.join(MODELS_DIR, 'encoders.pkl')}"
    )
    # Saved encoders ka path print karega

    print(
        f"Saved scaler to:     "
        f"{os.path.join(MODELS_DIR, 'scaler.pkl')}"
    )
    # Saved scaler ka path print karega

    print(
        f"Saved metadata to:   "
        f"{os.path.join(MODELS_DIR, 'metadata.json')}"
    )
    # Saved metadata file ka path print karega

    print(
        "\nDONE. You can now run the Flask app with:  python app.py"
    )
    # Training complete hone ka final message


if __name__ == "__main__":
    # Check karta hai ki ye Python file directly run hui hai ya kisi aur file se import hui hai

    main()
    # Agar directly run hui hai to main() function execute karega

