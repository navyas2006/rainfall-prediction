import pandas as pd                         # Pandas library import kar rahe hain, data/DataFrame handle karne ke liye
import numpy as np                          # NumPy import kar rahe hain, numerical operations ke liye

DROP_COLUMNS = ["date_of_record",           # Ye columns dataset se remove karne hain
                "station_name",             # Station ka naam model ke liye use nahi karenge
                "district"]                 # District column bhi remove karenge

RAW_TARGET_COLUMN = "rainfall"              # Original rainfall amount wala column
TARGET_COLUMN = "rain_occurred"             # Naya target column: rainfall hui = 1, nahi hui = 0

CATEGORICAL_COLUMNS = [                      # Ye categorical/string type ke columns hain
    "month",                                 # Month jaise January, February etc.
    "season",                                # Season jaise Summer, Winter etc.
    "state"                                  # State ka naam
]

NUMERICAL_COLUMNS = [                        # Ye saare numerical features hain
    "avg_temp",                              # Average temperature
    "min_temp",                              # Minimum temperature
    "max_temp",                              # Maximum temperature
    "wind_speed",                            # Wind ki speed
    "air_pressure",                          # Air pressure
    "elevation",                             # Location ki elevation
    "latitude",                              # Location ki latitude
    "longitude",                             # Location ki longitude
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
# Categorical + numerical columns ko combine karke saare input features bana rahe hain


def load_raw_dataset(csv_path):
    # Ye function CSV file ko load/read karne ke liye hai

    df = pd.read_csv(csv_path)
    # CSV file ko Pandas DataFrame mein read kar rahe hain

    return df
    # Loaded DataFrame ko function ke bahar return kar rahe hain


def create_binary_target(df):
    # Ye function rainfall ko binary target (0/1) mein convert karega

    df = df.copy()
    # Original DataFrame ko directly change na karke uski copy bana rahe hain

    df = df.dropna(subset=[RAW_TARGET_COLUMN])
    # Jahan rainfall ki value missing (NaN) hai, un rows ko remove kar rahe hain

    df[TARGET_COLUMN] = (df[RAW_TARGET_COLUMN] > 0).astype(int)
    # Agar rainfall > 0 hai to True → 1, warna False → 0
    # Isse rainfall ko "rain hui ya nahi" wale binary target mein convert kar rahe hain

    return df
    # Updated DataFrame return kar rahe hain


def clean_dataset(df):
    # Ye function dataset ko clean/preprocess karega

    df = df.copy()
    # Original DataFrame ko safe rakhne ke liye copy bana rahe hain

    for col in DROP_COLUMNS:
        # DROP_COLUMNS ki har column ko ek-ek karke check karenge

        if col in df.columns:
            # Check kar rahe hain ki ye column DataFrame mein actually exist karta hai ya nahi

            df = df.drop(columns=[col])
            # Agar column exist karta hai to usse dataset se remove kar denge


    for col in NUMERICAL_COLUMNS:
        # Har numerical column ko ek-ek karke process karenge

        if col in df.columns and df[col].isnull().any():
            # Check: column exist karta hai aur usme koi missing value hai ya nahi

            median_val = df[col].median()
            # Column ki median value calculate kar rahe hain

            df[col] = df[col].fillna(median_val)
            # Missing numerical values ko median se replace kar rahe hain


    for col in CATEGORICAL_COLUMNS:
        # Har categorical column ko ek-ek karke process karenge

        if col in df.columns and df[col].isnull().any():
            # Check kar rahe hain ki column exist karta hai aur usme missing value hai ya nahi

            mode_val = df[col].mode()[0]
            # Sabse frequently occurring value (mode) nikal rahe hain

            df[col] = df[col].fillna(mode_val)
            # Missing categorical values ko mode se replace kar rahe hain


    return df
    # Cleaned DataFrame ko return kar rahe hain


def encode_categoricals(df, encoders, fit=False):
    # Categorical/string values ko numerical values mein convert karne wala function

    from sklearn.preprocessing import LabelEncoder
    # Scikit-learn se LabelEncoder import kar rahe hain

    df = df.copy()
    # Original DataFrame ko modify na karne ke liye copy bana rahe hain

    for col in CATEGORICAL_COLUMNS:
        # Har categorical column ko ek-ek karke encode karenge

        if fit:
            # Agar fit=True hai, matlab training data par encoder banana hai

            le = LabelEncoder()
            # Ek naya LabelEncoder object create kar rahe hain

            df[col] = le.fit_transform(df[col].astype(str))
            # Column ki values ko string mein convert karke numbers mein encode kar rahe hain

            encoders[col] = le
            # Banaya hua encoder dictionary mein save kar rahe hain
            # Taaki baad mein test/new data par wahi mapping use ho


        else:
            # Agar fit=False hai, matlab existing encoder use karna hai

            le = encoders[col]
            # Pehle se trained/saved encoder dictionary se nikal rahe hain


            known_classes = set(le.classes_)
            # Training data mein jo categories already known hain unki list/set bana rahe hain

            safe_values = df[col].astype(str).apply(
                lambda v: v if v in known_classes else le.classes_[0]
            )
            # Agar new data mein koi unknown category milti hai,
            # to usko first known category se replace kar rahe hain
            # Isse LabelEncoder unknown-value error se bach sakta hai

            df[col] = le.transform(safe_values)
            # Safe categorical values ko trained encoder ke according numbers mein convert kar rahe hain


    return df
    # Encoded DataFrame return kar rahe hain


def build_feature_matrix(df, encoders, fit=False):
    # Ye function final ML input features (X) prepare karega

    df = encode_categoricals(df, encoders, fit=fit)
    # Pehle categorical columns ko numerical values mein encode kar rahe hain

    X = df[FEATURE_COLUMNS]
    # Dataset mein se sirf required feature columns select kar rahe hain

    return X
    # Final feature matrix X return kar rahe hain