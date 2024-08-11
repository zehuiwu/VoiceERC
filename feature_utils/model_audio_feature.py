import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def construct_filename(df):
    if 'filename' not in df.columns:
        if 'Dialogue_ID' in df.columns and 'Utterance_ID' in df.columns:
            df['filename'] = 'dia' + df['Dialogue_ID'].astype(str) + '_' + 'utt' + df['Utterance_ID'].astype(str) + '.wav'
        else:
            raise KeyError("'Dialogue_ID' or 'Utterance_ID' columns not found")
    return df

def process_features(X, y, feature_columns):
    # Separate numeric and categorical columns
    numeric_columns = X[feature_columns].select_dtypes(include=[np.number]).columns
    categorical_columns = X[feature_columns].select_dtypes(exclude=[np.number]).columns
    
    # Handle numeric features
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy='mean')
        X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
    
    # Handle categorical features
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X[feature_columns])
    
    # Remove any remaining rows with NaN values
    mask = X_encoded.notna().all(axis=1)
    X_clean = X_encoded[mask]
    y_clean = y.loc[X_clean.index]
    
    print(f"Loaded {len(X_clean)} samples with {X_clean.shape[1]} features")
    print(f"Removed {len(X) - len(X_clean)} samples with NaN values")
    print(f"Emotion distribution:\n{y_clean.value_counts()}")
    
    return X_clean, y_clean

def load_and_prepare_meld_data(audio_features_file, emotion_labels_file, use_categorical=False):
    try:
        # Load the processed audio features
        audio_df = pd.read_csv(audio_features_file)
        
        # Load the emotion labels and construct filename if necessary
        emotion_df = pd.read_csv(emotion_labels_file)
        emotion_df = construct_filename(emotion_df)
        
        # Check if 'filename' column exists in both DataFrames
        if 'filename' not in audio_df.columns:
            raise KeyError(f"'filename' column not found in {audio_features_file}")
        
        # Check if 'Emotion' column exists in emotion_df
        if 'Emotion' not in emotion_df.columns:
            raise KeyError(f"'Emotion' column not found in {emotion_labels_file}")
        
        # Merge the dataframes on filename
        df = pd.merge(audio_df, emotion_df[['filename', 'Emotion']], on='filename', how='inner')
        
        if use_categorical:
            feature_columns = ['avg_intensity_category', 'intensity_variation_category', 'avg_pitch_category', 'pitch_std_category', 'articulation_rate_category']
        else:
            feature_columns = [col for col in df.columns if col.endswith('_standardized') and not col.startswith('duration')]
        
        if not feature_columns:
            raise ValueError(f"No suitable feature columns found in {audio_features_file}")
        
        X = df[feature_columns]
        y = df['Emotion']
        
        return process_features(X, y, feature_columns)
    
    except Exception as e:
        print(f"Error loading MELD data: {str(e)}")
        return None, None

def load_and_prepare_iemocap_data(iemocap_file, use_categorical=False):
    try:
        # Load the IEMOCAP dataset
        df = pd.read_csv(iemocap_file)
        df = df.drop(df[(df['emotion'] == 'xxx') | (df['emotion'] == 'oth') | (df['emotion'] == 'fea') | (df['emotion'] == 'dis') | (df['emotion'] == 'sur')].index).reset_index()
        
        if use_categorical:
            feature_columns = ['avg_intensity_category', 'intensity_variation_category', 'avg_pitch_category', 'pitch_std_category', 'articulation_rate_category']
        else:
            feature_columns = [col for col in df.columns if col.endswith('_standardized') and not col.startswith('duration')]
        
        if not feature_columns:
            raise ValueError(f"No suitable feature columns found in {iemocap_file}")
        
        # Split the data based on the 'mode' column
        train_df = df[df['mode'] == 'train']
        test_df = df[df['mode'] == 'test']
        
        # Prepare features and labels for train and test sets
        X_train = train_df[feature_columns]
        y_train = train_df['emotion']
        X_test = test_df[feature_columns]
        y_test = test_df['emotion']
        
        # Process features
        X_train, y_train = process_features(X_train, y_train, feature_columns)
        X_test, y_test = process_features(X_test, y_test, feature_columns)
        
        print("Train set:")
        print(f"Loaded {len(X_train)} samples with {X_train.shape[1]} features")
        print(f"Emotion distribution:\n{y_train.value_counts()}")
        print("\nTest set:")
        print(f"Loaded {len(X_test)} samples with {X_test.shape[1]} features")
        print(f"Emotion distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading IEMOCAP data: {str(e)}")
        return None, None, None, None
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, le):
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print(f"Cannot train {model_name} due to data loading error")
        return

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report with emotion labels
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()


def main():
    # Choose dataset
    dataset = 'IEMOCAP' # MELD/IEMOCAP
    use_categorical = False # "Use categorical features? (True/False): ")
    number_of_classes = 5

    if dataset == "MELD":
        # File paths for MELD
        train_audio_file = f'speech_features/meld_processed_{number_of_classes}class_train_audio_features.csv'
        train_emotion_file = 'data/MELD/train/train_sent_emo.csv'
        test_audio_file = f'speech_features/meld_processed_{number_of_classes}class_test_audio_features.csv'
        test_emotion_file = 'data/MELD/test/test_sent_emo.csv'

        # Load and prepare the data
        X_train, y_train = load_and_prepare_meld_data(train_audio_file, train_emotion_file, use_categorical)
        X_test, y_test = load_and_prepare_meld_data(test_audio_file, test_emotion_file, use_categorical)

        if X_train is None or X_test is None:
            print("Error loading data. Exiting.")
            return


    elif dataset == "IEMOCAP":
        # File path for IEMOCAP
        iemocap_file = f'speech_features/processed_iemocap_audio_features_{number_of_classes}.csv'

        # Load and prepare the data
        X_train, X_test, y_train, y_test = load_and_prepare_iemocap_data(iemocap_file, use_categorical)

        if X_train is None or X_test is None or y_train is None or y_test is None:
            print("Error loading data. Exiting.")
            return

    else:
        print("Invalid dataset choice. Exiting.")
        return

    # Encode the labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Define the models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate_model(model, X_train, X_test, y_train_encoded, y_test_encoded, model_name, le)

    # Feature importance for Random Forest
    rf_model = models['Random Forest']
    rf_model.fit(X_train, y_train_encoded)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title(f'The Most Important Features (Random Forest) - {dataset}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{dataset}_{"categorical" if use_categorical else "numerical"}.png')
    plt.close()

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()