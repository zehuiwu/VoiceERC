import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_feature_distributions(df, features, thresholds, title, output_filename, num_classes):
    fig, axes = plt.subplots(4, 2, figsize=(20, 25))
    fig.suptitle(title, fontsize=16)
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        
        # Plot distribution
        sns.histplot(df[feature].replace(0, np.nan).dropna(), kde=True, ax=ax)
        
        ax.set_title(feature)
        
        # Add vertical lines for thresholds
        if num_classes == 3:
            thresholds_to_plot = ['low', 'medium_high']
        elif num_classes == 4:
            thresholds_to_plot = ['low', 'medium', 'high']
        elif num_classes == 5:
            thresholds_to_plot = ['very_low', 'low', 'medium', 'high']
        else:  # 6 classes
            thresholds_to_plot = ['very_low', 'low', 'medium_low', 'medium_high', 'high']
        
        colors = ['r', 'g', 'b', 'y', 'm']
        for threshold, color in zip(thresholds_to_plot, colors):
            if threshold in thresholds[feature]:
                ax.axvline(thresholds[feature][threshold], color=color, linestyle='--')
                ax.text(thresholds[feature][threshold], ax.get_ylim()[1], threshold,
                        ha='center', va='bottom', color=color, rotation=90)
        
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def extract_thresholds_and_stats(df, num_classes):
    features = ["duration", "avg_intensity", "intensity_variation", "avg_pitch", 
                "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
    thresholds = {}
    stats = {}
    
    # Calculate overall thresholds and stats
    overall_thresholds = {}
    overall_stats = {}
    for feature in features:
        feature_data = df[feature].replace(0, np.nan).dropna()
        if num_classes == 3:
            q1, q3 = feature_data.quantile([0.25, 0.75])
            overall_thresholds[feature] = {'low': q1, 'medium_high': q3}
        elif num_classes == 4:
            q1, q2, q3 = feature_data.quantile([0.25, 0.5, 0.75])
            overall_thresholds[feature] = {'low': q1, 'medium': q2, 'high': q3}
        elif num_classes == 5:
            q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
        else:  # 6 classes
            q1, q2, q3, q4, q5 = feature_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium_low': q3, 'medium_high': q4, 'high': q5}
        overall_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
    
    thresholds['overall'] = overall_thresholds
    stats['overall'] = overall_stats
    
    # Plot overall distribution
    plot_feature_distributions(df, features, overall_thresholds, 
                               "Distribution of Audio Features (Overall)", 
                               'feature_distributions_overall.png',
                               num_classes)
    
    # Calculate gender-specific thresholds, stats, and plot
    for gender in ['M', 'F']:
        gender_df = df[df['gender'] == gender]
        gender_thresholds = {}
        gender_stats = {}
        for feature in features:
            feature_data = gender_df[feature].replace(0, np.nan).dropna()
            if len(feature_data) > 0:
                if num_classes == 3:
                    q1, q3 = feature_data.quantile([0.25, 0.75])
                    gender_thresholds[feature] = {'low': q1, 'medium_high': q3}
                elif num_classes == 4:
                    q1, q2, q3 = feature_data.quantile([0.25, 0.5, 0.75])
                    gender_thresholds[feature] = {'low': q1, 'medium': q2, 'high': q3}
                elif num_classes == 5:
                    q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
                    gender_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
                else:  # 6 classes
                    q1, q2, q3, q4, q5 = feature_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                    gender_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium_low': q3, 'medium_high': q4, 'high': q5}
                gender_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
            else:
                gender_thresholds[feature] = overall_thresholds[feature]
                gender_stats[feature] = overall_stats[feature]
        thresholds[gender] = gender_thresholds
        stats[gender] = gender_stats
        
        # Plot gender-specific distribution
        plot_feature_distributions(gender_df, features, gender_thresholds, 
                                   f"Distribution of Audio Features (Gender: {gender})", 
                                   f'feature_distributions_gender_{gender}.png',
                                   num_classes)
    
    return thresholds, stats

def categorize(value, thresholds, num_classes):
    if pd.isna(value) or value == 0:
        return 'none'
    if num_classes == 3:
        if value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium_high']:
            return 'medium'
        else:
            return 'high'
    elif num_classes == 4:
        if value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium']:
            return 'medium_low'
        elif value <= thresholds['high']:
            return 'medium_high'
        else:
            return 'high'
    elif num_classes == 5:
        if value <= thresholds['very_low']:
            return 'very_low'
        elif value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium']:
            return 'medium'
        elif value <= thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    else:  # 6 classes
        if value <= thresholds['very_low']:
            return 'very_low'
        elif value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium_low']:
            return 'medium_low'
        elif value <= thresholds['medium_high']:
            return 'medium_high'
        elif value <= thresholds['high']:
            return 'high'
        else:
            return 'very_high'

def standardize_and_process_df(df, thresholds, stats, num_classes):
    features = list(thresholds['overall'].keys())
    
    # Standardize features
    for feature in features:
        df[f'{feature}_standardized'] = df.apply(lambda row: 
            (row[feature] - stats.get(row['gender'], stats['overall'])[feature]['mean']) / 
            stats.get(row['gender'], stats['overall'])[feature]['std']
        if not pd.isna(row[feature]) and row[feature] != 0 else np.nan, axis=1)
    
    # Categorize original features
    for feature in features:
        df[f'{feature}_category'] = df.apply(lambda row: categorize(
            row[feature], 
            thresholds.get(row['gender'], thresholds['overall'])[feature],
            num_classes
        ), axis=1)
    
    return df

def generate_concise_description(row, num_classes):
    features = ["avg_intensity", "intensity_variation", "avg_pitch", 
                "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
    descriptions = []

    # Mapping of categories to descriptive terms
    if num_classes == 3:
        category_terms = {
            'low': 'low', 'medium': 'moderate', 'high': 'high'
        }
    elif num_classes == 4:
        category_terms = {
            'low': 'low', 'medium_low': 'moderately low', 
            'medium_high': 'moderately high', 'high': 'high'
        }
    elif num_classes == 5:
        category_terms = {
            'very_low': 'very low', 'low': 'low', 'medium': 'moderate', 
            'high': 'high', 'very_high': 'very high'
        }
    else:  # 6 classes
        category_terms = {
            'very_low': 'extremely low', 'low': 'very low', 'medium_low': 'low',
            'medium_high': 'moderate', 'high': 'high', 'very_high': 'extremely high'
        }

    # Loudness and volume variation
    if row['avg_intensity_category'] != 'none':
        loudness = category_terms[row['avg_intensity_category']]
        variation = category_terms[row['intensity_variation_category']]
        descriptions.append(f"{loudness} volume with {variation} variation")

    # Pitch characteristics
    if row['avg_pitch_category'] != 'none':
        pitch = category_terms[row['avg_pitch_category']]
        if row['pitch_std_category'] != 'none' or row['pitch_range_category'] != 'none':
            pitch_var = category_terms[row['pitch_std_category']] if row['pitch_std_category'] != 'none' else category_terms[row['pitch_range_category']]
            descriptions.append(f"{pitch} pitch with {pitch_var} variation")
        else:
            descriptions.append(f"{pitch} pitch")

    # Speaking rate
    if row['articulation_rate_category'] != 'none':
        rate = category_terms[row['articulation_rate_category']]
        descriptions.append(f"{rate} speaking rate")

    # # Voice quality
    # if row['mean_hnr_category'] != 'none':
    #     hnr = category_terms[row['mean_hnr_category']]
    #     quality = 'clear and resonant' if hnr in ['medium', 'medium_high', 'high', 'very_high'] else 'rough or breathy'
    #     descriptions.append(f"{quality} voice quality")

    # Combine all parts into a concise description
    if descriptions:
        full_description = "Target speech characteristics: " + ", ".join(descriptions) + "."
    else:
        full_description = "Insufficient data to describe speech characteristics."

    return full_description


def generate_impression(row, num_classes):
    def get_level(category):
        if num_classes == 3:
            return ('medium', category)
        elif num_classes in [4, 5, 6]:
            if category in ['very_low']:
                return ('high', 'very low')
            elif category in ['low']:
                return ('medium', 'low')
            elif category in ['medium_low', 'medium', 'medium_high']:
                return ('low', 'medium')
            elif category in ['high']:
                return ('medium', 'high')
            else:  # very_high
                return ('high', 'very high')

    pitch_certainty, pitch = get_level(row['avg_pitch_category'])
    pitch_var_certainty, pitch_var = get_level(row['pitch_std_category'])
    volume_certainty, volume = get_level(row['avg_intensity_category'])
    volume_var_certainty, volume_var = get_level(row['intensity_variation_category'])
    rate_certainty, rate = get_level(row['articulation_rate_category'])

    def get_certainty_phrase(certainty):
        if certainty == 'high':
            return ""
        elif certainty == 'medium':
            return "likely "
        else:
            return "may "

    # Pitch impression
    if pitch in ['high', 'very high']:
        pitch_impression = f"{get_certainty_phrase(pitch_certainty)}uses a higher pitch"
    elif pitch in ['low', 'very low']:
        pitch_impression = f"{get_certainty_phrase(pitch_certainty)}uses a lower pitch"
    else:
        pitch_impression = "has a moderate pitch"

    if pitch_var in ['high', 'very high']:
        pitch_impression += f" with {get_certainty_phrase(pitch_var_certainty)}noticeable variation, suggesting expressiveness"
    elif pitch_var in ['low', 'very low']:
        pitch_impression += f" that {get_certainty_phrase(pitch_var_certainty)}remains steady, potentially indicating calmness or seriousness"
    else:
        pitch_impression += " with typical variation"

    # Volume impression
    if volume in ['high', 'very high']:
        volume_impression = f"{get_certainty_phrase(volume_certainty)}speaking loudly, which might indicate excitement, confidence, or urgency"
    elif volume in ['low', 'very low']:
        volume_impression = f"{get_certainty_phrase(volume_certainty)}speaking softly, possibly suggesting calmness, shyness, or caution"
    else:
        volume_impression = "using a moderate volume"

    if volume_var in ['high', 'very high']:
        volume_impression += f", with {get_certainty_phrase(volume_var_certainty)}significant volume changes"
    elif volume_var in ['low', 'very low']:
        volume_impression += f", with {get_certainty_phrase(volume_var_certainty)}little volume variation"
    else:
        volume_impression += ", with normal volume variation"

    # Speech rate impression
    if rate in ['high', 'very high']:
        rate_impression = f"{get_certainty_phrase(rate_certainty)}talking quickly, which could indicate excitement, urgency, or nervousness"
    elif rate in ['low', 'very low']:
        rate_impression = f"{get_certainty_phrase(rate_certainty)}talking slowly, possibly suggesting thoughtfulness, hesitation, or calmness"
    else:
        rate_impression = "speaking at a moderate pace"

    # Combine impressions into a single, flowing sentence
    impression = f"The target speaker {pitch_impression}, while {volume_impression}, and is {rate_impression}."
    
    return impression

def main():
    # Load the IEmocap dataset
    df = pd.read_csv('speech_features/iemocap_audio_features.csv')
    
    # Set the number of classes (3, 4, 5, or 6)
    num_classes = 5  # Change this to 3, 4, 5, or 6 for different categorizations
    
    # Extract thresholds and stats based on the training data
    train_df = df[df['mode'] == 'train']
    thresholds, stats = extract_thresholds_and_stats(train_df, num_classes)
    
    # Process the entire dataset
    processed_df = standardize_and_process_df(df, thresholds, stats, num_classes)
    
    # Generate descriptions and impressions
    processed_df['description'] = processed_df.apply(lambda row: generate_concise_description(row, num_classes), axis=1)
    processed_df['impression'] = processed_df.apply(lambda row: generate_impression(row, num_classes), axis=1)
    
    # List of original features to exclude
    original_features = ["avg_intensity", "intensity_variation", "avg_pitch", 
                         "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
    # Select columns to keep (excluding original features)
    columns_to_keep = [col for col in processed_df.columns if col not in original_features]
    
    # Create a new DataFrame with only the desired columns
    output_df = processed_df[columns_to_keep]

    # Save the processed data
    output_file = f'speech_features/processed_iemocap_audio_features_{num_classes}.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()