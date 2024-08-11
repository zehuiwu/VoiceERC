import os
import pandas as pd
import parselmouth
from parselmouth.praat import call
from syllable_nuclei import speech_rate
from tqdm import tqdm
import argparse
import numpy as np

def extract_audio_features_with_praat(audio_path):
    # Load the audio file into a Sound object
    snd = parselmouth.Sound(audio_path)
    
    # Extract the total duration of the audio file in seconds
    duration = snd.get_total_duration()
    
    # Convert the sound to an intensity object with a time step of 100 ms
    intensity = call(snd, "To Intensity", 100, 0.0, True)
    # Calculate the average intensity over the entire duration using energy averaging
    avg_intensity = call(intensity, "Get mean", 0, 0, "energy")
    # Volume variation
    intensity_values = intensity.values
    volume_variation = np.std(intensity_values)
    
    # Convert the sound to a pitch object with a time step of 0.0 (auto), min pitch 75 Hz, max pitch 600 Hz
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    # Calculate the average pitch over the entire duration in Hertz
    avg_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    # Pitch variation
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
    if len(pitch_values) > 0:
        pitch_std = np.std(pitch_values)
        pitch_range = np.ptp(pitch_values)
    else:
        pitch_std = pitch_range = 0
    
    try:
        articulation_rate = speech_rate(audio_path)['articulation rate(nsyll / phonationtime)']
    except:
        articulation_rate = None
    
    # Voice quality (using harmonics-to-noise ratio as a proxy)
    harmonicity = snd.to_harmonicity()
    mean_hnr = call(harmonicity, "Get mean", 0, 0)
    
    
    return {
        'filename': os.path.basename(audio_path),
        "duration": duration,
        "avg_intensity": avg_intensity,
        "intensity_variation": volume_variation,
        "avg_pitch": avg_pitch,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "articulation_rate": articulation_rate,
        "mean_hnr": mean_hnr,
    }


# Extract audio features for MELD dataset
def process_audio_folder(folder_path, output_csv):
    # List to store results
    results = []

    # Get list of audio files
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac'))]

    # Use tqdm to create a progress bar
    for filename in tqdm(audio_files, desc=f"Processing {os.path.basename(folder_path)} files"):
        file_path = os.path.join(folder_path, filename)
        try:
            results.append(extract_audio_features_with_praat(file_path))
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")

    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return df


# Extract audio features for IEMOCAP dataset
def process_csv(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Select only the required columns
    df = df[['video_id', 'segment_id', 'emotion', 'path', 'gender', 'mode','text']]
    
    # Create empty lists to store the extracted features
    features = []
    
    # Iterate through each row in the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        audio_path = 'data/IEMOCAP_full_release/' + row['path']
        
        # Extract features
        try:
            audio_features = extract_audio_features_with_praat(audio_path)
            features.append(audio_features)
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            features.append({})  # Add an empty dict if there's an error
    
    # Create a new dataframe with the extracted features
    features_df = pd.DataFrame(features)
    
    # Combine the selected columns from the original dataframe with the new features
    result_df = pd.concat([df.drop('path', axis=1), features_df.drop('filename', axis=1)], axis=1)
    
    # Write the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Results written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process audio files from a specified folder.")
    parser.add_argument("--dataset", choices=["meld", "iemocap"], help="Dataset to process (MELD or IEMOCAP)")
    parser.add_argument("--folder", choices=["dev", "train", "test"], help="MELD Folder to process (dev, train, or test)")
    args = parser.parse_args()

    if args.dataset == "meld":
        # Construct the full path to the chosen folder
        folder_path = f'data/MELD/{args.folder}/{args.folder}_audio'
        
        # Ensure the folder exists
        if not os.path.isdir(folder_path):
            print(f"Error: The folder {folder_path} does not exist.")
            return

        # Construct the output CSV filename
        output_csv = f"speech_features/meld_{args.folder}_audio_features.csv"

        # Process the chosen folder
        results_df = process_audio_folder(folder_path, output_csv)

    elif args.dataset == "iemocap":
        input_file = 'data/IEMOCAP_full_release/iemocap_full_dataset.csv'
        output_file = 'speech_features/iemocap_audio_features.csv'
        process_csv(input_file, output_file)

if __name__ == "__main__":
    main()