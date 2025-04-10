import os
import sys
import argparse
import json
import re

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Data Preparation')
    parser.add_argument('--base_path', type=str, default='corpora/bigc/bem', help='Base path')
    parser.add_argument('--output_dir', type=str, default='corpora', help='Output directory')
    parser.add_argument('--append', action='store_true', help='Append to existing files')
    args = parser.parse_args()
    return args

# main
def main(args):
    append = args.append

    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    wav_scp_dict = {}
    for audio_file in audio_files:
        wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
    # read the json file
    train_json = os.path.join(args.base_path, 'bem_eng.json')
    train_dict = {}
    with open(train_json, 'r') as f:
        dataset = json.load(f)
        for i, item in enumerate(dataset):
            audio_id = item["audio"]
            transcript = item["bem_transcript"]
            translation = item["eng_translation"]
            train_dict[audio_id] = {
                'audio_path': os.path.abspath(os.path.join(audio_path, audio_id)),
                'transcript': transcript,
                'translation': translation,
                'source_language': 'bem',
                'target_language': 'eng'
            }

    # create the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # if append, append to the existing train.json'
    if append:
        with open(os.path.join(output_dir, 'train.json'), 'r') as f:
            existing_data = json.load(f)
        print(f"found {len(existing_data)} existing entries in train.json")
        existing_data.update(train_dict)
        train_dict = existing_data
    # write the train.json file
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_dict, f, indent=4)
    print(f"wrote {len(train_dict)} entries in train.json")
    
    print("Data preparation complete")

if __name__ == '__main__':
    args = parse_args()
    main(args)