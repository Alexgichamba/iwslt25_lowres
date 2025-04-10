import os
import sys
import argparse
import pandas as pd
import json

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='FFSTC Data Preparation')
    parser.add_argument('--base_path', type=str, default='mymy', help='Base path')
    parser.add_argument('--output_dir', type=str, default='corpora', help='Output directory')
    parser.add_argument('--use_synth', action='store_true', help='Use synthetic data')
    parser.add_argument('--append', action='store_true', help='Append to existing files')
    args = parser.parse_args()
    return args

# main
def main(args):
    append = args.append
    use_synth = args.use_synth
    for split in ['train', 'valid']:
        audio_path = os.path.join(args.base_path, split)
        audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
        wav_scp_dict = {}
        for audio_file in audio_files:
            wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
        # read the csv file
        if use_synth:
            csv = os.path.join(args.base_path, f'{split}_synth.csv')
        else:
            csv = os.path.join(args.base_path, f'{split}.csv')
        data_dict = {}
        df = pd.read_csv(csv)
        # csv structure is ID,utterance,filename,duration
        for index, row in df.iterrows():
            audio_file = row['filename']
            if not use_synth:
                text = row['utterance'].strip()
                data_dict[audio_file] = {
                    'audio_path': os.path.abspath(os.path.join(audio_path, audio_file)),
                    'transcript': text,
                    'translation': None,
                    'source_language': 'fon',
                    'target_language': 'fra'
                }
            else:
                transcript = row['transcript'].strip()
                translation = row['translation'].strip()
                data_dict[audio_file] = {
                    'audio_path': os.path.abspath(os.path.join(audio_path, audio_file)),
                    'transcript': transcript,
                    'translation': translation,
                    'source_language': 'fon',
                    'target_language': 'fra'
                }
    
        # create the output directory if it does not exist
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        json_file = os.path.join(output_dir, f'{split}.json')
        if append:
            with open(json_file, 'r') as f:
                existing_data = json.load(f)
            print(f"found {len(existing_data)} existing entries in {json_file}")
            existing_data.update(data_dict)
            data_dict = existing_data
        # write the json file
        with open(json_file, 'w') as f:
            json.dump(data_dict, f, indent=4)
        print(f"wrote {len(data_dict)} entries in {json_file}")
        
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Data preparation complete")