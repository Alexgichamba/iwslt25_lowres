import os
import sys
import argparse
import json
import re

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='BembaSpeech Data Preparation')
    parser.add_argument('--base_path', type=str, default='corpora/bigc/bem', help='Base path')
    parser.add_argument('--output_dir', type=str, default='corpora', help='Output directory')
    parser.add_argument('--append', action='store_true', help='Append to existing files')
    args = parser.parse_args()
    return args

# main
def main(args):

    append = args.append

    # find the audio files and make a wav.scp where the key is the filename and the value is the absolute path to the audio file
    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    wav_scp_dict = {}
    for audio_file in audio_files:
        wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
        
    splits = ['train', 'valid', 'test']
    
    # make a dict of dicts for the text files
    # each entry will be indexed by audio id and will have fields audio_path, transcript, translation
    all_text_dict = {}
    all_text_dict['train'] = {}
    all_text_dict['valid'] = {}
    all_text_dict['test'] = {}
    for split in splits:
        split_json = os.path.join(f"{args.base_path}/splits", f"{split}.jsonl")
        with open(split_json, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # remove any \n literal in the json line
                line = re.sub(r'\\n', '', line)
                data = json.loads(line)
                audio_file = data['audio_id']
                transcript = data['bem_transcription'].strip()
                translation = data['en_translation'].strip()

                # skip over erroneous entries
                if not transcript or not translation:
                    print(f"Skipping {audio_file} due to missing transcript or translation")
                    continue
                elif audio_file not in wav_scp_dict:
                    print(f"Skipping {audio_file} due to missing audio file")
                    continue
                elif transcript == '' or translation == '':
                    print(f"Skipping {audio_file} due to empty transcript or translation")
                    continue
                elif transcript == '.' or translation == '.':
                    print(f"Skipping {audio_file} due to transcript or translation being a single period")
                    continue
                elif transcript == "NOT PLAYING" or translation == "NOT PLAYING":
                    print(f"Skipping {audio_file} due to transcript or translation being 'NOT PLAYING'")
                    continue
                elif transcript == "NOT FOUND" or translation == "NOT FOUND":
                    print(f"Skipping {audio_file} due to transcript or translation being 'NOT FOUND'")
                    continue
                
                all_text_dict[split][audio_file] = {
                    'audio_path': os.path.abspath(os.path.join(audio_path, audio_file)),
                    'transcript': transcript,
                    'translation': translation,
                    'source_language': 'bem',
                    'target_language': 'eng'
                    
                }
    
    # create the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # write json files for each split in the output dir
    for split in splits:
        output_json = os.path.join(output_dir, f"{split}.json")
        if append and os.path.exists(output_json):
            with open(output_json, 'r') as f:
                existing_data = json.load(f)
                all_text_dict[split].update(existing_data)
                print(f"Found {len(existing_data)} existing entries in {output_json}")
        with open(output_json, 'w') as f:
            json.dump(all_text_dict[split], f, indent=4)
        print(f"Wrote {len(all_text_dict[split])} entries in {output_json}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Data preparation complete.")