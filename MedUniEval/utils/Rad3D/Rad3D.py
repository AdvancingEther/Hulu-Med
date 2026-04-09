import os
import json
import numpy as np
from tqdm import tqdm
import re
from ..base_dataset import BaseDataset
from ..eval_3d import evaluate_3drad


class Rad3D(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "./Rad3D_test.json"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
        self.num_slices = int(os.environ.get("RAD3D_NUM_SLICES", 64))

    def load_data(self):
        json_path = self.dataset_path
        
        print(f"Loading 3D-RAD data from: {json_path}")
        
        with open(json_path, "r") as f:
            all_samples = json.load(f)
        
        print(f"Total samples in dataset: {len(all_samples)}")
        
        chunked_samples = np.array_split(all_samples, self.num_chunks)[self.chunk_idx].tolist()
        
        print(f"Chunk {self.chunk_idx}/{self.num_chunks}: Processing {len(chunked_samples)} samples")
        
        for idx, sample in enumerate(tqdm(chunked_samples, desc=f"Loading chunk {self.chunk_idx}")):
            try:
                processed_sample = self.construct_messages(sample)
                if processed_sample:
                    self.samples.append(processed_sample)
            except Exception as e:
                sample_id = sample.get("id", f"sample_{idx}") if isinstance(sample, dict) else f"sample_{idx}"
                print(f"Error processing sample {sample_id}: {type(e).__name__}: {e}")
                continue
        
        print(f"Chunk {self.chunk_idx}: Successfully loaded {len(self.samples)} samples")
        return self.samples

    def construct_messages(self, sample):
        try:
            question = sample["conversations"][0]["value"]
            question_type = sample.get("Question_Type", "open")
        except (KeyError, IndexError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None
        
        npy_path = sample.get("npy_path")

        if not npy_path:
            print(f"Warning: npy_path not found in sample: {sample.get('id', 'unknown')}")
            return None
        if not os.path.isfile(npy_path):
            print(f"Warning: npy file not found: {npy_path}")
            return None

        question_clean = question.replace('<image>\n', '').replace('<video>\n', '').strip()
        
        if question_type.lower() in ['close', 'closed']:
            prompt = question_clean + "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = question_clean + "\nPlease answer the question concisely."
        
        messages = {
            "prompt": prompt,
            "npy_path": npy_path,
            "num_slices": self.num_slices,
        }
        sample["messages"] = messages
        
        return sample

    def extract_choice_letter(self, text):
        text = str(text).lower().strip()
        
        match = re.match(r'^([a-d])[.\):]?\s*', text)
        if match:
            return match.group(1)
        
        if text in ['a', 'b', 'c', 'd']:
            return text
        
        return text

    def cal_metrics(self, out_samples):
        print("\n" + "="*80)
        print("Starting 3D-RAD Evaluation")
        print("="*80)
        
        TYPE_MAPPING = {
            "Medical_Computation": "Medical Computation",
            "Spatial_Relationship": "Spatial Relationship",
            "Abnormality_Detection": "Abnormality Detection",
            "Organ_Identification": "Organ Identification",
            "Image_Quality": "Image Quality"
        }
        
        SUB_TYPE_MAPPING = {
            "Thickness": "Thickness",
            "Distance": "Distance",
            "Volume": "Volume",
            "Density": "Density",
            "Location": "Location",
            "Relative_Position": "Relative Position",
            "Presence": "Presence",
            "Severity": "Severity",
            "Organ_Name": "Organ Name",
            "Organ_Count": "Organ Count",
            "Contrast": "Contrast",
            "Noise": "Noise"
        }
        
        results_text, metrics, wrong_answers = evaluate_3drad(out_samples)
        
        print(results_text)
        
        wrong_answers_readable = {}
        for key, value in wrong_answers.items():
            readable_key = TYPE_MAPPING.get(key, key)
            wrong_answers_readable[readable_key] = value
        
        os.makedirs(self.output_path, exist_ok=True)
        wrong_answers_path = os.path.join(self.output_path, "wrong_answers.json")
        
        with open(wrong_answers_path, 'w', encoding='utf-8') as f:
            json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
        print("="*80 + "\n")
        
        return metrics, out_samples
