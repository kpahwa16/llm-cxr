#### Start of environment setup ####

import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=Path,
                    help='Path to LLM-CXR model checkpoint.')
parser.add_argument('--cxr_vq_path', type=Path,
                    help='Path to Vector Quantized CXR dataset pickle.')
parser.add_argument('--output_root', type=Path,
                    help='Path to save result.')
parser.add_argument('--mimic_cxr_jpg_path', type=Path, default="/lustre/fs1/groups/sernam/datasets/physionet.org/files/data/mimic-cxr-jpg/",
                    help='Path to MIMIC-CXR-JPG dataset.')
parser.add_argument('--eval_dicom_ids_path', type=Path, default="data/eval_dicom_ids.pickle",
                    help='path to eval dicom ids pickle.')
parser.add_argument('--word_size', type=int, default=1,
                    help='Number of parallel processes.')
parser.add_argument('--rank', type=int, default=0,
                    help='Rank of current process.')
args = parser.parse_args()

#nohup python generate_eval.py --model_path /home/cteja/llm-cxr2/ckpts/llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2 --cxr_vq_path /home/cteja/llm-cxr/data/mimic-cxr-256-txvloss_codebook_indices.pickle --output_root /home/cteja/llm-cxr/eval_new --mimic_cxr_jpg_path /lustre/fs1/groups/sernam/datasets/physionet.org/files/data/mimic-cxr-jpg/ --eval_dicom_ids_path /home/cteja/llm-cxr/data/eval_dicom_ids.pickle
N_PARALLEL = args.word_size
I_PARALLEL = args.rank

os.environ["CUDA_VISIBLE_DEVICES"] = str(I_PARALLEL)

#### End of environment setup ####


import pandas as pd
import pickle

from tqdm import tqdm

from training.generate_old import generate_response, load_model_tokenizer_for_generate
from training.mimiccxr_vq_dataset import sample_cxr_vq_output_instruction, sample_cxr_vq_input_instruction, CXR_VQ_TOKENIZER_LEN


def dicom_id_to_report_path(db, report_path, dicom_id: str):
    db_series = db.loc[dicom_id]
    subject_id = "p" + db_series["subject_id"]
    study_id = "s" + db_series["study_id"] + ".txt"
    subject_id_prefix = subject_id[:3]

    return report_path / Path("reports/files") / Path(subject_id_prefix) / Path(subject_id) / Path(study_id)
    
def load_report(db, report_path, dicom_id: str, parse_fun):
    report_path = dicom_id_to_report_path(db, report_path, dicom_id)
    with open(report_path, "r") as f:
        txt = f.readlines()
        
    return parse_fun(txt)
    
def parse_report_fi(txt: str) -> str:
    txt = " ".join([line.strip() for line in txt if line.strip() != ""])

    try:
        _, f_and_i = txt.split("FINDINGS:")
        try:
            f, i = f_and_i.strip().split("IMPRESSION:")
            f_and_i = f.strip() + " " + i.strip()
        except:
            f_and_i = f_and_i.strip()
    except:
        try:
            f_and_i = txt
            _, i = f_and_i.strip().split("IMPRESSION:")
            f_and_i = i.strip()
        except:
            raise ValueError

    return f_and_i
    
def parse_report_i(txt: str) -> str:
    txt = " ".join([line.strip() for line in txt if line.strip() != ""])
    
    try:
        _, impression = txt.strip().split("IMPRESSION:")
    except:
        raise ValueError
    
    return impression.strip()


if __name__ == "__main__":

    RESULT_PATH = args.output_root / Path(f"llm-cxr__eval_results_{I_PARALLEL}_{N_PARALLEL}.pickle")
    PARSE_FUNCTION = parse_report_i

    print(f"Result will be saved to {RESULT_PATH}.")

    db_split = pd.read_csv(args.mimic_cxr_jpg_path / "mimic-cxr-2.0.0-split.csv", index_col="dicom_id", dtype=str)
    db_meta = pd.read_csv(args.mimic_cxr_jpg_path / "mimic-cxr-2.0.0-metadata.csv", index_col="dicom_id", dtype=str)
    with open(args.cxr_vq_path, "rb") as f:
        db_vq = pickle.load(f)

    # filter test and validate data
    db_split = db_split.loc[(db_split['split'] == 'test') | (db_split['split'] == 'validate')]
    db_split = pd.DataFrame(db_split.index, columns=["dicom_id"])
    db = db_split.merge(db_meta, on="dicom_id")
    db.set_index("dicom_id", inplace=True)

    # filter PA and AP data
    db = db.loc[(db["ViewPosition"] == "PA") | (db["ViewPosition"] == "AP")]
    db.sort_index(inplace=True)

    with open(args.eval_dicom_ids_path, "rb") as f:
        selected_dicom_ids = pickle.load(f)

    dataset = []
    for dicom_id, subject_id in zip(db.index, db["subject_id"]):
        if dicom_id not in selected_dicom_ids:
            continue

        try:
           # raw_report = load_report(db, args.mimic_cxr_jpg_path, dicom_id, PARSE_FUNCTION)
            raw_image = [vq_elem + CXR_VQ_TOKENIZER_LEN for vq_elem in db_vq[dicom_id]]
            dataset.append({"dicom_id": dicom_id, "subject_id": subject_id, 
                            "raw_image": raw_image, 
                            "gen_report": None, "gen_image": None})
        except:
            pass
        
    dataset = dataset[I_PARALLEL::N_PARALLEL]

    model, tokenizer = load_model_tokenizer_for_generate(args.model_path)
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    print(len(dataset))
   

    print(dataset[0]["raw_image"])
    count = 0
    for data in tqdm(dataset, colour="green"):
        instruction_text = sample_cxr_vq_input_instruction()
        input_text = data["raw_image"]
        print("instruction_text", instruction_text)
        print("input_text", input_text)
        # response, _ = generate_response((instruction_text, input_text), model=model, tokenizer=tokenizer, max_new_tokens=128)
        generated_text, words, word_probabilities = generate_response(
          (instruction_text, input_text),
          model=model,
          tokenizer=tokenizer,
          max_new_tokens=128
        )
        print(generated_text)
        print(words)
        print(word_probabilities)

        data["image_id"] = data["dicom_id"]
        data["gen_report"] = generated_text
        data["words"] = words
        data["token_probs"] = word_probabilities
        print(data["dicom_id"])
        
        
    args.output_root.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "wb") as f:
        pickle.dump(dataset, f)
