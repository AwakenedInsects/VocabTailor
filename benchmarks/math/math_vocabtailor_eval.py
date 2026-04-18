from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import torch
from lighteval_vocabtailor import VocabTailorModel

sys.path.append("../")
from input_aware.vocab_tailor import VocabTailor
import json

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vt = VocabTailor()
    vt.load_model(model)
    vt.load_tokenizer(tokenizer)

    special_token_ids = torch.tensor(tokenizer.all_special_ids)

    if args.init_file is None:
        init_ids = special_token_ids
    else:
        with open(args.init_file,'r') as f:
            static_vocab = json.load(f)    
        # init the lm head with common tokens and task-specific tokens
        static_ids = torch.tensor(list(static_vocab.values()))
        init_ids = special_token_ids
        init_ids = torch.concat([static_ids, special_token_ids])

    vt.update_lm_head(init_ids)

    le_model = VocabTailorModel(vt)

    # Set up evaluation tracking
    evaluation_tracker = EvaluationTracker(
        output_dir="results",
        save_details=True
    )

    # Configure the pipeline
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.CUSTOM,
    )

    # Create and run the pipeline
    pipeline = Pipeline(
        tasks="lighteval|math_500|0|0",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=le_model
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lighteval with VocabTailor")
    parser.add_argument("--model_name", type=str, default="microsoft/rho-math-1b-interpreter-v0.1", help="Model path or Hugging Face model id.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp32", help="Model dtype.")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files for Hugging Face model/tokenizer.")
    parser.add_argument("--save_predictions", action="store_true", help="Save extracted predictions to .json")
    args = parser.parse_args()
    
    main(args)