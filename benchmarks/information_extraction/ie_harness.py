"""
LM-eval adapter for Information extraction: VocabTailorLM wraps the package VocabTailor
so it can be used with lm_eval.evaluator.evaluate(). Used by ie_vocabtailor_eval.py only.
Baseline (ie_baseline_eval.py) uses lm_eval.models.huggingface.HFLM and does not import this.
IE is dynamic-only for VocabTailor: no static task vocabulary or profiling_file; input-aware pruning only.
Task-specific only; for shared helpers (e.g. infer_short_model_name) scripts import from benchmark_utils.
"""
import json
import logging
import os
import sys

from tqdm import tqdm
import torch

_BENCHMARKS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARKS not in sys.path:
    sys.path.insert(0, _BENCHMARKS)

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
    from lm_eval.utils import make_table
except ImportError as e:
    raise ImportError(
        "lm_eval is required for IE VocabTailor evaluation. Install with: pip install vocab-tailor[ie]"
    ) from e

logger = logging.getLogger(__name__)


class VocabTailorLM(LM):
    """lm_eval LM adapter for the package VocabTailor. Implements generate_until only."""
    
    def __init__(self, vocab_tailor, mode: str = "input_aware", max_gen_toks: int = 256):
        super().__init__()
        self.vocab_tailor = vocab_tailor
        self._tokenizer = vocab_tailor.tokenizer
        self._mode = mode
        self._max_gen_toks = max_gen_toks
        
    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id
    
    @property
    def max_length(self):
        return getattr(
            self.vocab_tailor.model.config,
            "max_position_embeddings",
            4096,
        )
    
    @property
    def max_gen_toks(self):
        return self._max_gen_toks # or whatever your max generation length should be
    
    @property
    def batch_size(self):
        return 1  # adjust based on your implementation
    
    @property
    def device(self):
        return str(next(self.vocab_tailor.model.parameters()).device)  # or whatever device your model is on
    
    def tok_encode(self, string: str, add_special_tokens: bool = False, **kwargs):
        return self._tokenizer.encode(string, add_special_tokens=add_special_tokens, **kwargs)
    
    def tok_decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens, **kwargs)
    
    def _model_call(self, inps):
        # This method should take numpy array of inputs and return logits
        # You'll need to implement this based on how your VocabTailor works
        raise NotImplementedError("This needs to be implemented based on your VocabTailor")
    
    def _model_generate(self, context, max_length, eos_token_id):
        # Use your VocabTailor's generate method
        return self.vocab_tailor.generate(
            context, 
            mode=self._mode,
            max_length=max_length, 
            original_eos_token_id = eos_token_id
        )
        
    def loglikelihood(self, requests):
        # Implement if needed for your tasks
        raise NotImplementedError("loglikelihood not implemented for VocabTailor")
    
    def loglikelihood_rolling(self, requests):
        # Implement if needed for your tasks
        raise NotImplementedError("loglikelihood_rolling not implemented for VocabTailor")
    
    def generate_until(self, requests):
        if not requests:
            return []
        
        res = []

        input_length = []
        for request in tqdm(requests):
            # Get the context and generation arguments
            context = request.args[0]
            gen_args = request.args[1]
            
            # Extract generation parameters
            max_length = gen_args.get("max_length", self.max_gen_toks)
            until = gen_args.get("until", None)
            # Generate using your VocabTailor
            try:
                context_id = torch.tensor([self.tok_encode(context)])
                input_length.append(len(set(context_id[0])))  # Ensure unique tokens
                generated = self.vocab_tailor.generate(
                    context_id,
                    mode="input_aware",
                    do_sample=False,
                    original_eos_token_id=self.eot_token_id,
                    max_length=context_id.shape[1] + self.max_gen_toks,
                    # stop_tokens=[self.tokenizer.encode(u, add_special_tokens=False)[0] for u in until]
                )
                generated_text = self.tok_decode(generated[0])
                # Extract the generated text (after context)
                generated_text = generated_text[len(context):].strip()
                res.append(generated_text)
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                res.append("")  # Append empty string on error
        print(f"avg input length: {sum(input_length)/len(input_length)}")
        return res


def evaluate_vocabtailor(vocab_tailor, tasks_to_run: list[str], limit = None) -> dict:
    """
    Evaluate a VocabTailor instance using lm-eval
    
    Args:
        vocab_tailor: Your VocabTailor instance
        tasks_to_run: List of tasks to evaluate on (e.g., ["hellaswag", "arc_challenge"])
        limit: Optional limit on number of samples per task
    """
    try:
        # Create the LM wrapper
        logger.info("Creating VocabTailor LM wrapper")
        # IE is always input-aware (no static task vocabulary)
        lm = VocabTailorLM(vocab_tailor, mode="input_aware", max_gen_toks=256)
        
        # Load tasks
        logger.info(f"Loading tasks: {tasks_to_run}")
        task_list = tasks.get_task_dict(tasks_to_run)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.evaluate(
            lm=lm,
            task_dict=task_list,
            limit=limit,
        )

        # Print results
        logger.info("Evaluation results:\n")
        logger.info(f"{make_table(results)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def _to_builtin(x) -> dict:
    """Convert numpy / torch scalar-like objects to plain Python types."""
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x


def save_lmeval_outputs(
    results: dict,
    task_name: str = "squad_completion",
    pred_path: str = None,
    metric_path: str = None,
) -> None:
    """
    Save the predictions and metrics from the results of the evaluation.

    Args:
        results: The results of the evaluation.
        task_name: The name of the task.
        pred_path: The .json path to save the predictions.
        metric_path: The .json path to save the metrics.
    """ 
    # --- 1) prediction file ---
    samples = results["samples"][task_name]

    prediction_entries = []
    for sample in samples:
        entry = {
            "doc": sample["doc"]["text"],
            "question": sample["doc"]["question"],
            "prediction": sample["filtered_resps"][0],
            "target": sample["target"],
        }
        prediction_entries.append(entry)

    if pred_path:
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(prediction_entries, f, indent=4)
        logger.info(f"Predictions saved to: {pred_path}")

    # --- 2) metric file ---
    # LM-Eval usually stores aggregate metrics under results["results"][task_name]
    raw_metrics = results["results"][task_name]

    metric_dict = {}
    for k, v in raw_metrics.items():
        metric_dict[k] = _to_builtin(v)

    if metric_path:
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(metric_dict, f, indent=4)
        logger.info(f"Metrics saved to: {metric_path}")