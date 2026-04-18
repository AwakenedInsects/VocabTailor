import logging
import os
from datetime import timedelta
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import calculate_maximum_sizes, convert_bytes, get_max_memory
from pydantic import Field, PositiveInt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
)
from transformers.generation.utils import GenerateOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    Batch,
    ModelResponse,
)
from lighteval.models.utils import ModelConfig, _get_dtype, _get_model_sha, _simplify_name
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import (
    is_accelerate_available,
)
from lighteval.models.model_input import GenerationParameters

import sys
sys.path.append('../')

from input_aware.vocab_tailor import VocabTailor


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformersModelConfig(ModelConfig):
    """
    Configuration class for HuggingFace Transformers models.

    This configuration is used to load and configure models from the HuggingFace Transformers library.

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to a pre-trained model. This corresponds to the
            `pretrained_model_name_or_path` argument in HuggingFace's `from_pretrained` method.
        tokenizer (str | None):
            Optional HuggingFace Hub tokenizer ID. If not specified, uses the same ID as model_name.
            Useful when the tokenizer is different from the model (e.g., for multilingual models).
        subfolder (str | None):
            Subfolder within the model repository. Used when models are stored in subdirectories.
        revision (str):
            Git revision of the model to load. Defaults to "main".
        batch_size (PositiveInt | None):
            Batch size for model inference. If None, will be automatically determined.
        max_length (PositiveInt | None):
            Maximum sequence length for the model. If None, uses model's default.
        model_loading_kwargs (dict):
            Additional keyword arguments passed to `from_pretrained`. Defaults to empty dict.
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        model_parallel (bool | None):
            Whether to use model parallelism across multiple GPUs. If None, automatically
            determined based on available GPUs and model size.
        dtype (str | None):
            Data type for model weights. Can be "float16", "bfloat16", "float32", "auto", "4bit", "8bit".
            If "auto", uses the model's default dtype.
        device (Union[int, str]):
            Device to load the model on. Can be "cuda", "cpu", or GPU index. Defaults to "cuda".
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        use_chat_template (bool):
            Whether to use chat templates for conversation-style prompts. Defaults to False.
        compile (bool):
            Whether to compile the model using torch.compile for optimization. Defaults to False.
        multichoice_continuations_start_space (bool | None):
            Whether to add a space before multiple choice continuations. If None, uses model default.
            True forces adding space, False removes leading space if present.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately or together. Defaults to False.

    Example:
        ```python
        config = TransformersModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=4,
            dtype="float16",
            use_chat_template=True,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```

    Note:
        This configuration supports quantization (4-bit and 8-bit) through the dtype parameter.
        When using quantization, ensure you have the required dependencies installed
        (bitsandbytes for 4-bit/8-bit quantization).
    """

    model_name: str
    tokenizer: str | None = None
    subfolder: str | None = None
    revision: str = "main"
    batch_size: PositiveInt | None = None
    max_length: PositiveInt | None = None
    model_loading_kwargs: dict = Field(default_factory=dict)
    add_special_tokens: bool = True
    model_parallel: bool | None = None
    dtype: str | None = None
    device: Union[int, str] = "cuda"
    trust_remote_code: bool = False
    use_chat_template: bool = False
    compile: bool = False
    pairwise_tokenization: bool = False

    def get_transformers_config(self) -> PretrainedConfig:
        revision = self.revision

        if self.subfolder:
            revision = f"{self.revision}/{self.subfolder}"

        auto_config = AutoConfig.from_pretrained(
            self.model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )

        return auto_config

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.model_name, revision=self.revision)


class VocabTailorModel(LightevalModel):
    def __init__(
        self,
        vt: VocabTailor,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        config = vt.model.config
        self.config = config
        self.accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
        self._device = self.accelerator.device
        self.use_chat_template = True
        self._add_special_tokens = False
        self.pairwise_tokenization = True
        self.batch_size = 1
        self.transformers_config = vt.model.config

        transformer_model_config = TransformersModelConfig(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.model_sha = transformer_model_config.get_model_sha()
        self._max_length = self._init_max_length()
        self._tokenizer = vt.tokenizer
        self.model = vt.model
        self.vt = vt

        # # We are in DP (and launch the script with `accelerate launch`)
        # if config.model_parallel is False and self.config.dtype not in ["4bit", "8bit"]:
        #     logger.info(f"Using Data Parallelism, putting model on device {self._device}")
        #     self.model = self.model.to(self._device)
        # if config.compile:
        #     try:
        #         logger.info("Compiling the model")
        #         self.model.model.compile()
        #     except AttributeError as e:
        #         logger.warning("Could not compile the model because: ", e)

        self.model_name = _simplify_name("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

        self.generation_config_dict = self.vt.model.generation_config

        if is_accelerate_available():
            model_size, _ = calculate_maximum_sizes(self.model)
            model_size = convert_bytes(model_size)
        else:
            model_size = -1

        self.model_info = ModelInfo(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            model_sha=self.model_sha,
            model_dtype=self.vt.model.dtype,
            model_size=model_size,
        )

        self.prompt_manager = PromptManager(
            use_chat_template=self.use_chat_template, tokenizer=self.tokenizer, system_prompt=None
        )

    def cleanup(self):
        """Clean up operations if needed, such as closing an endpoint."""
        del self.model
        del self._tokenizer
        torch.cuda.empty_cache()

    @classmethod
    def from_model(
        cls,
        model: Union[AutoModelForCausalLM, LightevalModel],
        config: TransformersModelConfig = None,
        accelerator: "Accelerator" = None,
        tokenizer_name: str = None,  # custom tokenizer
        trust_remote_code: bool = False,
        add_special_tokens: bool = True,
        pairwise_tokenization: bool = False,
        multichoice_continuations_start_space: bool = None,
    ):
        # Slightly hackish way to test if the model is a AutoModelForCausalLM, since the instances don't
        # derive from this class explicitely
        assert isinstance(model, LightevalModel) or type(model).__name__ in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()

        if isinstance(model, LightevalModel):
            return model

        # Instanciate the object without using __init__
        self = cls.__new__(cls)
        self.config = config
        self.transformers_config = model.config
        self.generation_config_dict = config.generation_parameters.to_transformers_dict()
        self._max_length = self._init_max_length()
        self._tokenizer = self._create_auto_tokenizer()
        self.batch_size = config.batch_size
        self.model_name = _simplify_name(model.name_or_path)
        self.model_sha = config.get_model_sha()

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

        self.accelerator = accelerator
        if accelerator is not None:
            self._device = accelerator.device
            self.model = self.accelerator.prepare(self.model.to(accelerator.device))
        else:
            self._device = self.config.device

        self.use_chat_template = config.use_chat_template if config else False
        self._add_special_tokens = add_special_tokens if add_special_tokens is not None else False
        self.pairwise_tokenization = True
        self.multichoice_continuations_start_space = multichoice_continuations_start_space

        self.precision = _get_dtype(model.dtype, config=self.transformers_config)

        if is_accelerate_available():
            model_size, _ = calculate_maximum_sizes(self.model)
            model_size = convert_bytes(model_size)
        else:
            model_size = -1
        self.model_info = ModelInfo(
            model_name=self.model_name,
            model_sha=self.model_sha,
            model_dtype=str(self.precision),
            model_size=int(model_size),
        )
        return self

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

    @property
    def disable_tqdm(self) -> bool:
        disable_tqdm = False
        if self.accelerator:
            disable_tqdm = bool(not self.accelerator.is_main_process)
        return disable_tqdm

    def init_model_parallel(self, model_parallel: bool | None = None) -> Tuple[bool, Optional[dict], Optional[str]]:
        """Compute all the parameters related to model_parallel"""
        if not is_accelerate_available():
            return False, None, None

        self.num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.num_machines = torch.cuda.device_count() // self.num_local_processes

        if self.num_machines == 1:
            logger.info("We are not in a distributed setting. Setting model_parallel to False.")
            model_parallel = False

        if model_parallel is None:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            model_parallel = bool(self.num_local_processes < len(max_memory_all_gpus))
            logger.info(
                f"Setting model parallel to {model_parallel} since "
                f"the number of local processes is {self.num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )
        if model_parallel is True:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_mem_this_process = {
                k: v
                for k, v in max_memory_all_gpus.items()
                if k % self.num_local_processes == (self.accelerator.process_index % self.num_local_processes)
            }
            device_map = "auto"
            logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_mem_this_process} and device map to {device_map}"
            )
        else:
            max_mem_this_process = None
            device_map = None
            logger.info(
                f"Model parallel was set to False, max memory set to {max_mem_this_process} and device map to {device_map}"
            )
        return model_parallel, max_mem_this_process, device_map

    def _create_auto_model(self) -> transformers.PreTrainedModel:
        """
        Creates an instance of the pretrained HF model.

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        model_parallel, max_memory, device_map = self.init_model_parallel(self.config.model_parallel)
        self.config.model_parallel = model_parallel

        if self.config.dtype == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif self.config.dtype == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        torch_dtype = _get_dtype(self.config.dtype)
        subfolder = self.config.subfolder
        revision = self.config.revision + (f"/{subfolder}" if subfolder is not None else "")

        pretrained_config = self.transformers_config

        kwargs = self.config.model_loading_kwargs.copy()
        if "quantization_config" not in pretrained_config.to_dict():
            kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=revision,
            max_memory=max_memory,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            **kwargs,
        )
        # model.to(self.device)
        model.eval()
        torch.set_grad_enabled(False)

        if self.config.compile:
            try:
                logger.info("Compiling the model")
                model.compile()
            except AttributeError as e:
                logger.warning("Could not compile the model because: ", e)

        return model

    def _create_auto_tokenizer(
        self,
    ) -> transformers.PreTrainedTokenizer:
        """
        Create a Hugging Face AutoTokenizer for language model.

        Returns:
            transformers.PreTrainedTokenizer: The created tokenizer.
        """
        tokenizer_name = self.config.tokenizer or self.config.model_name
        subfolder = self.config.subfolder
        revision = self.config.revision + (f"/{subfolder}" if subfolder is not None else "")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=revision,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
            truncation_side="left",
        )
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.max_length
        logger.info("Tokenizer truncation and padding size set to the left side.")

        return tokenizer

    def _init_max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Returns:
            int: Max length to use depending on the available args and config
        """

        if self.config.max_length is not None:
            return self.config.max_length

        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.transformers_config, attr):
                return getattr(self.transformers_config, attr)

        logger.warning(
            "No max_length attribute found in the model config. Using the default max sequence length setting {2048}. It is recomended to set max_length through the model args"
        )

        return 2048

    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        results = []
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            if split[0].generation_size is None:
                # No constraints on the generation size: max length allowed is the max model context
                max_context_continuation_size_allowed = self.max_length
            else:
                context = self.prompt_manager.prepare_prompt(split[0])
                tokenized_context = self.tokenizer(context)

                # Longest context in the current split is the first item (since we sort reversed)
                longest_context_continuation_size_in_split = len(tokenized_context) + split[0].generation_size
                max_context_continuation_size_allowed = min(
                    longest_context_continuation_size_in_split, self.max_length
                )
            batch_size = 1

            dataloader = DataLoader(split, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                contexts = [self.prompt_manager.prepare_prompt(doc) for doc in batch]

                # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
                if self.use_chat_template:
                    stop_tokens = []
                else:
                    # NOTE: we are assuming all items in a batch behave similarly (same
                    # stop_tokens and max_tokens genrated) which is not necessarily
                    # the case! Because of that we only use batch size of 1
                    stop_tokens = batch[0].stop_sequences

                max_new_tokens = batch[0].generation_size
                num_samples = batch[0].num_samples

                # See doc https://huggingface.co/docs/transformers/v4.38.2/en/pad_truncation#padding-and-truncation
                # Will do left truncation and padding, as defined when creating the tokenizer
                tokenized = self.tokenizer(
                    contexts,
                    truncation="longest_first",  # we truncate to the model max length if needed
                    padding="longest",  # we pad to the longest sequence
                    return_tensors="pt",
                    max_length=max_context_continuation_size_allowed,  # we always allow minimum one token of generation
                    add_special_tokens=self.add_special_tokens,
                )

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                context_size = tokenized["input_ids"].shape[1]
                if context_size > self.max_length:
                    logger.warning(
                        f"The context size of your batch ({context_size}) is bigger than the maximum context size allowed by the model ({self.max_length}) for a task in"
                        + str({i.task_name for i in batch})
                        + ". This is likely to lead to some errors."  # noqa C401
                    )
                    # There will be truncation of at least one sample, maximum generation size will be one
                    max_new_tokens = 1
                else:  # We can't allow generation of more than max_length
                    if max_new_tokens is None:  # If generation size is not set, we go all the way
                        max_new_tokens = self.max_length - context_size
                    else:
                        max_new_tokens = min(self.max_length - context_size, max_new_tokens)
                        if max_new_tokens < 1:
                            max_new_tokens = 1

                prepared_batch = Batch(
                    input_ids=tokenized["input_ids"],
                    input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                    input_mask=tokenized["attention_mask"],
                    truncated=[max(len(c) - tokenized["input_ids"].shape[1], 0) for c in contexts],
                    padded=[sum(mask == 0) for mask in tokenized["attention_mask"]],
                )

                # breakpoint()

                cur_reponses = self._generate(
                    batch=prepared_batch,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=False,
                    num_samples=num_samples,
                )
                results.extend(cur_reponses)

        return dataset.get_original_order(results)

    def _generate(
        self,
        batch: Batch,
        max_new_tokens: int,
        stop_tokens: list[str],
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
    ) -> list[ModelResponse]:
        """Contains the actual logic of the generation.
        First computes the stop sequences, then generates the predictions, then converts the outputs to GenerativeResponse.
        """
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop_sequences=stop_tokens, batch=batch)
        batch_size, _ = batch.input_ids.shape

        if num_samples > 1 and self.generation_config_dict["temperature"] == 0:
            raise ValueError(
                "You cannot generate multiple samples with temperature=0. Please set temperature > 0. Or use a non sampling metric."
            )

        generation_config = self.generation_config_dict.copy()
        generation_config.update(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=num_samples,
            output_logits=returns_logits,
            renormalize_logits=True,
        )
        if num_samples > 1 and generation_config["temperature"] == 0:
            logger.warning("num_samples > 1 but temperature is set to 0, this will not sample different outputs.")

        # Compute model generation
        outputs: GenerateOutput = self.vt.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            stopping_criteria=stopping_criteria,
            **generation_config,
        )
        generations = outputs.sequences[:, batch.input_ids.size(1) :]
        generations = torch.reshape(generations, (batch_size, num_samples, -1))
        generations, len_gens = self.pad_and_gather(generations, num_samples=num_samples)
        batch.input_ids, len_ids = self.pad_and_gather(batch.input_ids)

        logits, len_logits = None, None
        if returns_logits:
            logits, len_logits = self.pad_and_gather(outputs.logits)
            logits = logits.cpu().numpy()

        # We gather remaining info
        batch.truncated = torch.tensor(batch.truncated, device=self.device)
        if self.accelerator:
            batch.truncated = self.accelerator.gather_for_metrics(batch.truncated)
        batch.padded = torch.tensor(batch.padded, device=self.device)
        if self.accelerator:
            batch.padded = self.accelerator.gather_for_metrics(batch.padded)

        # We convert to GenerativeResponse outputs
        all_responses = []
        for ix, (batched_generations, batched_input, trunc, padded) in enumerate(
            zip(generations, batch.input_ids, batch.truncated, batch.padded)
        ):
            result_generations = []
            decoded_generations = []
            # Ensure the generated responses do not contain the stop sequences.
            for generation in batched_generations:
                generation = generation[: len_gens[ix]]
                result_generations.append(generation)
                decoded_generation = self.tok_decode([generation])[0]

                for term in stop_tokens:
                    decoded_generation = decoded_generation.split(term)[0]

                decoded_generations.append(decoded_generation)

            cur_response = ModelResponse(
                text=decoded_generations,
                output_tokens=result_generations,
                logits=logits[ix][: len_logits[ix]] if returns_logits else None,
                input_tokens=batched_input[: len_ids[ix]].tolist(),
                truncated_tokens_count=trunc.cpu().item(),
                padded_tokens_count=padded.cpu().item(),
            )
            all_responses.append(cur_response)

        return all_responses

    def loglikelihood(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (list[Tuple[str, dict]]): _description_

        Returns:
            list[Tuple[float, bool]]: _description_
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError
    
    def loglikelihood_single_token(self, token: str, context: str) -> float:
        """This function is used to compute the log likelihood of a single token."""
        raise NotImplementedError


    def pad_and_gather(
        self, output_tensor: torch.Tensor, drop_last_samples: bool = True, num_samples: int = None
    ) -> torch.Tensor:
        """
        Pads the `output_tensor` to the maximum length and gathers the lengths across processes.

        Args:
            output_tensor (torch.Tensor): The output tensor to be padded.
            drop_last_samples (bool, optional): Whether to drop the last samples during gathering.
            Last samples are dropped when the number of samples is not divisible by the number of processes.
                Defaults to True.

        Returns:
            torch.Tensor: The padded output tensor and the gathered length tensor.
        """
        # Create a tensor of size batch_size, [output_length] * batch_size, for each process
        # output_tensor can be of size: batch_size * num_samples * length_item or just batch_size * length_item
        length_tensor = torch.tensor([output_tensor.shape[-1]] * output_tensor.shape[0], device=self.device)
        if self.accelerator is not None:
            # Gather all the lengths, we end up with a tensor of size num_processes [output_length_1, output_length_2, ...]
            length_tensor = self.accelerator.gather(length_tensor)
        # We pad the output_tensor to the max length
        max_length = length_tensor.max().item()
        padding = (
            (0, max_length - output_tensor.shape[-1], 0, 0, 0, 0)
            if num_samples is not None
            else (0, max_length - output_tensor.shape[-1], 0, 0)
        )
        output_tensor = F.pad(output_tensor, padding, value=self.tokenizer.pad_token_id)
        if self.accelerator:
            if drop_last_samples:
                output_tensor = self.accelerator.gather_for_metrics(output_tensor)
            else:
                output_tensor = self.accelerator.gather(output_tensor)
        return output_tensor, length_tensor


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch: Batch = None,
        input_ids_shape: Tuple[int, int] = None,
    ):
        if batch is not None:
            initial_decoder_input_length = batch.input_ids.shape[1]
            batch_size = batch.input_ids.shape[0]
        else:
            initial_decoder_input_length = input_ids_shape[1]
            batch_size = input_ids_shape[0]

        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: list[str],
    batch: Batch,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[MultiTokenEOSCriteria(sequence, tokenizer, batch) for sequence in stop_sequences],
        ]
    )