"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import evaluate
import IPython
import torch
import transformers
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from transformers import (CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          TrainerCallback, TrainingArguments,
                          default_data_collator, is_torch_tpu_available,
                          set_seed)
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import EvalLoopOutput, get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import consts
import datasets
from datasets import load_dataset
from training import *
from util import set_all_seeds

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


class CustomTensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        for k, v in logs.items():  # type: ignore
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, state.global_step)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_counter = 0

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        self.step_counter += 1
        if self.step_counter > consts.EVAL_STEPS_WARMUP and self.step_counter % consts.EVAL_STEPS_OVERWRITE != 0:
            return EvalLoopOutput(predictions=[], label_ids=[],
                                  metrics={}, num_samples=0)
        return super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )


def main():
    model_args, data_args, training_args = get_args()
    set_all_seeds(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            shuffle=True,
        )
        if "validation" not in raw_datasets.keys():  # type: ignore
            raw_datasets["validation"] = load_dataset(  # type: ignore
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(  # type: ignore
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():  # type: ignore
            raw_datasets["validation"] = load_dataset(  # type: ignore
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(  # type: ignore
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()  # type: ignore
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer = get_tokenizer(model_args)
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel()
                       for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")

    def process_data(data):
        input_ids = []
        attention_masks = []
        labels = []
        for a, b in zip(data['a'], data['b']):  # type: ignore
            encoded = tokenizer('<|bos|>' + a + '<|sep|>' + b + '<|eos|>')
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
            label = [-100] * \
                (encoded.input_ids.index(tokenizer.sep_token_id))
            label.append(tokenizer.sep_token_id)
            label += encoded.input_ids[len(label):]
            labels.append(label)
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    with training_args.main_process_first(desc="dataset map tokenization"):
        lm_datasets = raw_datasets.map(
            process_data,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,  # type: ignore
            remove_columns=['a', 'b'],
            load_from_cache_file=not data_args.overwrite_cache,  # type: ignore
            desc="Preprocessing dataset",  # type: ignore
        )

    tokenized_datasets = lm_datasets

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(  # type: ignore
                range(max_train_samples))
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(  # type: ignore
                range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        bleu_metric = evaluate.load("bleu")
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        # def compute_metrics(eval_preds):
        #     preds, labels = eval_preds
        #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
        #     # by preprocess_logits_for_metrics but we need to shift the labels
        #     labels = labels[:, 1:].reshape(-1)
        #     preds = preds[:, :-1].reshape(-1)
        #     return metric.compute(predictions=preds, references=labels)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            preds_strings = []
            labels_strings = []
            preds_list = []
            labels_list = []

            for pred, label in zip(preds, labels):
                valid_mask = label != -100
                pred_non_ignore = pred[valid_mask]
                label_non_ignore = label[valid_mask]

                preds_strings.append(tokenizer.decode(
                    pred_non_ignore, skip_special_tokens=True))
                labels_strings.append(tokenizer.decode(
                    label_non_ignore, skip_special_tokens=True))

                preds_list.extend(pred_non_ignore.tolist())
                labels_list.extend(label_non_ignore.tolist())

            return bleu_metric.compute(
                predictions=preds_strings, references=labels_strings)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True)

    # optimizer = Adafactor(model.parameters(), scale_parameter=True,
    #                       relative_step=True, warmup_init=True, lr=None)
    lr = consts.TRAIN_LR
    optimizer = Adafactor(model.parameters(), scale_parameter=False,
                          relative_step=False, warmup_init=False, lr=lr)
    lr_scheduler = AdafactorSchedule(optimizer, initial_lr=lr)

    logging.get_verbosity = lambda: logging.NOTSET  # type: ignore

    tensorboard_callback = CustomTensorBoardCallback(log_dir="runs")

    # training_args.eval_steps = 1
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     optimizers=(optimizer, lr_scheduler),  # type: ignore
    #     train_dataset=train_dataset if training_args.do_train else None,  # type: ignore
    #     eval_dataset=eval_dataset if training_args.do_eval else None,  # type: ignore
    #     tokenizer=tokenizer,
    #     # Data collator will default to DataCollatorWithPadding, so we change it.
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available(  # type: ignore
    #     ) else None,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics  # type: ignore
    #     if training_args.do_eval and not is_torch_tpu_available()
    #     else None,
    #     callbacks=[tensorboard_callback],
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),  # type: ignore
        train_dataset=train_dataset if training_args.do_train else None,  # type: ignore
        eval_dataset=eval_dataset if training_args.do_eval else None,  # type: ignore
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available(  # type: ignore
        ) else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics  # type: ignore
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=[tensorboard_callback],
    )

    trainer.evaluate()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)  # type: ignore
        )
        metrics["train_samples"] = min(
            max_train_samples, len(train_dataset))  # type: ignore

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)  # type: ignore
        metrics["eval_samples"] = min(
            max_eval_samples, len(eval_dataset))  # type: ignore
        try:
            perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        except:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path,
              "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
