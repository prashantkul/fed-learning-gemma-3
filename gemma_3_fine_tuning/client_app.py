"""gemma-3-fine-tuning: A Flower / PyTorch app."""

import torch
import os
import warnings

from typing import Dict, Tuple
from omegaconf import DictConfig
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from gemma_3_fine_tuning.models import get_model, set_parameters, get_parameters, cosine_annealing
from gemma_3_fine_tuning.dataset import get_tokenizer_and_data_collator_and_propt_formatting, load_data, replace_keys
from transformers import TrainingArguments
from trl import SFTTrainer
from flwr.common.config import unflatten_dict

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset

        # instantiate model
        self.model = get_model(model_cfg)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )

    def evaluate(self, parameters, config):
        # No evaluation in this example
        return 0.0, 0, {}


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    )


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

