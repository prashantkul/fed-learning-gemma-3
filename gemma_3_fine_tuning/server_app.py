"""Gemma3 tuning: A Flower / FlowerTune app."""

import os
from datetime import datetime

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig

from gemma_3_fine_tuning.models import get_model, get_parameters, set_parameters
from gemma_3_fine_tuning.dataset import replace_keys
from gemma_3_fine_tuning.strategy import FlowerTuneLlm


# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the
    client's fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Limit server to 20% of GPU memory (adjust as needed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.2, 0)

    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # Define strategy
    strategy = FlowerTuneLlm(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
