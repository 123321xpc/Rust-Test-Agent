from __future__ import annotations

import json
import logging
import os
import threading

import httpx
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

from openai import BadRequestError, OpenAI
from simple_parsing.helpers.serialization.serializable import FrozenSerializable, Serializable
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sweagent.agent.commands import Command
from sweagent.utils.config import keys_config
from sweagent.utils.log import get_logger

logger = get_logger("api_models")


@dataclass(frozen=True)
class ModelArguments(FrozenSerializable):
    """Arguments configuring the model and its behavior."""

    # Name of the model to use
    model_name: str
    # Cost limit for every instance (task)
    per_instance_cost_limit: float = 0.0
    # Total cost limit
    total_cost_limit: float = 0.0
    # Sampling temperature
    temperature: float = 1.0
    # Sampling top-p
    top_p: float = 1.0
    # Path to replay file when using the replay model
    replay_path: str | None = None
    # Host URL when using Ollama model
    host_url: str = "localhost:11434"


@dataclass
class APIStats(Serializable):
    total_cost: float = 0
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other):
        if not isinstance(other, APIStats):
            msg = "Can only add APIStats with APIStats"
            raise TypeError(msg)

        return APIStats(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)},
        )

    def replace(self, other):
        if not isinstance(other, APIStats):
            msg = "Can only replace APIStats with APIStats"
            raise TypeError(msg)

        return APIStats(**{field.name: getattr(other, field.name) for field in fields(self)})


class ContextWindowExceededError(Exception):
    pass


class CostLimitExceededError(Exception):
    pass


class APIRequestBudget:
    """Count all HTTP attempts for one task, including subagent retries."""

    def __init__(self):
        raw = keys_config.get("API_LIMIT", "70")
        try:
            self.limit = int(raw)
        except (ValueError, TypeError):
            raise ValueError("API_LIMIT must be a positive integer") from None
        if self.limit < 1:
            raise ValueError("API_LIMIT must be a positive integer")
        self.count = 0
        self.lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.responses_without_usage = 0
        self.failed_requests = 0
        self.events = []

    def reset(self):
        with self.lock:
            self.count = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_tokens = 0
            self.responses_without_usage = 0
            self.failed_requests = 0
            self.events = []

    def record_event(self, event):
        with self.lock:
            self.events.append(event)

    def record_usage(self, usage):
        with self.lock:
            if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
                self.responses_without_usage += 1
                return
            self.input_tokens += usage.prompt_tokens
            self.output_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens if usage.total_tokens is not None else usage.prompt_tokens + usage.completion_tokens

    def record_failure(self):
        with self.lock:
            self.failed_requests += 1

    def snapshot(self):
        with self.lock:
            return {"api_calls": self.count, "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens,
                    "responses_without_usage": self.responses_without_usage,
                    "failed_requests": self.failed_requests, "api_limit": self.limit}

    def check(self):
        with self.lock:
            if self.count >= self.limit:
                raise RuntimeError(f"API_LIMIT reached ({self.limit} HTTP requests)")

    def consume(self):
        with self.lock:
            if self.count >= self.limit:
                raise RuntimeError(f"API_LIMIT reached ({self.limit} HTTP requests)")
            self.count += 1


class BudgetHTTPClient(httpx.Client):
    def __init__(self, model):
        # Avoid implicit redirect requests outside the task budget.
        super().__init__(follow_redirects=False)
        self.model = model

    def send(self, request, **kwargs):
        self.model.api_budget.consume()
        kwargs["follow_redirects"] = False
        try:
            response = super().send(request, **kwargs)
        except Exception:
            self.model.api_budget.record_failure()
            raise
        if response.status_code >= 300:
            self.model.api_budget.record_failure()
        return response


class BaseModel:
    MODELS = {}
    SHORTCUTS = {}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        self.args = args
        self.commands = commands
        self.model_metadata = {}
        self.stats = APIStats()
        self.api_budget = APIRequestBudget()

        # Map `model_name` to API-compatible name `api_model`
        self.api_model = (
            self.SHORTCUTS[self.args.model_name] if self.args.model_name in self.SHORTCUTS else self.args.model_name
        )

        # Unknown OpenAI-compatible models have no bundled pricing metadata.
        self.model_metadata = self.MODELS.get(self.api_model, {
            "max_context": 128000, "cost_per_input_token": 0, "cost_per_output_token": 0,
        })

    def reset_stats(self, other: APIStats | None = None):
        if other is None:
            self.stats = APIStats(total_cost=self.stats.total_cost)
            logger.info("Resetting model stats")
        else:
            self.stats = other

    def update_stats(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the cost of a response from the openai API.

        Args:
        input_tokens (int): The number of tokens in the prompt.
        output_tokens (int): The number of tokens in the response.

        Returns:
        float: The cost of the response.
        """
        # Calculate cost and update cost related fields
        cost = (
            self.model_metadata["cost_per_input_token"] * input_tokens
            + self.model_metadata["cost_per_output_token"] * output_tokens
        )
        self.stats.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        # Log updated cost values to std. out.
        logger.info(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.instance_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        logger.info(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={self.stats.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.args.total_cost_limit <= self.stats.total_cost:
            logger.warning(f"Cost {self.stats.total_cost:.2f} exceeds limit {self.args.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise CostLimitExceededError(msg)

        if 0 < self.args.per_instance_cost_limit <= self.stats.instance_cost:
            logger.warning(f"Cost {self.stats.instance_cost:.2f} exceeds limit {self.args.per_instance_cost_limit:.2f}")
            msg = "Instance cost limit exceeded"
            raise CostLimitExceededError(msg)
        return cost

    def query(self, history: list[dict[str, str]]) -> str:
        msg = "Use a subclass of BaseModel"
        raise NotImplementedError(msg)


class OpenAIModel(BaseModel):
    MODELS = {
        "gpt-3.5-turbo-0125": {
            "max_context": 16_385,
            "cost_per_input_token": 5e-07,
            "cost_per_output_token": 1.5e-06,
        },
        "gpt-3.5-turbo-1106": {
            "max_context": 16_385,
            "cost_per_input_token": 1.5e-06,
            "cost_per_output_token": 2e-06,
        },
        "gpt-3.5-turbo-16k-0613": {
            "max_context": 16_385,
            "cost_per_input_token": 1.5e-06,
            "cost_per_output_token": 2e-06,
        },
        "gpt-4-32k-0613": {
            "max_context": 32_768,
            "cost_per_input_token": 6e-05,
            "cost_per_output_token": 0.00012,
        },
        "gpt-4-0613": {
            "max_context": 8_192,
            "cost_per_input_token": 3e-05,
            "cost_per_output_token": 6e-05,
        },
        "gpt-4-1106-preview": {
            "max_context": 128_000,
            "cost_per_input_token": 1e-05,
            "cost_per_output_token": 3e-05,
        },
        "gpt-4-0125-preview": {
            "max_context": 128_000,
            "cost_per_input_token": 1e-05,
            "cost_per_output_token": 3e-05,
        },
        "gpt-4-turbo-2024-04-09": {
            "max_context": 128_000,
            "cost_per_input_token": 1e-05,
            "cost_per_output_token": 3e-05,
        },
        "gpt-4o-2024-05-13": {
            "max_context": 128_000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 15e-06,
        },
        "gpt-4o-2024-11-20": {
            "max_context": 128_000,
            "cost_per_input_token": 0.0000025,
            "cost_per_output_token": 15e-06,
        },
        "claude-3-7-sonnet-20250219":{
            "max_context": 128_000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06,
        },
        "Qwen3-32B":{
            "max_context": 32_000,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0,
        },
        "o4-mini": {
            "max_context": 200_000,
            "cost_per_input_token": 55e-08,
            "cost_per_output_token": 22e-07,
        }
    }

    SHORTCUTS = {
        "gpt3": "gpt-3.5-turbo-1106",
        "gpt3-legacy": "gpt-3.5-turbo-16k-0613",
        "gpt4": "gpt-4-1106-preview",
        "gpt4-legacy": "gpt-4-0613",
        "gpt4-0125": "gpt-4-0125-preview",
        "gpt3-0125": "gpt-3.5-turbo-0125",
        "gpt4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt4o": "gpt-4o-2024-05-13",
    }

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        logging.getLogger("openai").setLevel(logging.WARNING)
        self.client = OpenAI(api_key=keys_config["API_KEY"], base_url=keys_config.get("BASE_URL") or None,
                             http_client=BudgetHTTPClient(self), max_retries=0)

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `messages` by filtering out all keys except for role/content per `history` turn
        """
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry["content"] for entry in history])
        # Return history components with just role, content fields
        return [{k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history]

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        reraise=True,
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError)),
    )
    def query(self, history: list[dict[str, str]]) -> str:
        """
        Query the OpenAI API with the given `history` and return the response.
        """
        try:
            # SDK retries are disabled; tenacity retries are counted by the HTTP client.
            self.api_budget.check()
            response = self.client.chat.completions.create(
                messages=self.history_to_messages(history),
                model=self.api_model,
                temperature=self.args.temperature,
                # top_p=self.args.top_p,
            )
        except BadRequestError:
            msg = f"Context window ({self.model_metadata['max_context']} tokens) exceeded"
            raise CostLimitExceededError(msg)
        # Calculate + update costs, return response
        usage = response.usage
        self.api_budget.record_usage(usage)
        input_tokens = (usage.prompt_tokens or 0) if usage is not None else 0
        output_tokens = (usage.completion_tokens or 0) if usage is not None else 0
        self.update_stats(input_tokens, output_tokens)
        return response.choices[0].message.content


class HumanModel(BaseModel):
    MODELS = {"human": {}}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        # Determine which commands require multi-line input
        self.multi_line_command_endings = {
            command.name: command.end_name for command in commands if command.end_name is not None
        }

    def history_to_messages(
        self,
        history: list[dict[str, str]],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        """
        Create `messages` by filtering out all keys except for role/content per `history` turn
        """
        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry["content"] for entry in history])
        # Return history components with just role, content fields
        return [{k: v for k, v in entry.items() if k in ["role", "content"]} for entry in history]

    def query(self, history: list[dict[str, str]], action_prompt: str = "> ") -> str:
        """
        Logic for handling user input to pass to SWEEnv
        """
        action = input(action_prompt)
        command_name = action.split()[0] if action else ""

        # Special handling for multi-line input actions (i.e. edit)
        if command_name in self.multi_line_command_endings:
            buffer = [action]
            end_keyword = self.multi_line_command_endings[command_name]
            while True:
                action = input("... ")
                buffer.append(action)
                if action.rstrip() == end_keyword:
                    # Continue reading input until terminating keyword inputted
                    break
            action = "\n".join(buffer)
        elif action.strip() == "start_multiline_command":  # do arbitrary multi-line input
            buffer = []
            while True:
                action = input("... ")
                if action.rstrip() == "end_multiline_command":
                    break
                buffer.append(action)
            action = "\n".join(buffer)
        return action


class HumanThoughtModel(HumanModel):
    MODELS = {"human_thought": {}}

    def query(self, history: list[dict[str, str]]) -> str:
        """
        Logic for handling user input (both thought + action) to pass to SWEEnv
        """
        thought_all = ""
        thought = input("Thought (end w/ END_THOUGHT): ")
        while True:
            if "END_THOUGHT" in thought:
                thought = thought.split("END_THOUGHT")[0]
                thought_all += thought
                break
            thought_all += thought
            thought = input("... ")

        action = super().query(history, action_prompt="Action: ")

        return f"{thought_all}\n```\n{action}\n```"


class ReplayModel(BaseModel):
    MODELS = {"replay": {}}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        super().__init__(args, commands)

        if self.args.replay_path is None or not os.path.exists(self.args.replay_path):
            msg = "--replay_path must point to a file that exists to run a replay policy"
            raise ValueError(msg)

        self.replays = [
            list(json.loads(x).values())[0] for x in Path(self.args.replay_path).read_text().splitlines(keepends=True)
        ]
        self.replay_idx = 0
        self.action_idx = 0

    def _next_replay(self) -> None:
        """Called after last action"""
        self.replay_idx += 1
        self.action_idx = 0

    def query(self, history: list[dict[str, str]]) -> str:
        """
        Logic for tracking which replay action to pass to SWEEnv
        """
        actions = self.replays[self.replay_idx]
        try:
            action = actions[self.action_idx]
        except IndexError:
            msg = (
                "This seems to be an incomplete trajectory. "
                "We reached the end of it, but `submit` was not called. "
                "Calling it now."
            )
            logger.warning(msg)
            action = "```\nsubmit\n```"

        self.action_idx += 1

        # Assuming `submit` is always last action of replay trajectory
        if action == "submit":
            self._next_replay()

        return action


class InstantEmptySubmitTestModel(BaseModel):
    MODELS = {"instant_empty_submit": {}}

    def __init__(self, args: ModelArguments, commands: list[Command]):
        """This model immediately submits. Useful for testing purposes"""
        super().__init__(args, commands)
        self._action_idx = 0

    def query(self, history: list[dict[str, str]]) -> str:
        # Need to at least do _something_ to submit
        if self._action_idx == 0:
            self._action_idx = 1
            action = "DISCUSSION\nLet's reproduce the bug by creating a `reproduce.py` file.\n\n```\ncreate reproduce.py\n```\n"
        elif self._action_idx == 1:
            self._action_idx = 0
            action = "DISCUSSION\nThe task should be resolved, so let's submit the patch.\n\n```\nsubmit\n```\n"
        return action


def resolve_api_env(api: str | None = None, api_limit: int | None = None) -> None:
    """Resolve API profile (defaulting to yj) and API_LIMIT into os.environ."""
    if api_limit is not None:
        if isinstance(api_limit, bool) or api_limit < 1:
            raise ValueError("--api-limit must be a positive integer")
        os.environ["API_LIMIT"] = str(api_limit)
    if api is not None:
        profile = api.strip().lower()
        if profile not in ("yj", "ds"):
            raise ValueError("--api must be one of yj, ds")
        prefix = profile.upper()
        base_url = keys_config.get(f"{prefix}_BASE_URL")
        api_key = keys_config.get(f"{prefix}_API_KEY")
        model = keys_config.get(f"{prefix}_MODEL")
        if not base_url or not api_key or not model:
            raise ValueError(f"Set {prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_MODEL in .env")
        os.environ["BASE_URL"] = base_url
        os.environ["API_KEY"] = api_key
        os.environ["MODEL"] = model
        return
    if keys_config.get("API_KEY") and keys_config.get("MODEL"):
        return
    base_url = keys_config.get("YJ_BASE_URL")
    api_key = keys_config.get("YJ_API_KEY")
    model = keys_config.get("YJ_MODEL")
    if base_url and api_key and model:
        os.environ["BASE_URL"] = base_url
        os.environ["API_KEY"] = api_key
        os.environ["MODEL"] = model


def get_model(args: ModelArguments, commands: list[Command] | None = None):
    """Use the configured OpenAI-compatible API, retaining offline replay modes."""
    commands = commands or []
    offline = {"instant_empty_submit": InstantEmptySubmitTestModel, "human": HumanModel,
               "human_thought": HumanThoughtModel, "replay": ReplayModel}
    if args.model_name in offline:
        return offline[args.model_name](args, commands)
    from dataclasses import replace
    resolve_api_env()
    model_name = keys_config.get("MODEL")
    if not model_name:
        raise ValueError("Set YJ_*/DS_* (or MODEL and API_KEY) in the repository root .env file")
    return OpenAIModel(replace(args, model_name=model_name), commands)
