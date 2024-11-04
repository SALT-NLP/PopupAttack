"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any
from multiprocessing import Process, Queue, current_process
import requests
from requests.exceptions import SSLError, Timeout, RequestException
import aiolimiter
import openai
from openai import AsyncOpenAI, OpenAI
from openai import AzureOpenAI, AsyncAzureOpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

az_client = AzureOpenAI(
        api_key="XXX",  
        api_version="XXX",
        azure_endpoint = "XXX"
    )

az_aclient = AsyncAzureOpenAI(
        api_key="XXX",  
        api_version="XXX",
        azure_endpoint = "XXX"
    )

from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    print(f"[{datetime.now()}] Maximum number of retries ({max_retries}) exceeded.")
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Log retry attempt
                print(f"[{datetime.now()}] Retry {num_retries} after error: {e}. Next retry in {delay:.2f} seconds.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await az_aclient.completions.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    response = az_client.completions.create(
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response["choices"][0]["text"]
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await az_aclient.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    response = az_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    answer: str = response.choices[0].message.content
    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer

def make_request(messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    queue):
    """The target function that will be executed in a separate process."""
    try:
        response = az_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        answer: str = response.choices[0].message.content
        print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), "requess succeed!")
        # print(response.json())
        queue.put(answer)  # Put the response object in the queue
    except Exception as e:
        queue.put("CALLING ERROR:" + str(e))  # Put the error message in the queue

def run_with_limited_time(func, args, time_limit, check_interval=5):
    """Runs a function with a time limit using multiprocessing and checks periodically for early termination.

    :param func: The function to run
    :param args: The function's args, given as tuple
    :param time_limit: The time limit in seconds
    :param check_interval: The interval in seconds for checking if the process has finished
    :return: A tuple (success, result) where success is True if the function ended successfully and
             result is either the response or an error message.
    """

    queue = Queue()  # Create a queue to communicate between processes
    p = Process(target=func, args=(*args, queue))  # Pass the queue to the subprocess
    p.start()

    start_time = time.time()
    
    try:
        while time.time() - start_time < time_limit:
            try:
                # Use timeout on get() to avoid hanging
                result = queue.get(timeout=check_interval)
                return True, result  # Successfully got a result
            except Exception:
                # Timeout passed, check again
                print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), f"Wait {time.time() - start_time:.2f} seconds")

        # If we reached the time limit, terminate the process
        if p.is_alive():
            p.terminate()
            return False, None
    finally:
        p.join()  # Ensure the process is always joined to avoid zombie processes

def make_request_with_retry_multiprocess(messages: list[dict[str, str]], model: str, temperature: float, context_length: int, max_tokens: int, top_p: float, max_retries=5, per_attempt_timeout=300, backoff_factor=20):
    """Attempts to make a request with retry logic and per-attempt timeout using multiprocessing."""
    print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), "make request with retry and multiprocessing...")
    for attempt in range(max_retries):
        success, result = run_with_limited_time(make_request, (messages, model, temperature, max_tokens, top_p), per_attempt_timeout)

        if success:
            if "CALLING ERROR:" not in result:
                return result  # Successful response, return it
            else:
                print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), f"Request failed with error: {result}. Retrying... (Attempt {attempt + 1} of {max_retries})")
        else:
            print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), f"Request timed out. Retrying... (Attempt {attempt + 1} of {max_retries})")

        # Wait for some time before the next retry
        time.sleep(backoff_factor * attempt)

    # Raise an error after max retries
    raise Exception(f"Failed to get a valid response after {max_retries} attempts")