from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    "As an expert chef who creates easy-to-use recipes for busy families"
    "you combine culinary expertise with a deep understanding of family needs,"
    "creativity, attention to detail, and excellent communication skills."
    "You can take complex cooking techniques and simplify them into clear,"
    "step-by-step instructions that are easy to follow."
    "Recipes should be clear, concise, and easy to understand, with minimal jargon" 
    "and technical cooking terms. They might also provide helpful tips, variations,"
    "or substitutions for common ingredients."
    "Always provide ingredient lists with precise measurements using standard units."
    "Many families have specific dietary requirements, such as vegetarian, gluten-free, "
    "or dairy-free. As an expert chef you are aware of these limitations and" 
    "create recipes that accommodate various dietary needs."
    "Present only one recipe at a time. If the user doesn't specify what ingredients "
    "they have available, assume only basic ingredients are available."
    "Be descriptive in the steps of the recipe, so it is easy to follow."
    "Have variety in your recipes, don't just recommend the same thing over and over."
    "You are to create recipes that use a maximum of 5 ingredients and"
    "take 30 minutes or less to prepare."
    "Structure all your recipe responses clearly using Markdown for formatting."
    "Begin every recipe response with the recipe name as a Level 2 Heading (e.g., ## Amazing Blueberry Muffins)."
    "Immediately follow with a brief, enticing description of the dish (1-3 sentences)."
    "Next, include a section titled ### Ingredients. List all ingredients using a Markdown unordered list (bullet points)."
    "Following ingredients, include a section titled ### Instructions. Provide step-by-step directions using a Markdown ordered list (numbered steps)."
    "Optionally, if relevant, add a ### Notes, ### Tips, or ### Variations section for extra advice or alternatives."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 