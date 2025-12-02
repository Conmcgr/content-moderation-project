import os
import time
import re
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for batch content moderation classification
SYSTEM_PROMPT = """You are a content moderation classifier. Your task is to classify tweets into exactly one of three categories:

0 - Hate Speech: Content containing severe toxicity, identity-based attacks, or threats of violence
1 - Offensive Language: Content with general toxicity, insults, or profanity that does not constitute hate speech
2 - Neither: Content that is safe and does not contain offensive language or hate speech

You will receive multiple tweets numbered sequentially. Classify each one independently."""


async def predict_tweet_batch(tweets, batch_num, model="gpt-5-nano-2025-08-07"):
    """
    Classify a batch of tweets (up to 10) asynchronously.

    Args:
        tweets: List of tweet texts to classify (up to 10 tweets)
        batch_num: Batch number for progress tracking
        model: OpenAI model to use

    Returns:
        list: List of predicted classes (integers 0, 1, or 2)
    """
    try:
        # Format tweets for the prompt
        tweets_text = "\n\n".join([f"Tweet {i+1}: {tweet}" for i, tweet in enumerate(tweets)])

        # Create JSON schema for batch response
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": tweets_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "batch_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "classifications": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "enum": [0, 1, 2]
                                }
                            }
                        },
                        "required": ["classifications"],
                        "additionalProperties": False
                    }
                }
            }
        )

        response_text = response.choices[0].message.content.strip()
        result = json.loads(response_text)

        # Type checking and validation
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")
        if "classifications" not in result:
            raise ValueError("Response missing 'classifications' field")

        classifications = result["classifications"]

        # Validate we got the right number of classifications
        if len(classifications) != len(tweets):
            print(f"Warning: Expected {len(tweets)} classifications, got {len(classifications)}")
            # Pad or truncate to match
            if len(classifications) < len(tweets):
                classifications.extend([1] * (len(tweets) - len(classifications)))
            else:
                classifications = classifications[:len(tweets)]

        print(f"✓ Batch {batch_num} completed ({len(tweets)} tweets)")
        return classifications

    except Exception as e:
        print(f"Error processing batch {batch_num}: {e}")
        # Fallback: classify all as offensive (class 1)
        return [1] * len(tweets)


async def process_batches_async(batches, model="gpt-5-nano-2025-08-07", concurrency=5):
    """
    Process multiple batches concurrently with a concurrency limit.

    Args:
        batches: List of tweet batches
        model: OpenAI model to use
        concurrency: Maximum number of concurrent requests

    Returns:
        list: List of all predictions
    """
    all_predictions = []

    # Process batches in groups of 'concurrency'
    for i in range(0, len(batches), concurrency):
        batch_group = batches[i:i+concurrency]
        print(f"\nProcessing batches {i+1}-{min(i+concurrency, len(batches))} of {len(batches)}...")

        # Run batches concurrently
        tasks = [
            predict_tweet_batch(batch, i+j+1, model)
            for j, batch in enumerate(batch_group)
        ]
        results = await asyncio.gather(*tasks)

        # Flatten results
        for result in results:
            all_predictions.extend(result)

    return all_predictions


def predict_batch(comment_list, model="gpt-5-nano-2025-08-07", batch_size=10, concurrency=5):
    """
    Run prediction on a list of tweets using batched async requests.

    Args:
        comment_list: List of tweet texts to classify
        model: OpenAI model to use
        batch_size: Number of tweets to process in each batch (default: 10)
        concurrency: Number of batches to process concurrently (default: 5)

    Returns:
        list: List of predicted classes (integers 0, 1, or 2)
    """
    try:
        # Split comments into batches
        batches = [
            comment_list[i:i+batch_size]
            for i in range(0, len(comment_list), batch_size)
        ]

        print(f"Processing {len(comment_list)} tweets in {len(batches)} batches of up to {batch_size}")
        print(f"Running {concurrency} batches concurrently\n")

        # Run async processing - handle Jupyter's event loop
        try:
            # Check if there's already a running event loop (e.g., in Jupyter)
            loop = asyncio.get_running_loop()
            # If we're here, there's a running loop - use nest_asyncio or create task
            import nest_asyncio
            nest_asyncio.apply()
            predictions = asyncio.run(process_batches_async(batches, model, concurrency))
        except RuntimeError:
            # No running loop, use asyncio.run normally
            predictions = asyncio.run(process_batches_async(batches, model, concurrency))

        print(f"\n✓ Completed processing {len(predictions)} tweets.")
        return predictions

    except KeyboardInterrupt:
        print(f"\n\nInterrupted!")
        print("Returning empty results...")
        return []
