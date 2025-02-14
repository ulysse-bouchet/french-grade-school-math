# Imports
import asyncio
import json
import sys
import os
from langchain_openai import ChatOpenAI
from datetime import datetime
from dotenv import load_dotenv


# Asynchronous translation function
async def translate(semaphore, text, task_id):
    async with semaphore:  # Limits access
        log(f"Translating {task_id.lower()}...")
        output_msg = await llm.ainvoke(
            f"I will give you a text that you will have to translate to French. \
            Just give me the translation, without any additional context or formatting. \
            Keep the syntax as it was, for example if there is no punctuation, don't add any. \
            Here is the text you have to translate : {text}"
        )
        log(f"{task_id} translated.")
        return output_msg


# Function to save translations and metadata to JSONL file
def save_to_jsonl(translations, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for translation in translations:
            json.dump(translation, f, ensure_ascii=False)
            f.write("\n")


# Load data from JSONL file
def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


# Recursively traverse and translate text fields in a JSON object
async def translate_json_object(semaphore, obj, task_id):
    if isinstance(obj, dict):
        tasks = []
        for key, value in obj.items():
            if isinstance(value, str):
                tasks.append(translate(semaphore, value, f"{task_id} ({key})"))
            elif isinstance(value, (dict, list)):
                tasks.append(
                    translate_json_object(semaphore, value, f"{task_id} ({key})")
                )
        results = await asyncio.gather(*tasks)
        result_index = 0
        for key, value in obj.items():
            if isinstance(value, str):
                obj[key] = results[result_index].content
                result_index += 1
            elif isinstance(value, (dict, list)):
                obj[key] = results[result_index]
                result_index += 1
    elif isinstance(obj, list):
        tasks = []
        for i, item in enumerate(obj):
            if isinstance(item, str):
                tasks.append(translate(semaphore, item, f"{task_id} ({i})"))
            elif isinstance(item, (dict, list)):
                tasks.append(translate_json_object(semaphore, item, f"{task_id} ({i})"))
        results = await asyncio.gather(*tasks)
        result_index = 0
        for i in range(len(obj)):
            if isinstance(obj[i], str):
                obj[i] = results[result_index].content
                result_index += 1
            elif isinstance(obj[i], (dict, list)):
                obj[i] = results[result_index]
                result_index += 1
    return obj


# Main asynchronous function
async def main(json_objects, tasks, limit):
    semaphore = asyncio.Semaphore(tasks)  # Allows a limited amount of tasks at a time
    tasks = []

    # Determine the effective limit
    if limit == -1:
        effective_limit = len(json_objects)
    else:
        effective_limit = min(limit, len(json_objects))

    # Create tasks for JSON objects up to the effective limit
    for i in range(effective_limit):
        tasks.append(
            translate_json_object(semaphore, json_objects[i], f"Object #{i + 1}")
        )

    results = await asyncio.gather(*tasks)
    return results


# Wrapper for the print function that adds the current datetime at the beginning of the message
def log(text):
    now = datetime.now()
    print(
        f"[{now.hour:02d}:{now.minute:02d}:{now.second:02d}.{round(now.microsecond / 1000):03d}] {text}"
    )


"""
Main process of the script.
Usage : python3 translate.py <jsonl_file> <tasks_number> <lines_limit>
jsonl_file: the JSONL file to translate (required)
tasks_number: the number of async tasks run (default: 8)
lines_limit: the limit of lines to translate from the file (default: -1 | no limit)
"""
if __name__ == "__main__":
    # Check if at least the jsonl_file argument is present
    if len(sys.argv) < 2:
        print(
            """Error: argument <jsonl_file> missing.

Usage : python3 translate.py <jsonl_file> <tasks_number> <lines_limit>

jsonl_file: the JSONL file to translate (required)
tasks_number: the number of async tasks run (default: 8)
lines_limit: the limit of lines to translate from the file (default: -1 | no limit)
              """
        )
        sys.exit(1)

    # Load environment variables from .env file
    log("Loading settings from environment...")
    load_dotenv()

    # Define common settings
    common_settings = {
        "temperature": float(os.getenv("TEMPERATURE", 0.2)),
        "max_retries": int(os.getenv("MAX_RETRIES", 3)),
        "timeout": int(os.getenv("TIMEOUT", 60)),
    }
    log(f"Temperature : {common_settings['temperature']}")
    log(f"Max retries : {common_settings['max_retries']}")
    log(f"Timeout : {common_settings['timeout']} seconds")

    # Define specific settings for the language model
    llama_settings = {
        "api_key": os.getenv("API_KEY"),
        "model": os.getenv("MODEL"),
        "base_url": os.getenv("BASE_URL"),
        **common_settings,
    }
    log(f"Model : {llama_settings['model']}")
    log(f"URL : {llama_settings['base_url']}")

    log("Settings loaded.")

    # Parse script args
    log("Parsing script arguments...")

    input_file = sys.argv[1]
    output_file = f"translated/{os.path.basename(input_file)}"
    log(f"Input file : {input_file}")
    log(f"Output file : {output_file}")

    tasks = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
    log(f"Number of tasks : {tasks}")

    limit = int(sys.argv[3]) if len(sys.argv) == 4 else -1
    log(f"Lines limit : {limit if int(limit) > 0 else 'no limit'}")

    log("Script arguments parsed.")

    # Initialize the language model
    log("Initializing language model...")
    llm = ChatOpenAI(**llama_settings)
    log("Language model initialized.")

    # Load JSON objects from JSONL file
    log("Loading JSON objects from the input file...")
    json_objects = load_jsonl(input_file)
    log("JSON objects loaded.")

    # Translate JSON objects
    log("Beginning translation tasks...")
    start_time = datetime.now()
    translations = asyncio.run(main(json_objects, tasks, limit))
    time_spent = datetime.now() - start_time
    log(f"Translation tasks completed in {time_spent}.")

    # Save translations to JSONL file
    log(f"Saving translations to {output_file}...")
    save_to_jsonl(translations, output_file)
    log(f"Translations saved to {output_file}.")
