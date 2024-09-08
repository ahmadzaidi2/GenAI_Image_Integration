"""
This module demonstrate a fastAPI application that contains 2 routes 
1) Accept instructions and generate image as output
2) Accept image and generate variation of the image
"""

import json
from datetime import datetime
import configparser
import logging
import traceback
import os
from hashlib import sha256
from base64 import b64decode
import openai
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from prompts import SYSTEM_PROMPT, PROMPT_IMAGE, PROMPT_IMAGE_ELEMENTS
from retry_utils import retry
from rate_limiter import rate_limiter
import dotenv

dotenv.load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

max_tokens = int(config['llm']['max_tokens'])
temperature = int(config['llm']['temperature'])
summary_length = int(config['llm']['summary_length'])
summary_llm = config['llm']['summary_llm']
number = int(config['image']['number'])
image_size = config['image']['image_size']
image_model = config['image']['image_model']
max_retries = int(config['retry_settings']['retries'])
min_delay = int(config['retry_settings']['delay'])


openai.api_key = os.environ["OPENAI_API_KEY"]

# Cache dictionary to store previously generated results
cache = {}

# Initialize OpenAI client
def create_openai_client() -> str:
    """Takes input and create openai client 

    Returns:
        str: client
    """
    client = openai.OpenAI()

    return client

class ImageInput(BaseModel):
    """
    Model representing the input data for the FastAPI application for image route.

    Args:
        BaseModel (BaseModel): Pydantic base model class.
    
    Attributes:

    """
    Topic: str
    Output_Format: str
    Guidelines: dict
    Content: dict

class VariationInput(BaseModel):
    """
    Model representing the input data for the FastAPI application for variation route.

    Args:
        BaseModel (BaseModel): Pydantic base model class.
    """
    b64_json: str
    Output_Format: str

def generate_cache_key(input_data: ImageInput) -> str:
    """Generates a unique cache key based on input data."""
    input_str = json.dumps(input_data.model_dump(), sort_keys=True)
    return sha256(input_str.encode()).hexdigest()

@app.post("/generate_image", dependencies=[Depends(rate_limiter)])
async def generate_image(input_data: ImageInput):
    """Accepts input data and return image blob URL/base64 encoded image

    Args:
        input_data (InputData): JSON input

    Raises:
        HTTPException: Internal server error on exception

    Returns:
        image url: image url of an image that best describe the content along with revised prompt.
    """
    start_time = datetime.now()
    logger.info("POST /generate_image started at %s", start_time)

    # Generate cache key from input data
    cache_key = generate_cache_key(input_data)

    # Check if result is in cache
    if cache_key in cache:
        logger.info("Cache hit for input data")
        return cache[cache_key]
    try:
        client = create_openai_client()
        content_summary = summarize_content(content_json=input_data.Content,
                                            length=summary_length,
                                            client=client)
        image, revised_prompt = create_image(topic=input_data.Topic,
                                                 output_format=input_data.Output_Format,
                                                 content=content_summary,
                                                 guideline=input_data.Guidelines,
                                                 client=client)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("POST /generate_image ended at %s, duration: %s", end_time, duration)

        # Store the result in cache
        cache[cache_key] = {"image": image, "revised_prompt": revised_prompt}
        return cache[cache_key]

    except Exception as e:
        logger.error("Error generating Image: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@app.post("/generate_image_variation", dependencies=[Depends(rate_limiter)])
async def generate_image_variation(input_data: VariationInput):
    """Accepts input data and return variation of image blob URL/base64 encoded image

    Args:
        input_data (InputData): JSON input

    Raises:
        HTTPException: Internal server error on exception

    Returns:
        image :image that is variation of the supplied input image.
    """
    start_time = datetime.now()
    logger.info("POST /generate_image_variation started at %s", start_time)

    try:
        client = create_openai_client()
        image = create_image_variation(imagedata=input_data.b64_json,
                                                 output_format=input_data.Output_Format,
                                                 client=client)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("POST /generate_image_variation ended at %s, duration: %s", end_time, duration)

        return {"image": image}

    except Exception as e:
        logger.error("Error generating Image variation: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@retry(retries=max_retries, delay=min_delay, exceptions=(Exception,))
def summarize_content(content_json: str, length:int, client:str) -> str:
    """Take JSON content as input and provide a paragraph about main elements that need to be present in image 
    as per the content

    Args:
        content_json (str): content
        length (str): lenght of instructions in character generated as output
        client (str): client

    Returns:
        str: Instructions about main elements included in the image
    """
    try:
        formatted_prompt_summarize_content = PROMPT_IMAGE_ELEMENTS.format(content=content_json,
                                                                             summary_length=length)
        response = client.chat.completions.create(
            model=summary_llm,
            temperature= temperature,
            max_tokens= max_tokens,
            messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": SYSTEM_PROMPT
                }
            ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt_summarize_content},
                    ],
                }
            ],

        )
        logger.info("Total tokens for content summary call: %s", response.usage.total_tokens)
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error creating content summary: %s\n%s", str(e), traceback.format_exc())
        raise

@retry(retries=max_retries, delay=min_delay, exceptions=(Exception,))
def create_image(topic: str, output_format: str, content: str, guideline: str, client:str) -> str:
    """create image as per provided input instructions and guidelines

    Args:
        topic (str): topic
        output_format (str): base 64 or url
        content (str): instructions
        guideline (str): image guidelines
        client (str): client

    Returns:
        str: Image URL/base64 version
    """
    try:
        formatted_prompt_image = PROMPT_IMAGE.format(
            topic=topic, content=content , guideline=guideline)
        result = client.images.generate(
            model=image_model,
            prompt=formatted_prompt_image,
            size= image_size,
            n=number,
            response_format = output_format
        )

        image = json.loads(result.model_dump_json())['data'][0][output_format]
        revised_prompt = json.loads(result.model_dump_json())['data'][0]['revised_prompt']
        logger.info("Revised prompt for generated image is : %s",revised_prompt)
        return image, revised_prompt
    except Exception as e:
        logger.error("Error creating Image: %s\n%s", str(e), traceback.format_exc())
        raise

@retry(retries=max_retries, delay=min_delay, exceptions=(Exception,))
def create_image_variation(imagedata: str, output_format: str, client:str) -> str:
    """
    Takes a PNG image format and create a variation of this image

    Args:
        imagedata (str): input image
        output_format (str): base64 or URL
        client (str): client
    Returns:
        str: Image that is variation of supplied input
    """

    try:
        imagedata1 = b64decode(imagedata)
        result = client.images.create_variation(
            image=imagedata1,
            n=number,
            size=image_size,
            response_format=output_format,

        )
        image = json.loads(result.model_dump_json())['data'][0][output_format]
        return image
    except Exception as e:
        logger.error("Error creating Image variation: %s\n%s", str(e), traceback.format_exc())
        raise

# Root endpoint for health check
@app.get("/", dependencies=[Depends(rate_limiter)])
async def root():
    """Check the API GET function."""
    return {"message": "Welcome to fastAPI!"}