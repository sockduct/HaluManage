import os
import litellm
from litellm import completion
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
 
# 1. Define the Query (Simulated FRAMES Dataset Query)
# A multi-hop, challenging question, as defined by the FRAMES paper methodology.
FRAME_QUERY = (
    "The inventor of the first successful self-propelled car, born in a town whose name is "
    "an anagram of 'BLAST', also founded a major industrial company. What is the **current** "
    "stock ticker symbol of the conglomerate that company became after a series of mergers?"
)

# 2. Define the Base Gemini Model (as per your prompt)
BASE_MODEL = "gemini-1.5-flash-001"
# Working - BASE_MODEL = "gemini/gemini-2.5-flash-lite"

# 3. Define HaluManage Models (Technique + Base Model)
# HaluManage uses the LiteLLM interface, where the model string is prefixed with the technique.
HALUMANAGE_MODELS = { # The model name needs to be prefixed with "custom_openai/" for LiteLLM to use the base_url.
    "MOA (Mixture of Agents)": f"moa-{BASE_MODEL}"
}

 #   "MARS (Multi-Agent Reasoning)": f"mars-{BASE_MODEL}",
 #   "CePO (Cerebras Planning & Optimization)": f"cepo-{BASE_MODEL}",

print(f"--- FRAME Query ---\n{FRAME_QUERY}\n")

# 4. Loop through techniques and call the API via the HaluManage proxy
for technique, model_name in HALUMANAGE_MODELS.items():
    print(f"\n=======================================================")
    print(f"INVOKING MODEL: {model_name} (Technique: {technique})")
    print(f"=======================================================")
    
    try:
        # The completion() call is routed through the HaluManage proxy.
        # HaluManage processes the request using the specified technique (e.g., 'moa', 'mars').
        # We must provide the `base_url` to point to the proxy and a dummy `api_key`.
        response = completion(
            model=f"litellm_proxy/{model_name}",
            messages=[
                {"role": "user", "content": FRAME_QUERY}
            ],
            api_key="halumanage",  # A placeholder key is sufficient for the proxy
            base_url="http://localhost:8000/v1", # Point to your local HaluManage server
            # Use lower temperature for factual/reasoning tasks, aligning with mitigation best practices
            temperature=0.1
        )
        
        # Extract the final answer
        final_answer = response.choices[0].message.content
        
        print(f"\n[HaluManage Response for {technique}]:\n{final_answer}")
        
    except Exception as e:
        print(f"An error occurred with {technique}: {e}")
        print("Ensure your GEMINI_API_KEY is valid and the OptiLLM service is running and accessible.")