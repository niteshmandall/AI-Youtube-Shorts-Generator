import google.generativeai as genai
from dotenv import load_dotenv
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"


import json

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Make sure it is defined in the .env file.")

genai.configure(api_key=GEMINI_API_KEY)


# Function to extract start and end times
def extract_times(json_string):
    try:
        # Parse the JSON string
        data = json.loads(json_string)

        # Extract start and end times as floats
        start_time = float(data[0]["start"])
        end_time = float(data[0]["end"])

        # Convert to integers
        start_time_int = int(start_time)
        end_time_int = int(end_time)
        return start_time_int, end_time_int
    except Exception as e:
        print(f"Error in extract_times: {e}")
        return 0, 0


system = """

Baised on the Transcription user provides with start and end, Highilight the main parts in less then 1 min which can be directly converted into a short. highlight it such that its intresting and also keep the time staps for the clip to start and end. only select a continues Part of the video

Follow this Format and return in valid json 
[{
start: "Start time of the clip",
content: "Highlight Text",
end: "End Time for the highlighted clip"
}]
it should be one continues clip as it will then be cut from the video and uploaded as a tiktok video. so only have one start, end and content

Dont say anything else, just return Proper Json. no explanation etc


IF YOU DONT HAVE ONE start AND end WHICH IS FOR THE LENGTH OF THE ENTIRE HIGHLIGHT, THEN 10 KITTENS WILL DIE, I WILL DO JSON['start'] AND IF IT DOESNT WORK THEN...
"""

User = """
Any Example
"""


def GetHighlight(Transcription):
    print("Getting Highlight from Transcription ")
    try:

        # Using gemini-2.0-flash which has better rate limits for free tier
        model = genai.GenerativeModel('gemini-2.0-flash')
        try:
            response = model.generate_content(
                f"""{system}
                
                Transcription:
                {Transcription}""",
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
            )
        except Exception as e:
            if "quota" in str(e).lower() or "rate_limit" in str(e).lower():
                print("⚠️  Rate limit reached. Please wait a few minutes and try again.")
                print("💡 Tip: Consider upgrading your Google Cloud plan for higher limits.")
                return 0, 0
            raise
        
        json_string = response.text
        json_string = json_string.replace("json", "")
        json_string = json_string.replace("```", "")
        # print(json_string)
        Start, End = extract_times(json_string)
        if Start == End:
            Ask = input("Error - Get Highlights again (y/n) -> ").lower()
            if Ask == "y":
                Start, End = GetHighlight(Transcription)
        return Start, End
    except Exception as e:
        print(f"Error in GetHighlight: {e}")
        return 0, 0


if __name__ == "__main__":
    print(GetHighlight(User))
