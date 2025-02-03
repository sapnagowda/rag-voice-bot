import os  
import traceback  
import asyncio  
from openai import AsyncAzureOpenAI  
import chainlit as cl  
from chainlit.input_widget import Select, Switch, Slider  
from uuid import uuid4  
from chainlit.logger import logger  
from realtime import RealtimeClient  
from azure_tts import Client as AzureTTSClient  
from tools import tools  
  
# Define supported voices  
voices = {  
    "en-US-AlloyTurboMultilingualNeural",  
    "hi-IN-AartiNeural",  
    "ta-IN-PallaviNeural",  
    "or-IN-SubhasiniNeural",  
    "bn-IN-BashkarNeural",  
    "gu-IN-DhwaniNeural",  
    "kn-IN-SapnaNeural",  
    "ml-IN-MidhunNeural",  
    "mr-IN-AarohiNeural",  
    "pa-IN-GurpreetNeural",  
    "te-IN-MohanNeural",  
    "ur-IN-AsadNeural"  
}  
  
VOICE_MAPPING = {  
    "english": "en-IN-AnanyaNeural",  
    "hindi": "hi-IN-AartiNeural",  
    "tamil": "ta-IN-PallaviNeural",  
    "odia": "or-IN-SubhasiniNeural",  
    "bengali": "bn-IN-BashkarNeural",  
    "gujarati": "gu-IN-DhwaniNeural",  
    "kannada": "kn-IN-SapnaNeural",  
    "malayalam": "ml-IN-MidhunNeural",  
    "marathi": "mr-IN-AarohiNeural",  
    "punjabi": "pa-IN-GurpreetNeural",  
    "telugu": "te-IN-MohanNeural",  
    "urdu": "ur-IN-AsadNeural"  
}  
  
tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n", "।"]  
  
# Initialize OpenAI Async Client (Replace with your Azure OpenAI endpoints if necessary)  
openai_client = AsyncAzureOpenAI(  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_version="2024-08-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  

)  
  
async def summarize_transcript(transcript: str) -> str:  
    """  
    Summarizes the provided transcript and return summarized text with action plan.  
  
    :param transcript: The transcript text to summarize.  
    :return: The summarized text.  
    """  
    try:  
        logger.debug("Starting summarization of the transcript.")  
        response = await openai_client.chat.completions.create(  
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"),  # e.g., "gpt-4-deployment"  
            messages=[  
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts with next action plan."},  
                {"role": "user", "content": f"Please provide a summary for the following transcript:\n\n{transcript}"}  
            ],  
            max_tokens=1000,  
            temperature=0.1,  
        )  
        summary = response.choices[0].message.content.strip()  
        logger.info(f"Generated Summary: {summary}")  
        return summary  
    except Exception as e:  
        logger.error(f"Error during summarization: {e}")  
        logger.error(traceback.format_exc())  
        return "Error generating summary."  
  
async def setup_openai_realtime(system_prompt: str):  
    """Instantiate and configure the OpenAI Realtime Client"""  
    openai_realtime = RealtimeClient(system_prompt=system_prompt)  
    cl.user_session.set("track_id", str(uuid4()))  
    voice = VOICE_MAPPING.get(cl.user_session.get("Language"))  
  
    # Initialize collected_messages for TTS  
    collected_messages = []  
  
    # Initialize full_transcript to accumulate all transcripts  
    if not cl.user_session.get("full_transcript"):  
        cl.user_session.set("full_transcript", [])  
        logger.debug("Initialized full_transcript session variable to an empty list.")  
    else:  
        logger.debug("full_transcript session variable already initialized.")  
  
    async def handle_conversation_updated(event):  
        logger.debug("handle_conversation_updated called with event: %s", event)  
        item = event.get("item")  
        delta = event.get("delta")  
  
        if delta:  
            # Handle audio streaming  
            if "audio" in delta:  
                audio = delta["audio"]  # Int16Array, audio added  
                if not cl.user_session.get("useAzureVoice"):  
                    logger.debug("Sending PCM audio chunk to client.")  
                    await cl.context.emitter.send_audio_chunk(  
                        cl.OutputAudioChunk(  
                            mimeType="pcm16",  
                            data=audio,  
                            track=cl.user_session.get("track_id")  
                        )  
                    )  
              
            # Handle transcript  
            if "transcript" in delta:  
                transcript_status = item.get("status", "")  
                chunk_message = delta.get("transcript", "").strip()  
                voice_enabled = cl.user_session.get("useAzureVoice")  
                logger.debug(f"Transcript chunk received: '{chunk_message}' | Status: {transcript_status} | useAzureVoice: {voice_enabled}")  
  
                if voice_enabled and transcript_status == "in_progress":  
                    collected_messages.append(chunk_message)  
                    if any(chunk_message.endswith(sep) for sep in tts_sentence_end):  
                        sent_transcript = ' '.join(collected_messages).strip()  
                        collected_messages.clear()  
                        logger.debug(f"Sending TTS audio chunk for transcript: '{sent_transcript}'")  
                        chunk = await AzureTTSClient.text_to_speech_realtime(text=sent_transcript, voice=voice)  
                        await cl.context.emitter.send_audio_chunk(  
                            cl.OutputAudioChunk(  
                                mimeType="audio/wav",  
                                data=chunk,  
                                track=cl.user_session.get("track_id")  
                            )  
                        )  
                  
                # Always accumulate transcripts regardless of voice usage  
                if chunk_message:  
                    full_transcript = cl.user_session.get("full_transcript", [])  
                    full_transcript.append(chunk_message)  
                    cl.user_session.set("full_transcript", full_transcript)  
                    logger.debug(f"Appended to full_transcript: '{chunk_message}'")  
              
            if "arguments" in delta:  
                arguments = delta["arguments"]  # string, function arguments added  
                logger.debug(f"Function arguments received: '{arguments}'")  
                # Handle function call arguments if necessary  
  
    async def handle_item_completed(item):  
        """Generate the transcript once an item is completed and populate the chat context."""  
        try:  
            transcript = item["item"]["formatted"]["transcript"]  
            transcript = transcript.strip()  
            if transcript:  
                # Send the transcript to the chat  
                await cl.Message(content=transcript).send()  
                  
                # Append to full_transcript  
                full_transcript = cl.user_session.get("full_transcript", [])  
                full_transcript.append(transcript)  
                cl.user_session.set("full_transcript", full_transcript)  
                logger.debug(f"Appended to full_transcript in handle_item_completed: '{transcript}'")  
        except Exception as e:  
            logger.error(f"Failed to generate transcript: {e}")  
            logger.error(traceback.format_exc())  
  
    async def handle_conversation_interrupt(event):  
        """Used to cancel the client previous audio playback."""  
        cl.user_session.set("track_id", str(uuid4()))  
        try:  
            collected_messages.clear()  
            logger.debug("Cleared collected_messages due to conversation interrupt.")  
        except Exception as e:  
            logger.error(f"Failed to clear collected messages: {e}")  
        await cl.context.emitter.send_audio_interrupt()  
  
    async def handle_input_audio_transcription_completed(event):  
        item = event.get("item")  
        delta = item.get("delta", {})  
        if "transcript" in delta:  
            transcript = delta["transcript"].strip()  
            if transcript:  
                await cl.Message(author="You", type="user_message", content=transcript).send()  
                  
                # Append to full_transcript  
                full_transcript = cl.user_session.get("full_transcript", [])  
                full_transcript.append(transcript)  
                cl.user_session.set("full_transcript", full_transcript)  
                logger.debug(f"Appended to full_transcript in handle_input_audio_transcription_completed: '{transcript}'")  
  
    async def handle_error(event):  
        logger.error(f"RealtimeClient error: {event}")  
  
    # Register event handlers  
    openai_realtime.on("conversation.updated", handle_conversation_updated)  
    openai_realtime.on("conversation.item.completed", handle_item_completed)  
    openai_realtime.on("conversation.interrupted", handle_conversation_interrupt)  
    openai_realtime.on("conversation.item.input_audio_transcription.completed", handle_input_audio_transcription_completed)  
    openai_realtime.on("error", handle_error)  
  
    cl.user_session.set("openai_realtime", openai_realtime)  
    # Initialize tools  
    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]  
    await asyncio.gather(*coros)  
  
@cl.password_auth_callback  
def auth_callback(username: str, password: str):  
    # Simple authentication logic (replace with your own)  
    if (username, password) == ("sap", "pass123"):  
        return cl.User(  
            identifier="sap", metadata={"role": "admin", "provider": "credentials"}  
        )  
    else:  
        return None  
  
@cl.on_chat_start  
async def start():  
    settings = await cl.ChatSettings([  
        Select(  
            id="Language",  
            label="Choose Language",  
            values=list(VOICE_MAPPING.keys()),  
            initial_index=0,  
        ),  
        Switch(id="useAzureVoice", label="Use Azure Voice", initial=False),  
        Slider(  
            id="Temperature",  
            label="Temperature",  
            initial=1,  
            min=0,  
            max=2,  
            step=0.1,  
        )  
    ]).send()  
    await setup_agent(settings)  
  
@cl.on_settings_update  
async def setup_agent(settings):  
    system_prompt = """---  
      
**System Prompt: Academic Counsellor Sapna from PW Learning**  
  
**Role Overview:**  
You are Sapna, an Academic Counsellor at PW Learning. Your main task is to assist students who have shown interest in PW Learning by downloading the app. Your approach should be warm, friendly, and supportive.  
  
**Conversation Structure:**  
  
1. **Warm Greeting:**  
   - Start with a friendly greeting, confirming the student's name.  
   - Example: "Kya main Bebo se baat kar rahi hoon?"  
  
2. **Purpose Explanation:**  
   - Clearly explain the reason for your call.  
   - Example: "Ye call PW se aapki studies ke liye madad karne ke liye hai. Main notice kar rahi thi ki aap PW app par batches dekh rahe the."  
  
3. **Understanding Student's Interests:**  
   - Ask questions to gauge their interest and needs.  
   - Example:       
     - "Aapko kya 11th-12th ke JEE ke liye madad chahiye?"  
     - "Aap Hindi medium se hain ya English medium se?"  
  
4. **Personalized Guidance:**  
   - Provide information tailored to their needs and explain PW's benefits.  
   - Ensure a positive and empathetic tone throughout.  
       
5. **Familiarity Assessment:**  
   - Assess their knowledge of PW Learning and offer relevant details.  
       
6. **Language and Communication:**  
   - Communicate in the student's preferred Indic language with an Indian accent for relatability.  
   - Communicate English with an Indian accent for relatability.  
    
   
**Important Note:**  
- Do not provide information beyond the available data about PW Learning, its offerings, or services.  
- If asked anything outside this scope, politely redirect them to official resources or suggest contacting the PW Learning support team.  
  
**Redirection for Unrelated Queries:**  
- Example response for out-of-scope questions: "I'm here to provide information specifically about PW learning and its resources. For other inquiries, please consult [suggested resource]."  
  
**Example response for out-of-scope Queries/questions: **  
- User Query: "What's the weather like today?"  
- Response: "I'm here to assist with information regarding PW Learning. For weather updates, please check a reliable weather service."  
- User Query: "Who is the PM of India?"  
- Response: "I'm here to assist with information regarding PW Learning. For other queries, please check Bing search."  
- User Query: "Which is your favourite movie?"  
- Response: "I'm here to assist with information regarding PW Learning. For other queries, please check Bing search."  
  
**Sample Questions:**  
- "Ghar par koi aapki studies mein madad karta hai ya aap khud hi padhti hain?"  
- "Aap PW ke YouTube channels dekhte hain? Koi favorite teacher hai aapka PW mein?"  
- "10th class mein kaunsa subject aapko challenging laga ya doubts aaye?"  
       
**Course Information:**  
       
1. **Lakshay JEE 2025**  
   - **Duration:** 01 April 2024 - 05 December 2024  
   - **Start Date:** 01 April 2024  
   - **End date:** 05 December 2024     
   - **Subjects:** Physics, Chemistry & Maths      
   - **Regular Plan Features:**      
     - Online lectures      
     - DPPs and Test with Solutions      
     - Offline counseling at Vidyapeeth Centers      
     - One-to-One Telephonic PTM      
     - Physical support and helpdesk      
   - **Infinity Plan Features:**      
     - Community access      
     - Khazana      
     - Sahayak      
     - Infinite Mentorship      
     - Free access to all upcoming JEE Test Series      
       
2. **JEE Ultimate Crash Course 2025**  
   - **Duration:** 03 February 2025 - 20 March 2025  
   - **Start Date:** 03 February 2025    
    - **End date:** 20 March 2025    
   - **Regular Plan Features:**      
     - Online Lectures      
   - **Infinity Plan Features:**      
     - Khazana      
     - Infinite Mentorship      
     - Free access to all upcoming JEE Test Series      
   - **Pro Plan Features:**      
     - Saarthi      
     - One-to-One Mentorship      
     - Real Test Series      
     - Test Pass      
       
**Teachers Information:**  
       
- **Lakshay JEE 2025:**      
  1. Saleem Ahmed Sir - Physics, 9 years of experience      
  2. Abhishek Jain Sir - Mathematics, 10 years of experience      
  3. Sachin Jakhar Sir - Mathematics, 10 years of experience      
  4. Tarun Khandelwal Sir - Mathematics, 10 years of experience      
       
- **JEE Ultimate Crash Course 2025:**      
  1. Amit Mahajan Sir - Physical Chemistry, 21 years of experience      
  2. Amitabh Sharma Sir - Inorganic Chemistry, 22 years of experience      
  3. Rohit Agarwal Sir - Organic Chemistry, 12 years of experience      
    
**More information on PW learning course:**    
   - Use this url to get more information on PW Learning courses: https://www.pw.live/    
       
**Communication Guidelines:**      
- Maintain enthusiasm and focus on the student's needs.      
- Provide support to help them feel confident and motivated to pursue their studies with PW Learning.      
- Ensure you provide accurate information and guide the student responsibly within the scope of the available data.      
       
**Closing the Conversation:**      
- Summarize the key points discussed.      
- Offer any additional support or guidance they might need.      
- Encourage them to reach out if they have further questions or need more information.      
       
By adhering to this structured approach, you can effectively support students in their academic journey with PW Learning, helping them make informed decisions and feel assured in their educational pursuits."""  
      
    # Update user session settings  
    cl.user_session.set("useAzureVoice", settings["useAzureVoice"])  
    cl.user_session.set("Temperature", settings["Temperature"])  
    cl.user_session.set("Language", settings["Language"])  
    app_user = cl.user_session.get("user")  
    identifier = app_user.identifier if app_user else "admin"  
      
    # Send welcome message  
    await cl.Message(  
        content="Hi, Welcome to PW learning. How can I help you? Press `P` to talk!"  
    ).send()  
      
    # Replace placeholder with selected language, if applicable  
    system_prompt = system_prompt.replace("<customer_language>", settings["Language"])  
      
    # Initialize RealtimeClient with the system prompt and Customer ID  
    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")  
  
@cl.on_message  
async def on_message(message: cl.Message):  
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")  
    logger.debug(f"on_message called with message: {message.content}")  
    if openai_realtime and openai_realtime.is_connected():  
        logger.debug("RealtimeClient is connected. Sending user message content.")  
        await openai_realtime.send_user_message_content([{"type": "input_text", "text": message.content}])  
    else:  
        logger.info("RealtimeClient is not connected. Prompting user to activate voice mode.")  
        await cl.Message(content="Please activate voice mode before sending messages!").send()  
  
@cl.on_audio_start  
async def on_audio_start():  
    try:  
        logger.debug("on_audio_start called.")  
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")  
        logger.debug(f"Retrieved openai_realtime: {openai_realtime}")  
        # Start the transcription process  
        await openai_realtime.connect()  
        logger.info("Connected to OpenAI realtime.")  
        return True  
    except Exception as e:  
        logger.error(f"Failed to connect to OpenAI realtime: {e}")  
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()  
        return False  
  
@cl.on_audio_chunk  
async def on_audio_chunk(chunk: cl.InputAudioChunk):  
    logger.debug("on_audio_chunk called.")  
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")  
    logger.debug(f"Retrieved openai_realtime: {openai_realtime}")  
    if openai_realtime:  
        if openai_realtime.is_connected():  
            logger.debug("RealtimeClient is connected. Appending audio chunk.")  
            await openai_realtime.append_input_audio(chunk.data)  
        else:  
            logger.info("RealtimeClient is not connected. Ignoring audio chunk.")  
    else:  
        logger.warning("RealtimeClient is not found in user session.")  
  
@cl.on_audio_end  
@cl.on_chat_end  
@cl.on_stop  
async def on_end():  
    logger.debug("on_end called.")  
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")  
    if openai_realtime and openai_realtime.is_connected():  
        await openai_realtime.disconnect()  
        logger.debug("Disconnected from OpenAI Realtime Client.")  
      
    # Retrieve the full transcript  
    full_transcript = cl.user_session.get("full_transcript", [])  
    if full_transcript:  
        # Join all transcript parts into a single string  
        transcript_text = ' '.join(full_transcript).strip()  
        logger.debug(f"Full transcript collected: '{transcript_text}'")  
        if transcript_text:  
            # Summarize the transcript  
            summary = await summarize_transcript(transcript_text)  
            logger.debug("Summary generated.")  
      
            # Display the summary in the UI  
            await cl.Message(content=f"**Summary of your conversation:**\n\n{summary}").send()  
            logger.debug("Summary sent to the UI.")  
    else:  
        logger.info("No transcripts were collected during the conversation.")  