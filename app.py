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
voice = {"en-US-AlloyTurboMultilingualNeural",
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
        "ur-IN-AsadNeural"}

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


tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n", "।"]
async def setup_openai_realtime(system_prompt: str):
    """Instantiate and configure the OpenAI Realtime Client"""
    openai_realtime = RealtimeClient(system_prompt = system_prompt)
    cl.user_session.set("track_id", str(uuid4()))
    voice = VOICE_MAPPING.get(cl.user_session.get("Language"))
    collected_messages = []
    async def handle_conversation_updated(event):
        item = event.get("item")
        delta = event.get("delta")
        
        """Currently used to stream audio back to the client."""
        if delta:
            # Only one of the following will be populated for any given event
            if 'audio' in delta:
                audio = delta['audio']  # Int16Array, audio added
                if not cl.user_session.get("useAzureVoice"):
                    await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id")))
            if 'transcript' in delta:
                if cl.user_session.get("useAzureVoice"):
                    chunk_message = delta['transcript']
                    if item["status"] == "in_progress":
                        collected_messages.append(chunk_message)  # save the message
                        if chunk_message in tts_sentence_end: # sentence end found
                            sent_transcript = ''.join(collected_messages).strip()
                            collected_messages.clear()
                            chunk = await AzureTTSClient.text_to_speech_realtime(text=sent_transcript, voice= voice)
                            await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="audio/wav", data=chunk, track=cl.user_session.get("track_id")))
            if 'arguments' in delta:
                arguments = delta['arguments']  # string, function arguments added
                pass
    
    async def handle_item_completed(item):
        """Generate the transcript once an item is completed and populate the chat context."""
        try:
            transcript = item['item']['formatted']['transcript']
            if transcript.strip() != "":
                await cl.Message(content=transcript).send()      
                
        except Exception as e:
            logger.error(f"Failed to generate transcript: {e}")
            logger.error(traceback.format_exc())
    
    async def handle_conversation_interrupt(event):
        """Used to cancel the client previous audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        try:
            collected_messages.clear()
        except Exception as e:
            logger.error(f"Failed to clear collected messages: {e}")    
        await cl.context.emitter.send_audio_interrupt()
        
    async def handle_input_audio_transcription_completed(event):
        item = event.get("item")
        delta = event.get("delta")
        if 'transcript' in delta:
            transcript = delta['transcript']
            if transcript != "":
                await cl.Message(author="You", type="user_message", content=transcript).send()
        
    async def handle_error(event):
        logger.error(event)
        
    
    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('conversation.item.input_audio_transcription.completed', handle_input_audio_transcription_completed)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    #cl.user_session.set("tts_client", tts_client)
    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]
    await asyncio.gather(*coros)
    

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
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
- Example response for out-of-scope questions: "I'm here to provide information specifically about PW learning and its resources. For other inquiries, I recommend consulting [suggested resource]."

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


    cl.user_session.set("useAzureVoice", settings["useAzureVoice"])
    cl.user_session.set("Temperature", settings["Temperature"])
    cl.user_session.set("Language", settings["Language"])
    app_user = cl.user_session.get("user")
    identifier = app_user.identifier if app_user else "admin"
    await cl.Message(
        content="Hi, Welcome to PhysicsWallah. How can I help you?. Press `P` to talk!"
    ).send()
    system_prompt = system_prompt.replace("<customer_language>", settings["Language"])
    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")
    
@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    print(f"Debug: Retrieved openai_realtime: {openai_realtime}")
    if openai_realtime and openai_realtime.is_connected():
        print("Debug: Retrieved openai_realtime: ...inside...")
        await openai_realtime.send_user_message_content([{ "type": 'input_text', "text": message.content}])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()

@cl.on_audio_start
async def on_audio_start():
    try:
        print("Debug: on_audio_start called...........")
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        print(f"Debug: Retrieved openai_realtime: {openai_realtime}")
        # TODO: might want to recreate items to restore context
        # openai_realtime.create_conversation_item(item)
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    print("Debug: on_audio_chunk called")
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    print(f"Debug: Retrieved openai_realtime: {openai_realtime}")
    if openai_realtime:            
        if openai_realtime.is_connected():
            print("Debug: RealtimeClient is connected")
            await openai_realtime.append_input_audio(chunk.data)
        else:
            print("Debug: RealtimeClient is not connected")
            logger.info("RealtimeClient is not connected")

@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()
