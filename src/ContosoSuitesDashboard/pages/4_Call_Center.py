import json
import time
import re
import uuid
import streamlit as st
from scipy.io import wavfile
import azure.cognitiveservices.speech as speechsdk

st.set_page_config(layout="wide")

@st.cache_data
def create_transcription_request(audio_file, speech_recognition_language="en-US"):
    """Transcribe the contents of an audio file. Key assumptions:
    - The audio file is in WAV format.
    - The audio file is mono.
    - The audio file has a sample rate of 16 kHz.
    - Speech key and region are stored in Streamlit secrets."""
    
    speech_key = st.secrets["speech"]["key"]
    speech_region = st.secrets["speech"]["region"]
    
    # Create an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = speech_recognition_language
    
    # Prepare audio settings for the wave stream
    channels = 1
    bits_per_sample = 16
    samples_per_second = 16000
    
    # Create audio configuration using the push stream
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second, bits_per_sample, channels)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config, audio_config)
    
    all_results = []
    done = False
    
    def handle_final_result(evt):
        all_results.append(evt.result.text)
    
    def stop_cb(evt):
        print(f'CLOSING on {evt}')
        nonlocal done
        done = True
    
    # Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(handle_final_result)
    transcriber.session_started.connect(lambda evt: print(f'SESSION STARTED: {evt}'))
    transcriber.session_stopped.connect(lambda evt: print(f'SESSION STOPPED {evt}'))
    transcriber.canceled.connect(lambda evt: print(f'CANCELED {evt}'))
    
    # Stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)
    
    transcriber.start_transcribing_async()
    
    # Read the whole wave files at once and stream it to sdk
    _, wav_data = wavfile.read(audio_file)
    stream.write(wav_data.tobytes())
    stream.close()
    
    while not done:
        time.sleep(.5)
    
    transcriber.stop_transcribing_async()
    
    return all_results

def main():
    """Main function for the call center dashboard."""
    call_contents = []
    st.write(
        """
        # Call Center
        This Streamlit dashboard is intended to replicate some of the functionality
        of a call center monitoring solution. It is not intended to be a
        production-ready application.
        """
    )
    st.write("## Upload a Call")
    uploaded_file = st.file_uploader("Upload an audio file", type="wav")
    if uploaded_file is not None and ('file_transcription_results' not in st.session_state):
        st.session_state.file_transcription_results = create_transcription_request(uploaded_file)
        st.success("Transcription complete!")
    if 'file_transcription_results' in st.session_state:
        st.write(st.session_state.file_transcription_results)

if __name__ == "__main__":
    main()
