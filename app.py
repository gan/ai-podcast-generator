import os
import time
import torch
import subprocess
import torchaudio
import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
import ChatTTS.ChatTTS as ChatTTS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

st.title("🎙️AI播客生成器🎙️")

if "characters_and_topics_submitted" not in st.session_state or st.sidebar.button("重启播客生成器"):
    st.session_state["messages"] = []
    st.session_state["characters_and_topics_submitted"] = False
    st.session_state["character_persona_submitted"] = False
    st.rerun()


class PodcastScript(BaseModel):
    dialogs: List[dict] = Field(...,
                                description="Contains dictionaries with these key values: speaker, content and the dialog_counter. speaker: name of the speaker, content: content of the speech, dialog_counter: The number of the dialog. Should be incremented by 1 for each dialog")


def create_podcast_template():
    character_personas = "".join(
        [f"\n{guest} 个性: {st.session_state[f'{guest}_persona']}" for guest in st.session_state["guests"]]
    )
    # 添加主持人的性格
    host_character = st.session_state["host_character"]
    host_persona = f"- {host_character} 个性: {st.session_state[f'{host_character}_persona']}"

    guest_introductions = ", ".join(st.session_state["guests"])

    podcast_template = f"""## 播客大纲
                        这是一个由 {st.session_state["host_character"]}主持的播客节目，主持人的性格特征是：{host_persona}。
                        这期节目还有{guest_introductions} 参与。
                        嘉宾的角色人设:
                        {character_personas}
                        播客内容:
                        {st.session_state["podcast_topic"]}
                        """

    return podcast_template


def generate_dialog(number_of_dialogs, timestamp, debug=True):
    podcast_template = create_podcast_template()
    instructions = f"""Instructions:
                    - 用中文写对话。
                    - 播客应包含超过 {number_of_dialogs}次对话。一定要包括播客的结束对话.
                    - 不要使用非语言提示，如 *laughs* or *ahem* 或括号中的内容. 使用 哈哈 而不是 *laughs*.
                    - 也不要在内容参数中使用说话者标签，但应始终为正确的说话者设置speaker键."""

    st.write(podcast_template)

    os.makedirs('podcasts', exist_ok=True)
    transcript_file_name = f"podcasts/podcast{timestamp}.txt"

    podcast_assistant = Assistant(
        llm=OpenAIChat(model="gpt-4o", api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, max_tokens=32768),
        #llm=Ollama(model="llama3"),
        description="你是一个中文播客内容写手",
        output_model=PodcastScript
    )
    print(f"生成博客的大纲: {podcast_template} {instructions}\n")

    result = podcast_assistant.run(f"生成博客的大纲: {podcast_template} {instructions}")

    dialogs = result.dialogs
    print(dialogs)

    with open(transcript_file_name, "w") as transcript_file:
        for dialog in dialogs:
            transcript_line = f"{dialog['speaker']} says: {dialog['content']}\n"
            transcript_file.write(transcript_line)

    return dialogs


def generate_audio(dialogs, timestamp):
    # 初始化并加载ChatTTS模型
    chat = ChatTTS.Chat()
    chat.load_models(compile=False)  # Set to True for better performance

    # 定义不同角色对应的声音
    all_possible_speakers = {"Sam", "Jensen", "马老师", "老刘", "李博士", "吴医生"}
    voice_embeddings = {speaker: chat.sample_random_speaker() for speaker in all_possible_speakers}
    dialog_files = []
    os.makedirs('dialogs', exist_ok=True)

    with open("concat.txt", "w") as concat_file:
        for i, dialog in enumerate(dialogs):
            filename = f"dialogs/dialog{i}.wav"
            texts = [dialog["content"]]
            spk_emb = voice_embeddings[dialog["speaker"]]
            params_infer_code = {
                'spk_emb': spk_emb,
                'temperature': 0.1,
                'top_P': 0.7,
                'top_K': 20,
            }
            wavs = chat.infer(texts, params_infer_code=params_infer_code)
            torchaudio.save(filename, torch.from_numpy(wavs[0]), 24000)
            concat_file.write(f"file {filename}\n")
            dialog_files.append(filename)

    podcast_file = f"podcasts/podcast{timestamp}.wav"

    print("Concatenating audio")
    subprocess.run(f"ffmpeg -f concat -safe 0 -i concat.txt -c copy {podcast_file}", shell=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)

    os.remove("concat.txt")
    for file in dialog_files:
        os.unlink(file)

    st.audio(podcast_file, format='audio/wav')


def generate_podcast():
    current_time = time.time()
    with st.spinner("📜 正在生成播客内容..."):
        dialogs = generate_dialog(st.session_state["dialog_count"], current_time, st.session_state["podcast_topic"])
    st.write("播客内容生成成功")
    with st.spinner("🎤 正在生成音频..."):
        generate_audio(dialogs, current_time)


if not st.session_state["characters_and_topics_submitted"]:
    with st.form("characters_and_topics"):
        st.selectbox(
            "选择主持人",
            ("Sam", "Jensen"),
            key="host_character"
        )
        st.multiselect(
            "选择参与播客的人",
            ["马老师", "老刘", "李博士", "吴医生"],
            key="guests"
        )
        st.text_area(
            "输入播客内容",
            key="podcast_topic"
        )
        st.slider(
            "对话数量", 2, 30, 12,
            key="dialog_count"
        )
        st.session_state["characters_and_topics_submitted"] = st.form_submit_button("Submit")

if st.session_state["characters_and_topics_submitted"]:
    with st.form("character_persona"):
        st.text_area(
            f"输入{st.session_state['host_character']}的性格特征",
            key=f"{st.session_state['host_character']}_persona",
            placeholder="乐观、幽默、善于沟通"
        )
        for guest in st.session_state["guests"]:
            st.text_area(
                f"输入{guest}的性格特征",
                key=f"{guest}_persona",
                placeholder="严肃、专业、有趣"
            )
        st.session_state["character_persona_submitted"] = st.form_submit_button("Submit")

if st.session_state["character_persona_submitted"]:
    generate_podcast()
    st.write("Podcast generated successfully")
