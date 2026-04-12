import os
import random
import re

import pandas as pd
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import streamlit as st

TOKEN_START_FACE = "<START_FACE>"
TOKEN_START_GENERATION = "<START_GENERATION>"
TOKEN_END_GENERATION = "<END_GENERATION>"
MAX_LEN = 300

selected_features = []
generated_codes = []
temperature = 0.9
top_k = 50
top_p = 0.95

gen_image_key = 'GEN_IMG'

GPT_MODEL_ID = "yosef-samy019/gpt-face-celeb-generator"
codeBook_path = os.path.join('vq_vae', 'celeb_face_vq_vae_codebook.csv')
decoder_path = os.path.join('vq_vae', 'celeb_face_vq_vae_decoder.onnx')


def build_ui():
    global selected_features, generated_codes
    global temperature, top_k, top_p

    st.set_page_config(
        page_title="AI Face Generator",
        page_icon="🧠",
        layout="wide"
    )

    # ======================
    # Hide Sidebar
    # ======================
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 GPT Face Generator")

    # ======================
    # Data
    # ======================
    FACE_ATTRIBUTES = {
        "Attractive": ["<NOT_ATTRACTIVE>", "<ATTRACTIVE>"],
        "Bags_Under_Eyes": ["<NO_EYE_BAGS>", "<EYE_BAGS>"],
        "Bald": ["<NOT_BALD>", "<BALD>"],
        "Bangs": ["<NO_BANGS>", "<BANGS>"],
        "Big_Lips": ["<SMALL_LIPS>", "<BIG_LIPS>"],
        "Big_Nose": ["<SMALL_NOSE>", "<BIG_NOSE>"],
        "Black_Hair": ["<NOT_BLACK_HAIR>", "<BLACK_HAIR>"],
        "Blond_Hair": ["<NOT_BLOND_HAIR>", "<BLOND_HAIR>"],
        "Brown_Hair": ["<NOT_BROWN_HAIR>", "<BROWN_HAIR>"],
        "Blurry": ["<CLEAR_IMAGE>", "<BLURRY_IMAGE>"],
        "Eyeglasses": ["<NO_GLASSES>", "<GLASSES>"],
        "Gray_Hair": ["<NO_GRAY_HAIR>", "<GRAY_HAIR>"],
        "Heavy_Makeup": ["<LIGHT_MAKEUP>", "<HEAVY_MAKEUP>"],
        "High_Cheekbones": ["<LOW_CHEEKBONES>", "<HIGH_CHEEKBONES>"],
        "Male": ["<FEMALE>", "<MALE>"],
        "Mouth_Slightly_Open": ["<MOUTH_CLOSED>", "<MOUTH_OPEN>"],
        "Mustache": ["<NO_MUSTACHE>", "<MUSTACHE>"],
        "Narrow_Eyes": ["<WIDE_EYES>", "<NARROW_EYES>"],
        "No_Beard": ["<BEARD>", "<NO_BEARD>"],
        "Oval_Face": ["<NON_OVAL_FACE>", "<OVAL_FACE>"],
        "Pointy_Nose": ["<ROUND_NOSE>", "<POINTY_NOSE>"],
        "Smiling": ["<NOT_SMILING>", "<SMILING>"],
        "Straight_Hair": ["<NOT_STRAIGHT_HAIR>", "<STRAIGHT_HAIR>"],
        "Wavy_Hair": ["<NOT_WAVY_HAIR>", "<WAVY_HAIR>"],
        "Wearing_Lipstick": ["<NO_LIPSTICK>", "<LIPSTICK>"],
        "Young": ["<OLD>", "<YOUNG>"]
    }
    DEFAULT_FACE = {
        "Attractive": "<ATTRACTIVE>",
        "Bags_Under_Eyes": "<NO_EYE_BAGS>",
        "Bald": "<NOT_BALD>",
        "Bangs": "<NO_BANGS>",
        "Big_Lips": "<SMALL_LIPS>",
        "Big_Nose": "<SMALL_NOSE>",
        "Black_Hair": "<BLACK_HAIR>",
        "Blond_Hair": "<NOT_BLOND_HAIR>",
        "Brown_Hair": "<NOT_BROWN_HAIR>",
        "Blurry": "<CLEAR_IMAGE>",
        "Eyeglasses": "<NO_GLASSES>",
        "Gray_Hair": "<NO_GRAY_HAIR>",
        "Heavy_Makeup": "<LIGHT_MAKEUP>",
        "High_Cheekbones": "<HIGH_CHEEKBONES>",
        "Male": "<MALE>",
        "Mouth_Slightly_Open": "<MOUTH_CLOSED>",
        "Mustache": "<NO_MUSTACHE>",
        "Narrow_Eyes": "<WIDE_EYES>",
        "No_Beard": "<BEARD>",
        "Oval_Face": "<OVAL_FACE>",
        "Pointy_Nose": "<ROUND_NOSE>",
        "Smiling": "<SMILING>",
        "Straight_Hair": "<STRAIGHT_HAIR>",
        "Wavy_Hair": "<NOT_WAVY_HAIR>",
        "Wearing_Lipstick": "<NO_LIPSTICK>",
        "Young": "<YOUNG>"
    }

    # ======================
    # Layout: 2 Columns
    # ======================
    left_col, right_col = st.columns([1.5, 1])

    with left_col:
        st.subheader("⚙️ Generation Settings")
        generation_cols = st.columns(3)
        temperature = generation_cols[0].slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.9,
            step=0.05
        )

        top_k = generation_cols[1].slider(
            "Top-K",
            min_value=1,
            max_value=200,
            value=50,
            step=1
        )

        top_p = generation_cols[2].slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01
        )

        st.subheader("🧍 Face Attributes")

        cols = st.columns(3)

        selected_features = []
        for i, (attr, options) in enumerate(FACE_ATTRIBUTES.items()):
            with cols[i % 3]:
                default_val = DEFAULT_FACE.get(attr, options[0])

                choice = st.selectbox(
                    clean_tag_to_visualize(attr),
                    options,
                    index=options.index(default_val),
                    key=attr
                )

                selected_features.append(choice)

        choice = st.selectbox(
            clean_tag_to_visualize(attr),
            options,
            index=options.index(default_val),
            key=attr
        )

    with right_col:
        # Loading resources
        load_resources()

        st.subheader("🖼️ Generated Image")
        generate_btn = st.button("🎨 Generate Face", type="primary")

        image_placeholder = st.empty()

        if generate_btn:
            with st.spinner("Generating..."):
                predict()
                # YOUR MODEL LOGIC HERE

        if gen_image_key in st.session_state:
            image_placeholder.image(
                st.session_state[gen_image_key],
                caption="Generated Face",
                use_container_width=True
            )

        st.subheader(f"Generated Codes({len(generated_codes)})")
        # show as a scrollable text box
        if len(generated_codes) > 0:
            st.write(
                " ".join(
                    [f"`<CODE_{c:03d}>`" for c in generated_codes]
                )
            )
        else:
            st.write("Press Generate Face button to generate image")


def clean_tag_to_visualize(tag):
    tag = tag.replace("_", " ")
    tag = tag.replace("<", "")
    tag = tag.replace(">", "")
    tag = tag.strip().capitalize()
    return tag


@st.cache_resource()
def load_resources():
    codeBook = CodeBook(codeBook_path)
    print(f"CodeBook Loaded, {codeBook.get_n_tokens()} Tokens")

    decoder_model = ModelEncapsule(decoder_path)
    print(f"VQ-VAE loaded, Output shape: {decoder_model.output_shape}")

    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_ID)
    print(f"Tokenizer Loaded")

    gpt_model = AutoModelForCausalLM.from_pretrained(
        GPT_MODEL_ID,
        torch_dtype="auto",  # safe default
    )
    print(f"GPT-2 Loaded")

    return {
        "tokenizer": tokenizer,
        "gpt_model": gpt_model,
        "decoder_model": decoder_model,
        "codeBook": codeBook,
    }


def predict():
    global generated_codes
    resources_dict = load_resources()
    tokenizer = resources_dict["tokenizer"]
    gpt_model = resources_dict["gpt_model"]
    decoder_model = resources_dict["decoder_model"]
    codeBook = resources_dict["codeBook"]

    prompt = " ".join(
        [TOKEN_START_FACE] +
        selected_features +
        [TOKEN_START_GENERATION]
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(gpt_model.device)

    output_ids = gpt_model.generate(
        **inputs,
        max_new_tokens=MAX_LEN - inputs["input_ids"].shape[1],
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    gpt_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    got_output_codes = re.findall(r'<CODE_(\d+)>', gpt_output)
    got_output_codes = [int(code) for code in got_output_codes]
    generated_codes = list(got_output_codes)

    print("Sequence Out:")
    print(gpt_output)
    print()

    print("Codes:")
    print(*got_output_codes, sep=' ')
    print()

    print("Analysis:")
    print(f"Predicted: {len(got_output_codes):3d} Code")
    print(f"Total:     {256:3d} Code")
    print(f"Missing:   {256 - len(got_output_codes):3d} Code")
    print("Fill missing Code with random values")
    print()

    while len(got_output_codes) < 256:
        got_output_codes.append(random.randint(0, codeBook.get_n_tokens() - 1))

    while len(got_output_codes) > 256:
        got_output_codes.pop()

    code_mat = codeBook.codes_2_embedding_mat(got_output_codes)

    print("Full Codes:")
    print(*got_output_codes, sep=' ')
    print("Latent Shape:", code_mat.shape)

    img_reconstructed = decoder_model.predict(np.expand_dims(code_mat, axis=0))[0]
    print("Image Shape:", img_reconstructed.shape)

    st.session_state[gen_image_key] = img_reconstructed

    return 1


class CodeBook:
    def __init__(self, codeBook_path):
        self.codeBook = pd.read_csv(codeBook_path)
        self.embedding_mat = self.codeBook.iloc[:, 1:].values.astype(np.float32)

        print('-' * 30)
        print("Path:", codeBook_path)
        print("Embedding Mat Shape:", self.embedding_mat.shape)
        print('-' * 30)

    def get_n_tokens(self):
        return self.embedding_mat.shape[0]

    def embedding_mat_2_codes(self, x):
        x = x.astype(np.float32)
        codebook = self.embedding_mat

        # L2 distance
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        e_sq = np.sum(codebook ** 2, axis=1)
        cross = np.dot(x, codebook.T)

        distances = x_sq + e_sq - 2 * cross

        indices = np.argmin(distances, axis=1)

        return indices

    def codes_2_embedding_mat(self, codes):
        """
        codes: (num_tokens,) -> indices of nearest embeddings
        returns: (num_tokens, embedding_dim)
        """

        return self.embedding_mat[codes]


class ModelEncapsule:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)

        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.input_type = self.sess.get_inputs()[0].type
        self.output_name = self.sess.get_outputs()[0].name
        self.output_shape = self.sess.get_outputs()[0].shape
        self.output_type = self.sess.get_outputs()[0].type

        print('-' * 30)
        print("Path:", model_path)
        print("Input Name:", self.input_name)
        print("Input Shape:", self.input_shape)
        print("Input Type:", self.input_type)
        print("Output Name:", self.output_name)
        print("Output Shape:", self.output_shape)
        print("Output Type:", self.output_type)
        print('-' * 30)

    def predict(self, x):
        return self.sess.run([self.output_name], {self.input_name: x})[0]


build_ui()
