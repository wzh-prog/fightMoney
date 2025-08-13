import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ================= 配置 =================
DOCS_DIR = "./docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.title("本地 AI 助手")

# ================= 上传文档 =================
st.header("上传文档 (PDF 或 DOCX)")
uploaded_file = st.file_uploader("选择文件", type=["pdf", "docx"])
if uploaded_file is not None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"文件 {uploaded_file.name} 已上传")

# ================= 加载模型 =================
@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()
if model:
    st.success("模型加载成功！")

# ================= 读取文档内容 =================
def read_doc(file_path):
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif file_path.endswith(".docx"):
            text = docx2txt.process(file_path)
        else:
            text = ""
        return text
    except Exception as e:
        st.error(f"读取文档失败: {e}")
        return ""

content = ""
if uploaded_file is not None:
    content = read_doc(file_path)
    st.text_area("文档内容预览", content, height=300)

# ================= 查询功能 =================
st.header("输入查询")
query = st.text_input("请输入查询内容")

if st.button("查询") and model is not None and uploaded_file is not None:
    if content.strip() == "":
        st.warning("文档为空或读取失败")
    elif query.strip() == "":
        st.warning("请输入查询内容")
    else:
        # 按段落切分文档
        paragraphs = [p for p in content.split("\n") if p.strip()]
        if not paragraphs:
            st.warning("文档没有有效段落")
        else:
            # 文档向量化
            doc_vectors = model.encode(paragraphs, convert_to_numpy=True)
            # 查询向量化
            query_vec = model.encode([query], convert_to_numpy=True)
            # 计算余弦相似度
            sims = cosine_similarity(query_vec, doc_vectors)[0]
            top_idx = np.argmax(sims)
            st.subheader("最相关段落")
            st.write(paragraphs[top_idx])
            st.info(f"相似度: {sims[top_idx]:.4f}")
