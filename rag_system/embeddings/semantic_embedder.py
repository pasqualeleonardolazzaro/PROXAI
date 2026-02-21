from sentence_transformers import SentenceTransformer
from rag_system.config import settings



def compute_text_embeddings(nodes, model_name=settings.TEXT_EMBED_MODEL):
    st = SentenceTransformer(model_name)
    text_emb = {}

    for nid, node in nodes.items():
        node_vecs = []
        node_texts = []

        for field in settings.TEXT_FIELDS:
            if field in node["props"] and node["props"][field]:
                txt = f"{field}: {node['props'][field]}"
                node_texts.append(txt)
                node_vecs.append(st.encode(txt))

        # embed labels separately
        labels_txt = "labels: " + ", ".join(node["labels"])
        node_texts.append(labels_txt)
        node_vecs.append(st.encode(labels_txt))

        text_emb[nid] = {
            "vecs": node_vecs,
            "texts": node_texts
        }

    return text_emb