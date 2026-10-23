"""Hai bài thực hành Word2Vec tiếng Việt. Python 3.12; xem README.md."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from pyvi import ViTokenizer

SEED = 42
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.vi"
N_ARTICLES = 2000
# Chọn trước khi huấn luyện; từ ghép dùng dấu gạch dưới theo Pyvi.
QUERY_WORDS = ["học_sinh", "giáo_viên", "bác_sĩ", "bệnh_viện", "máy_tính",
               "điện_thoại", "ô_tô", "bóng_đá", "âm_nhạc", "toán_học"]
PARAMETERS = dict(vector_size=100, window=5, min_count=5, sg=1,
                  negative=10, hs=0, sample=1e-3, ns_exponent=0.75,
                  alpha=0.025, min_alpha=0.0001, epochs=15,
                  workers=1, seed=SEED, sorted_vocab=1, shrink_windows=True)


def stable_hash(value):
    """Hash cố định, không phụ thuộc PYTHONHASHSEED của tiến trình."""
    return zlib.crc32(value.encode("utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def download_corpus(root, n_articles=N_ARTICLES):
    """Lấy các hàng 0..1999, lưu văn bản cùng URL để truy nguyên nguồn."""
    root = Path(root)
    data_dir = root / "data"
    cache_dir = root.parent / "work/wiki_download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "wiki_vi_2000.jsonl.gz"
    metadata = data_dir / "download_metadata.json"
    if output.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["n_articles"] != n_articles or sha256(output) != saved["sha256"]:
            raise ValueError("Dữ liệu cache không khớp cấu hình hoặc checksum.")
        return output
    rows = []
    for offset in range(0, n_articles, 100):
        length = min(100, n_articles - offset)
        cached = cache_dir / f"page_{offset:05d}.json"
        params = urllib.parse.urlencode(dict(dataset=DATASET, config=CONFIG,
                                             split="train", offset=offset, length=length))
        url = "https://datasets-server.huggingface.co/rows?" + params
        if cached.exists():
            page = json.loads(cached.read_text(encoding="utf-8"))
        else:
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "WordEmbeddingCourse/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        page = json.load(response)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            dump_json(cached, page)
        if len(page["rows"]) != length:
            raise ValueError(f"Thiếu hàng dữ liệu tại offset={offset}.")
        for item in page["rows"]:
            if item.get("truncated_cells"):
                raise ValueError("API trả văn bản bị cắt; không dùng để huấn luyện.")
            rows.append(dict(row_index=item["row_idx"], **item["row"]))
        print(f"Đã tải {len(rows)}/{n_articles} bài", flush=True)
    assert [r["row_index"] for r in rows] == list(range(n_articles))
    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_json(metadata, dict(dataset=DATASET, config=CONFIG, split="train",
                            row_start=0, row_end=n_articles-1, n_articles=len(rows),
                            selection="First 2000 rows; convenience sample, not random",
                            source="https://huggingface.co/datasets/wikimedia/wikipedia",
                            license="CC BY-SA 3.0 / GFDL, per dataset card",
                            sha256=sha256(output)))
    return output


def tokenize_text(text):
    """Giữ dấu tiếng Việt; tách câu gần đúng, tách từ rồi chuyển chữ thường."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for sentence in re.split(r"[.!?;\n]+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = ViTokenizer.tokenize(sentence).lower().split()
        tokens = [w for w in tokens if all(part.isalpha() for part in w.split("_"))]
        if len(tokens) >= 3:
            yield tokens


def prepare_corpus(root):
    root = Path(root)
    raw = root / "data/wiki_vi_2000.jsonl.gz"
    corpus = root / "data/corpus_tokenized.txt"
    info_path = root / "data/corpus_metadata.json"
    if corpus.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["raw_sha256"] == sha256(raw) and info["corpus_sha256"] == sha256(corpus):
            return corpus, info
        raise ValueError("Corpus hoặc dữ liệu nguồn đã thay đổi.")
    started = time.perf_counter()
    n_docs = n_sentences = n_tokens = duplicates = 0
    seen = set()
    with gzip.open(raw, "rt", encoding="utf-8") as f, corpus.open("w", encoding="utf-8", newline="\n") as out:
        for index, line in enumerate(f, 1):
            record = json.loads(line)
            text = unicodedata.normalize("NFC", record["text"])
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            contributed = False
            for tokens in tokenize_text(text):
                out.write(" ".join(tokens) + "\n")
                n_sentences += 1
                n_tokens += len(tokens)
                contributed = True
            n_docs += int(contributed)
            if index % 100 == 0:
                print(f"Tách từ: {index} bài; {n_tokens:,} token", flush=True)
    info = dict(articles_used=n_docs, duplicates_removed=duplicates,
                sentences=n_sentences, tokens_before_min_count=n_tokens,
                raw_sha256=sha256(raw), corpus_sha256=sha256(corpus),
                preprocessing_seconds=round(time.perf_counter()-started, 2),
                tokenizer="pyvi.ViTokenizer 0.1.1", normalization="NFC; lowercase after segmentation",
                sentence_split="Approximate regex: punctuation and newline",
                filtering="URLs, numbers and punctuation removed; at least 3 tokens per sentence",
                stopwords="Kept; frequent-word subsampling handled by Word2Vec")
    dump_json(info_path, info)
    return corpus, info


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.started = time.perf_counter()

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch}/{model.epochs}; {time.perf_counter()-self.started:.1f}s", flush=True)


def train_model(root):
    root = Path(root)
    corpus, corpus_info = prepare_corpus(root)
    np.random.seed(SEED)
    started = time.perf_counter()
    sentences = LineSentence(str(corpus))
    model = Word2Vec(**PARAMETERS, hashfxn=stable_hash)
    model.build_vocab(sentences)
    print(f"Từ vựng: {len(model.wv):,}; kích thước vector: {model.vector_size}", flush=True)
    effective, raw = model.train(sentences, total_examples=model.corpus_count,
                                epochs=model.epochs, callbacks=[EpochLogger()])
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    # Hash không còn cần sau khởi tạo; dùng hàm chuẩn giúp model nạp độc lập.
    model.hashfxn = hash
    model.save(str(model_dir / "word2vec_vi.model"), separately=[])
    model.wv.save(str(model_dir / "word_vectors_vi.kv"), separately=[])
    from importlib.metadata import version
    info = dict(parameters=PARAMETERS, initialization_hash="CRC32 UTF-8",
                vocabulary_size=len(model.wv), vector_shape=list(model.wv.vectors.shape),
                corpus=corpus_info, effective_training_tokens=effective,
                raw_training_tokens=raw, training_seconds=round(model.total_train_time, 2),
                versions=dict(python=sys.version.split()[0], gensim=version("gensim"),
                              numpy=np.__version__, scipy=version("scipy"),
                              sklearn=version("scikit-learn"), matplotlib=version("matplotlib"), pyvi="0.1.1"))
    dump_json(root / "training_metadata.json", info)
    return model, info


def find_pairs(model, queries=QUERY_WORDS):
    """Láng giềng gần nhất theo cosine, tìm trên toàn bộ từ vựng của mô hình."""
    pairs = []
    seen = set()
    for query in queries:
        if query not in model.wv:
            raise KeyError(f"Từ truy vấn ngoài từ vựng: {query}")
        # topn=20 cho phép bỏ cặp đảo chiều đã xuất hiện, nếu có.
        for rank, (neighbor, score) in enumerate(model.wv.most_similar(query, topn=20), 1):
            key = tuple(sorted([query, neighbor]))
            if key not in seen:
                seen.add(key)
                pairs.append(dict(word1=query, word2=neighbor, cosine=float(score),
                                  neighbor_rank=rank,
                                  count1=int(model.wv.get_vecattr(query, "count")),
                                  count2=int(model.wv.get_vecattr(neighbor, "count"))))
                break
    if len(pairs) != 10:
        raise ValueError("Phải tìm đủ 10 cặp không trùng.")
    return pairs


def reduce_dimensions(model, pairs):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    words = list(dict.fromkeys(w for p in pairs for w in [p["word1"], p["word2"]]))
    vectors = np.vstack([model.wv[w] for w in words])
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # Fit một PCA 3 chiều; hai cột đầu chính là phép chiếu PCA 2 chiều.
    pca = PCA(n_components=3, svd_solver="full")
    pca3 = pca.fit_transform(normalized)
    perplexity = min(5.0, len(words)-1.0)
    tsne2 = TSNE(n_components=2, perplexity=perplexity, init="pca",
                 learning_rate="auto", max_iter=1500, random_state=SEED,
                 metric="euclidean").fit_transform(normalized)
    return dict(words=words, original_vectors=vectors, normalized_vectors=normalized,
                pca2=pca3[:, :2], pca3=pca3, tsne2=tsne2,
                explained_variance_ratio=pca.explained_variance_ratio_,
                tsne_perplexity=perplexity, pca=pca)


def make_plots(root, pairs, reduced):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    output = Path(root) / "figures"
    output.mkdir(exist_ok=True)
    words = reduced["words"]
    lookup = {w:i for i,w in enumerate(words)}
    colors = list(plt.colormaps["tab10"].colors)
    ev = reduced["explained_variance_ratio"]
    legend_labels = [f'{i+1:02d}. {p["word1"].replace("_", " ")} / {p["word2"].replace("_", " ")}'
                     for i,p in enumerate(pairs)]
    handles = [Line2D([0],[0],color=colors[i],marker="o",lw=1.5,label=label)
               for i,label in enumerate(legend_labels)]
    for key, title in [("pca2", f"PCA 2D | Giữ {sum(ev[:2]):.1%} phương sai"),
                       ("tsne2", "t-SNE 2D | Perplexity = 5, seed = 42")]:
        points = reduced[key]
        fig, ax = plt.subplots(figsize=(12.5, 8))
        fig.subplots_adjust(left=.09,right=.96,top=.85,bottom=.34)
        fig.suptitle(title, x=.08, y=.96, ha="left", fontsize=20, weight="bold", color="#18344d")
        fig.text(.08,.90,"10 cặp truy vấn và láng giềng | Word2Vec tiếng Việt, 100 chiều",fontsize=11,color="#526273")
        for i,pair in enumerate(pairs):
            ids = [lookup[pair["word1"]],lookup[pair["word2"]]]
            xy = points[ids]
            ax.plot(xy[:,0],xy[:,1],color=colors[i],alpha=.65,lw=1.4)
            ax.scatter(xy[:,0],xy[:,1],color=colors[i],s=60,zorder=3)
            # Nhãn số giữ hình rõ; bảng chú giải xác định từng cặp.
            for j,index in enumerate(ids):
                other = xy[1-j]
                dx = 6 if points[index,0] >= other[0] else -6
                dy = 10 if points[index,1] >= other[1] else -15
                if key == "pca2" and i == 0:
                    dx,dy = (-9,-15) if j == 0 else (-10,12)
                if key == "pca2" and i == 5 and j == 0:
                    dx,dy = 12,16
                ax.annotate(f"{i+1}{'a' if j==0 else 'b'}", points[index],
                            xytext=(dx,dy),textcoords="offset points", fontsize=10,
                            ha="left" if dx > 0 else "right",
                            bbox=dict(boxstyle="round,pad=.13",fc="white",ec="none",alpha=.8))
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})" if key=="pca2" else "t-SNE 1")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})" if key=="pca2" else "t-SNE 2")
        ax.grid(alpha=.2)
        ax.margins(.12)
        fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.07,.065),ncol=2,
                   fontsize=10,frameon=False,columnspacing=3,handlelength=1.7,labelspacing=.65)
        fig.text(.08,.025,"a: từ truy vấn; b: láng giềng. Tính cosine trên 100 chiều, không trên tọa độ hình.",
                 fontsize=9,color="#526273")
        fig.savefig(output / f"{key}.png",dpi=180)
        plt.close(fig)
    fig = plt.figure(figsize=(12.5,8.5))
    ax=fig.add_axes([.02,.26,.96,.64],projection="3d")
    points=reduced["pca3"]
    for i,pair in enumerate(pairs):
        ids=[lookup[pair["word1"]],lookup[pair["word2"]]]
        xyz=points[ids]
        ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],color=colors[i],lw=1.5)
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],color=colors[i],s=50,depthshade=False)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})",labelpad=8)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})",labelpad=8)
    ax.set_zlabel(f"PC3 ({ev[2]:.1%})",labelpad=8)
    ax.view_init(elev=22,azim=38)
    from mpl_toolkits.mplot3d import proj3d
    for i,pair in enumerate(pairs):
        ids=[lookup[pair["word1"]],lookup[pair["word2"]]]
        projected=[proj3d.proj_transform(*points[index],ax.get_proj()) for index in ids]
        for j,(x,y,z) in enumerate(projected):
            other=projected[1-j]
            dx=7 if x >= other[0] else -7
            dy=9 if y >= other[1] else -12
            if (i,j) == (5,0):
                dx,dy=25,5
            elif (i,j) == (6,0):
                dx,dy=22,-7
            elif (i,j) == (8,0):
                dx,dy=-22,-8
            ax.annotate(f"{i+1}{'a' if j==0 else 'b'}",(x,y),xytext=(dx,dy),
                        textcoords="offset points",ha="left" if dx > 0 else "right",
                        fontsize=10,bbox=dict(boxstyle="round,pad=.12",fc="white",ec="none",alpha=.8))
    fig.suptitle(f"PCA 3D | Giữ {sum(ev):.1%} phương sai",x=.08,y=.96,ha="left",
                 fontsize=20,weight="bold",color="#18344d")
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.07,.045),ncol=2,
               fontsize=10,frameon=False,columnspacing=3,handlelength=1.7,labelspacing=.65)
    fig.text(.08,.018,"a: từ truy vấn; b: láng giềng. Phép chiếu làm mất một phần thông tin.",fontsize=9,color="#526273")
    fig.savefig(output / "pca3.png",dpi=180)
    plt.close(fig)


def analyze(root, model=None):
    root=Path(root)
    if model is None:
        model=Word2Vec.load(str(root / "models/word2vec_vi.model"))
    pairs=find_pairs(model)
    reduced=reduce_dimensions(model,pairs)
    results=root / "results"
    results.mkdir(exist_ok=True)
    with (results / "10_cap_tuong_dong.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    with (results / "toa_do_2d_3d.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["word","PCA_PC1","PCA_PC2","PCA_PC3","TSNE_1","TSNE_2"])
        for i,word in enumerate(reduced["words"]):
            writer.writerow([word,*reduced["pca3"][i],*reduced["tsne2"][i]])
    summary=dict(pairs=pairs,words=reduced["words"],
                 explained_variance_ratio=reduced["explained_variance_ratio"].tolist(),
                 pca_fit_scope="Only unique words belonging to the 10 reported pairs; L2 normalized",
                 tsne=dict(perplexity=reduced["tsne_perplexity"],seed=SEED,max_iter=1500),
                 selection="One nearest available neighbor per predefined query; not global top 10")
    dump_json(results / "analysis.json",summary)
    np.savez_compressed(results / "vectors_and_projection.npz",words=reduced["words"],
                        original_vectors=reduced["original_vectors"],
                        pca_mean=reduced["pca"].mean_,pca_components=reduced["pca"].components_,
                        pca2=reduced["pca2"],pca3=reduced["pca3"],tsne2=reduced["tsne2"])
    make_plots(root,pairs,reduced)
    for i,pair in enumerate(pairs,1):
        print(f'{i:2d}. {pair["word1"]:16s} {pair["word2"]:18s} {pair["cosine"]:.4f}')
    print("Phương sai PCA:",reduced["explained_variance_ratio"])
    return pairs,reduced


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--stage",choices=["all","download","train","analyze"],default="all")
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage in ["all","download"]:
        download_corpus(args.root)
    if args.stage in ["all","train"]:
        model,info=train_model(args.root)
    if args.stage in ["all","analyze"]:
        analyze(args.root)


if __name__ == "__main__":
    main()
