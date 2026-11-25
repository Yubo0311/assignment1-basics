MODELS = {
    "gpt2_small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12,
        "d_ff": 2048,  # 8/3 * 768 = 2048
    },
    "gpt2_medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 24,
        "d_model": 1024,
        "num_heads": 16,
        "d_ff": 2752,  # 8/3 * 1024 ≈ 2730.67 -> 43 * 64
    },
    "gpt2_large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 36,
        "d_model": 1280,
        "num_heads": 20,
        "d_ff": 3392,  # 8/3 * 1280 ≈ 3413.33 -> 53 * 64
    },
    "gpt2_xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": 4288,  # 8/3 * 1600 ≈ 4266.67 -> 67 * 64
    },
    "gpt2_xl_16k": {
        "vocab_size": 50257,
        "context_length": 16384,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": 4288,
    },
}
import matplotlib.pyplot as plt
import numpy as np
from wandb import plot


def plot_flops_analysis(models_dict, save_path="flops_analysis.png"):
    names = list(models_dict.keys())
    totals = [calculate_matmul_FLOPs(**cfg) / 1e12 for cfg in models_dict.values()]
    props = [calculate_components_proportion(**cfg) for cfg in models_dict.values()]

    components = ["ffn", "attn.proj", "attn.interaction", "classfication"]
    labels = ["FFN (SwiGLU)", "Attn Proj (QKV+O)", "Attn Interaction (QK^T, AV)", "LM Head"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(names))

    # 左图：总 TFLOPs（对数坐标）
    bars = ax1.bar(x, totals, color="#4C72B0", width=0.55)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Total FLOPs (TFLOPs, log scale)")
    ax1.set_title("Total Forward-Pass FLOPs")
    for bar, val in zip(bars, totals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 右图：各组件 FLOPs 占比（100% 堆叠柱状图）
    bottom = np.zeros(len(names))
    for comp, label, color in zip(components, labels, colors):
        values = np.array([p[comp] * 100 for p in props])
        ax2.bar(x, values, bottom=bottom, label=label, color=color, width=0.55)
        for i, v in enumerate(values):
            if v >= 5:  # 占比大于 5% 才在色块里标数值，避免重叠
                ax2.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        bottom += values

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Proportion of Total FLOPs (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("FLOPs Breakdown by Component")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
def calculate_matmul_FLOPs(vocab_size = 50257,
                  context_length = 1024,
                  num_layers = 48,d_model = 1600,
                  num_heads = 25,d_ff = 4288):
    single_transformer_block =  (
        3*2*context_length*d_model*d_model+
        2*context_length*d_model*context_length+
        2*context_length*context_length*d_model +
          2*context_length*d_model*d_model
          )
    single_transformer_block +=3*2*context_length*d_model*d_ff
    classification_head = 2*context_length*d_model*vocab_size
    return num_layers*single_transformer_block+classification_head


def calculate_components_proportion(vocab_size = 50257,
                  context_length = 1024,
                  num_layers = 48,d_model = 1600,
                  num_heads = 25,d_ff = 4288):
        attn_proj =  num_layers*(
        3*2*context_length*d_model*d_model
          +2*context_length*d_model*d_model)
        attn_interaction = num_layers*(
        2*context_length*d_model*context_length+
        2*context_length*context_length*d_model
          )
        attn = attn_interaction+attn_proj
        ffn = num_layers*3*2*context_length*d_model*d_ff
        classification_head = 2*context_length*d_model*vocab_size
        total = ffn+attn+classification_head
        return {"attn.proj": attn_proj/total,
                "attn.interaction":attn_interaction/total,
                "ffn": ffn/total,
                "classfication":classification_head/total}

if __name__ == "__main__":
    for key, value in MODELS.items():
        print(f"model name: {key}",
                f"total TFLOPs: {calculate_matmul_FLOPs(**value)*1e-12}",
                f"proportion: {calculate_components_proportion(**value)}", sep="\n")
    plot_flops_analysis(MODELS)