"""Loading and figure helpers shared by probe.py and explain.py."""
import pickle

import pandas
import plotly.express


def load_embeddings(path):
    """Load a SPINE.EmbeddingWriter pickle as a frame indexed by utterance, with integer columns."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return pandas.DataFrame(data["embedding"], index=pandas.Index(data["label"], name="utterance_id"))


def load_labels(path):
    """Load a CSV with `utterance_id` and `group` columns, plus `label` for probing."""
    return pandas.read_csv(path, dtype={"utterance_id": str, "group": str}).set_index("utterance_id")


def write_parallel_figure(rows, columns, path):
    """Write the paper's parallel categories figure, with line width set by each row's `weight`."""
    repeated = rows.loc[rows.index.repeat(rows["weight"])]
    figure = plotly.express.parallel_categories(
            repeated,
            dimensions=columns,
            color=pandas.factorize(repeated["dimension"])[0],
            color_continuous_scale=plotly.express.colors.sequential.Plasma,
            )
    figure.write_html(path, include_plotlyjs="cdn")
