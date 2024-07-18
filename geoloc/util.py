import tabulate
import numpy as np

table_format = tabulate.TableFormat(
    lineabove=tabulate.Line("", "-", "  ", ""),
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=None,
    headerrow=tabulate.DataRow("", "  ", ""),
    datarow=tabulate.DataRow("", "  ", ""),
    padding=0,
    with_header_hide=None,
)

def print_state(split, batch, metrics, total_batches=None):
    for k in ["q0", "q1", "q2"]:
        if k in metrics and metrics[k] > 0.99:
            metrics.pop(k)
    if not any([k in metrics for k in ["q0", "q1", "q2"]]):
        for k in ["t-air", "t-pv", "t-collate"]:
            if k in metrics:
                metrics.pop(k)

    reports = []
    reports.append(("Stage", split.capitalize()))
    if total_batches is None:
        reports.append(("Batch", str(batch + 1)))
    else:
        reports.append(("Batch", f"{batch + 1}/{total_batches}"))
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            v = f"{float(v):.6f}"
        else:
            v = str(v)
        reports.append((k, v))
    print(tabulate.tabulate([[v for k, v in reports]], headers=[k for k, v in reports], tablefmt=table_format))

def flatten(tree, separator="//"):
    result = {}
    for ok, ov in tree.items():
        if isinstance(ov, dict):
            for ik, iv in flatten(ov, separator=separator).items():
                result[ok + separator + ik] = iv
        else:
            result[ok] = ov
    return result

def unflatten(values, separator="//"):
    result = {}
    for name, value in values.items():
        keys = name.split(separator)
        node = result
        for k in keys[:-1]:
            if not k in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return result