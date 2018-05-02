import pandas as pd
from sklearn.model_selection import train_test_split
from .facets_utils.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
import base64
from IPython.core.display import display, HTML
import os

def facets_overview(proto, mem_control=1000, path='reportsource/overview.html'):
    mem_total = 0
    for dic in proto:
        mem = dic['table'].memory_usage(deep=True).sum() / (1024 ** 2)
        print(dic['name'], 'occupies ', mem, 'M memory')
        mem_total += mem
    print('Whole data occupy %sM memory' % mem_total)
    if mem_total > mem_control:
        print('mem exceeds the threshold %sM' % mem_control)
        return None

    gfsg = GenericFeatureStatisticsGenerator()
    proto = gfsg.ProtoFromDataFrames(proto)
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
    HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-dist/facets-jupyter.html" >
            <facets-overview id="elem"></facets-overview>
            <script>
              document.querySelector("#elem").protoInput = "{protostr}";
            </script>"""
    html = HTML_TEMPLATE.format(protostr=protostr)

    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(path, "w") as fout:
        fout.write(html)

    display(HTML(html))
    return html


def facets_dive(df, mem_control=500, path='reportsource/dive.html'):
    mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print('data occupies ', mem, 'M memory')
    if mem > mem_control:
        print('mem exceeds the threshold %sM' % mem_control)
        return None

    jsonstr = df.to_json(orient='records')
    HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-dist/facets-jupyter.html">
            <facets-dive id="elem" height="600"></facets-dive>
            <script>
              var data = {jsonstr};
              document.querySelector("#elem").data = data;
            </script>"""
    html = HTML_TEMPLATE.format(jsonstr=jsonstr)

    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(path, "w") as fout:
        fout.write(html)

    display(HTML(html))
    return html


if __name__ == '__main__':
    df = pd.read_pickle('test_data/iv_test.pkl')
    train_data, test_data = train_test_split(df, test_size=0.2)
    proto = [{'name': 'train', 'table': train_data},
             {'name': 'test', 'table': test_data}]
    facets_overview(proto)
    facets_dive(df)
