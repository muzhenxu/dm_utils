from flask import Flask, send_from_directory, redirect, url_for, request
import sys
import os
import uuid
from werkzeug import secure_filename
import base64
import pandas as pd

app = Flask(__name__)

# %%============================================================================ # upload file config #============================================================================== #
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = os.getcwd() + "/data"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
html = '''    <!DOCTYPE html>    <title>Upload File</title>    <link rel="shortcut icon" href="static/favicon.ico">    <h1>文件上传</h1>    <form method=post enctype=multipart/form-data>         <input type=file name=file>         <input type=submit value=上传>    上传的文件必须是以.csv结尾，分割符是','，第一行是列名    </form>    '''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# %%============================================================================ # flask app route #==============================================================================
@app.route(app.config['UPLOAD_FOLDER'] + '/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # filename = str(uuid.uuid4())+'.csv'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            return 'File upload success'
    return html

def fun_facets_overview(file_url):
    gfsg = GenericFeatureStatisticsGenerator()
    train_data = pd.read_csv(file_url, sep = r'\s*,\s*'  ,na_values='?')
    proto = gfsg.ProtoFromDataFrames([{'name': 'train', 'table': train_data}])
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
    HTML_TEMPLATE = """    <link rel="import" href="../facets-jupyter.html" >    <facets-overview id="elem"></facets-overview>    <script>        document.querySelector("#elem").protoInput = "{protostr}";    </script>"""
    html = HTML_TEMPLATE.format(protostr=protostr)
    html_name = "./static/Overview_html/"+str(uuid.uuid4())+".html"
    with open(html_name, "w") as fout:
        fout.write(html)
    return redirect(html_name)

def fun_facets_dive(file_url):
    jsonstr = pd.read_csv(file_url, na_values = '?').to_json(orient='records')
    print('current url:\t', os.getcwd())
    HTML_TEMPLATE = """<link rel="import" href="../facets-jupyter.html">        <facets-dive id="elem" height="600"></facets-dive>        <script>          var data = {jsonstr};          document.querySelector("#elem").data = data;        </script>"""
    html = HTML_TEMPLATE.format(jsonstr = jsonstr)
    html_name = "./static/Dive_html/"+str(uuid.uuid4())+".html"
    with open(html_name, "w") as fout:
        fout.write(html)
    return redirect(html_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(sys.argv[1]), debug=True)